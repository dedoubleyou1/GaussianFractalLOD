"""Train GaussianFractalLOD on Modal cloud GPUs.

Usage:
    modal run modal_train.py                    # Train with defaults
    modal run modal_train.py --max-levels 9     # Override settings
    modal run modal_train.py --resume <path>    # Resume from checkpoint

Use --detach for long runs:
    modal run --detach modal_train.py --max-levels 9
"""

import modal
import os
import time

# Build the Modal image with all dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands("apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("torchmetrics", "lpips", "tensorboard", "numpy", "Pillow",
                 "pyyaml", "huggingface_hub", "ninja", "imageio[ffmpeg]")
    .pip_install("gsplat==1.4.0")  # 1.4 has pre-built CUDA; 1.5+ needs JIT
    .add_local_dir("gaussianfractallod", remote_path="/app/gaussianfractallod", copy=True)
    .add_local_file("setup.py", remote_path="/app/setup.py", copy=True)
    .run_commands("cd /app && pip install -e .")
    # Include all NeRF Synthetic scenes (complete dataset from repo via LFS)
    .add_local_dir("nerf_synthetic/chair", remote_path="/app/nerf_synthetic/chair", copy=True)
    .add_local_dir("nerf_synthetic/drums", remote_path="/app/nerf_synthetic/drums", copy=True)
    .add_local_dir("nerf_synthetic/ficus", remote_path="/app/nerf_synthetic/ficus", copy=True)
    .add_local_dir("nerf_synthetic/hotdog", remote_path="/app/nerf_synthetic/hotdog", copy=True)
    .add_local_dir("nerf_synthetic/lego", remote_path="/app/nerf_synthetic/lego", copy=True)
    .add_local_dir("nerf_synthetic/materials", remote_path="/app/nerf_synthetic/materials", copy=True)
    .add_local_dir("nerf_synthetic/mic", remote_path="/app/nerf_synthetic/mic", copy=True)
    .add_local_dir("nerf_synthetic/ship", remote_path="/app/nerf_synthetic/ship", copy=True)
)

app = modal.App("gaussianfractallod", image=image)

# Persistent volumes
vol = modal.Volume.from_name("gflod-checkpoints", create_if_missing=True)
data_vol = modal.Volume.from_name("gflod-data", create_if_missing=True)


@app.function(
    gpu="L4",
    timeout=86400,  # 24 hours — volumes auto-commit periodically
    volumes={"/checkpoints": vol, "/data": data_vol},
)
def train(
    scene: str = "lego",
    num_roots: int = 1,
    sh_degree: int = 0,
    max_levels: int = 9,
    resume_from: str | None = None,
    auto_eval: bool = True,
    run_name: str | None = None,
    config_overrides: dict | None = None,
):
    import torch
    import os
    import logging
    logging.basicConfig(level=logging.INFO)

    from gaussianfractallod.config import Config
    from gaussianfractallod.train import train as do_train

    # Data is bundled in the image from local repo (complete via Git LFS)
    data_dir = f"/app/nerf_synthetic/{scene}"
    print(f"Dataset at {data_dir}")

    suffix = run_name or f"sh{sh_degree}_l{max_levels}"
    checkpoint_dir = f"/checkpoints/{scene}_{suffix}"

    overrides = config_overrides or {}
    base_args = dict(
        data_dir=data_dir,
        num_roots=num_roots,
        sh_degree=sh_degree,
        max_levels=max_levels,
        checkpoint_dir=checkpoint_dir,
    )
    base_args.update(overrides)  # overrides win
    cfg = Config(**base_args)
    if overrides:
        print(f"Config overrides: {overrides}")

    print(f"Training {scene}: {cfg.num_roots} root, SH{cfg.sh_degree}, {cfg.max_levels} levels")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Checkpoints: {checkpoint_dir}")

    roots, tree = do_train(cfg, resume_from=resume_from)

    print(f"\nTraining complete! {tree.depth} levels")
    for d in range(tree.depth):
        g = tree.get_gaussians_at_depth(d)
        print(f"  Level {d}: {g.num_gaussians} Gaussians")

    vol.commit()

    # Run eval server-side so it works with --detach
    eval_results = None
    if auto_eval:
        print("\n--- Auto-evaluating ---")
        from gaussianfractallod.data import NerfSyntheticDataset
        from gaussianfractallod.eval import evaluate as do_eval
        from gaussianfractallod.checkpoint import save_checkpoint as save_ckpt

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        background = torch.tensor([1.0, 1.0, 1.0], device=device)
        test_dataset = NerfSyntheticDataset(data_dir, split="test")

        eval_results = {}
        for depth in range(tree.depth):
            r = do_eval(tree.to(device), test_dataset, depth, device, background)
            eval_results[depth] = r
            print(f"Depth {depth}: PSNR={r['psnr']:.2f}, {r['num_gaussians']} Gaussians")

        # Save eval results alongside final checkpoint
        from pathlib import Path
        save_ckpt(
            Path(checkpoint_dir) / "eval_results.pt",
            roots, tree, phase=2, level=cfg.max_levels,
            eval=eval_results,
            config_overrides=overrides,
        )
        vol.commit()
        print("Eval results saved to checkpoint.")

    return {"depth": tree.depth, "checkpoint_dir": checkpoint_dir, "eval": eval_results}


@app.function(
    gpu="L4",
    timeout=1800,  # 30 min — eval is faster but needs time for CUDA compile
    volumes={"/checkpoints": vol, "/data": data_vol},
)
def evaluate(
    scene: str = "lego",
    sh_degree: int = 0,
    max_levels: int = 9,
    checkpoint_path: str | None = None,
    run_name: str | None = None,
):
    vol.reload()  # Get latest checkpoints
    import torch
    import glob
    import logging
    logging.basicConfig(level=logging.INFO)

    from gaussianfractallod.config import Config
    from gaussianfractallod.data import NerfSyntheticDataset
    from gaussianfractallod.eval import evaluate as do_eval
    from gaussianfractallod.checkpoint import load_checkpoint

    suffix = run_name or f"sh{sh_degree}_l{max_levels}"
    checkpoint_dir = f"/checkpoints/{scene}_{suffix}"

    if checkpoint_path is None:
        ckpts = sorted(glob.glob(f"{checkpoint_dir}/phase2_level_*.pt"),
                           key=lambda p: int(p.split("_level_")[1].split(".")[0]))
        if not ckpts:
            print(f"No checkpoints found in {checkpoint_dir}")
            return
        checkpoint_path = ckpts[-1]

    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots, tree, meta = load_checkpoint(checkpoint_path, device=device)

    test_dataset = NerfSyntheticDataset(f"/app/nerf_synthetic/{scene}", split="test")
    background = torch.tensor([1.0, 1.0, 1.0], device=device)

    results = {}
    for depth in range(tree.depth):
        r = do_eval(tree.to(device), test_dataset, depth, device, background)
        results[depth] = r
        print(f"Depth {depth}: PSNR={r['psnr']:.2f}, {r['num_gaussians']} Gaussians")

    return results


@app.function(
    gpu="L4",
    timeout=600,
    volumes={"/checkpoints": vol},
)
def analyze_residuals(
    scene: str = "lego",
    sh_degree: int = 0,
    max_levels: int = 9,
    run_name: str | None = None,
):
    """Analyze how far trained Gaussians drift from their initialization."""
    vol.reload()
    import torch
    import glob

    from gaussianfractallod.checkpoint import load_checkpoint

    suffix = run_name or f"sh{sh_degree}_l{max_levels}"
    checkpoint_dir = f"/checkpoints/{scene}_{suffix}"
    ckpts = sorted(glob.glob(f"{checkpoint_dir}/phase2_level_*.pt"),
                           key=lambda p: int(p.split("_level_")[1].split(".")[0]))
    if not ckpts:
        print(f"No checkpoints in {checkpoint_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots, tree, meta = load_checkpoint(ckpts[-1], device=device)

    print(f"{'Level':>6} {'N':>8} {'pos_drift':>10} {'pos/scale':>10} {'scale_drift':>12} {'quat_drift':>11} {'color_drift':>12} {'opacity_drift':>14}")
    print("-" * 95)

    for level_idx in range(1, tree.depth):
        lm = tree.levels[level_idx]
        with torch.no_grad():
            # Position drift (absolute and relative to Gaussian scale)
            pos_drift = (lm.delta_means).norm(dim=-1)  # (N,)
            avg_scale = torch.exp(lm.init_log_scales + lm.delta_log_scales).mean(dim=-1)  # (N,)
            pos_relative = pos_drift / (avg_scale + 1e-8)

            # Scale drift (in log-space)
            scale_drift = (lm.delta_log_scales).abs().mean(dim=-1)  # (N,)

            # Quaternion drift (angle between init and trained)
            # For now approximate: init quats aren't stored, but we know
            # children inherit parent quats. Use dot product as similarity.
            # quat drift = arccos(|dot(q_trained, q_init)|)
            # Since we don't store init_quats, measure deviation from identity-like behavior
            # by checking how far from the parent direction

            # Color drift
            sh_dc = lm.init_sh_dc + lm.delta_sh_dc
            color_drift = (sh_dc - sh_dc.mean()).abs().mean(dim=-1).mean(dim=-1) if hasattr(lm, 'init_sh_dc') else torch.zeros(1)

            # Opacity drift
            init_opacity_est = 0.1  # rough estimate — children start near this after reset
            opacity_drift = lm.delta_opacities.squeeze(-1).abs().mean()

            N = lm.num_gaussians
            print(f"{level_idx:>6} {N:>8} {pos_drift.mean():.4f} {pos_relative.mean():>10.2f}σ {scale_drift.mean():>12.4f} {'N/A':>11} {color_drift.mean():>12.4f} {opacity_drift:>14.4f}")

    # Drift direction analysis: is there a systematic bias?
    print("\n--- Drift Direction Analysis ---")
    for level_idx in range(1, min(tree.depth, 8)):
        lm = tree.levels[level_idx]
        with torch.no_grad():
            drift_vec = lm.delta_means  # (N, 3)
            avg_scale = torch.exp(lm.init_log_scales + lm.delta_log_scales).mean(dim=-1, keepdim=True)  # (N, 1)

            # Normalize drift by scale to get direction in Gaussian-relative units
            drift_norm = drift_vec / (avg_scale + 1e-8)  # (N, 3)

            # Is drift radial (away from parent center)?
            # Compare drift direction to init_position direction (from origin)
            init_dir = lm.init_means / (lm.init_means.norm(dim=-1, keepdim=True) + 1e-8)
            drift_dir = drift_vec / (drift_vec.norm(dim=-1, keepdim=True) + 1e-8)
            radial_component = (drift_dir * init_dir).sum(dim=-1)  # dot product

            # If children from same parent drift together, measure sibling coherence
            if hasattr(lm, 'parent_indices'):
                parent_idx = lm.parent_indices
                unique_parents = parent_idx.unique()
                sibling_coherences = []
                for pid in unique_parents[:100]:  # sample 100 parents
                    mask = parent_idx == pid
                    if mask.sum() < 2:
                        continue
                    sibling_drifts = drift_norm[mask]  # (K, 3)
                    mean_drift = sibling_drifts.mean(dim=0)
                    coherence = mean_drift.norm() / (sibling_drifts.norm(dim=-1).mean() + 1e-8)
                    sibling_coherences.append(coherence.item())

                avg_coherence = sum(sibling_coherences) / max(len(sibling_coherences), 1)
            else:
                avg_coherence = 0.0

            # Mean drift direction (global bias)
            mean_drift = drift_norm.mean(dim=0)

            print(f"Level {level_idx}: mean_drift=({mean_drift[0]:.3f}, {mean_drift[1]:.3f}, {mean_drift[2]:.3f}), "
                  f"radial_bias={radial_component.mean():.3f}, sibling_coherence={avg_coherence:.3f}")

    # Summary: what fraction of parameters are "small residuals"?
    print("\n--- Compressibility Analysis ---")
    for level_idx in range(1, tree.depth):
        lm = tree.levels[level_idx]
        with torch.no_grad():
            pos_drift = (lm.delta_means).norm(dim=-1)
            avg_scale = torch.exp(lm.init_log_scales + lm.delta_log_scales).mean(dim=-1)
            relative = pos_drift / (avg_scale + 1e-8)

            # Fraction of Gaussians that stayed within 1σ of init
            within_1sigma = (relative < 1.0).float().mean().item()
            within_2sigma = (relative < 2.0).float().mean().item()

            scale_change = (lm.delta_log_scales).abs()
            small_scale = (scale_change < 0.5).float().mean().item()  # <50% scale change

            print(f"Level {level_idx}: {within_1sigma:.0%} within 1σ, {within_2sigma:.0%} within 2σ, {small_scale:.0%} small scale change")


@app.function(
    gpu="L4",
    timeout=600,
    volumes={"/checkpoints": vol},
)
def export_plys(
    scene: str = "lego",
    sh_degree: int = 0,
    max_levels: int = 9,
    run_name: str | None = None,
    y_up: bool = False,
) -> list[tuple[str, bytes]]:
    """Export PLY files for all levels. Returns list of (filename, ply_bytes)."""
    vol.reload()  # Get latest checkpoints
    import torch
    import glob
    import io

    from gaussianfractallod.checkpoint import load_checkpoint
    from gaussianfractallod.export_ply import export_ply

    suffix = run_name or f"sh{sh_degree}_l{max_levels}"
    checkpoint_dir = f"/checkpoints/{scene}_{suffix}"
    ckpts = sorted(glob.glob(f"{checkpoint_dir}/phase2_level_*.pt"),
                           key=lambda p: int(p.split("_level_")[1].split(".")[0]))
    if not ckpts:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []

    checkpoint_path = ckpts[-1]
    print(f"Loading: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots, tree, meta = load_checkpoint(checkpoint_path, device=device)

    results = []
    for depth in range(tree.depth):
        with torch.no_grad():
            g = tree.get_gaussians_at_depth(depth)
        filename = f"{scene}_level_{depth}.ply"
        local_path = f"/tmp/{filename}"
        export_ply(g, local_path, sh_degree=sh_degree, y_up=y_up)
        with open(local_path, "rb") as f:
            ply_bytes = f.read()
        results.append((filename, ply_bytes))
        print(f"Exported level {depth}: {g.num_gaussians} G, {len(ply_bytes)/1024:.0f} KB")

    return results


@app.function(
    gpu="L4",
    timeout=600,
    volumes={"/checkpoints": vol},
)
def render_lod_gif(
    scene: str = "lego",
    sh_degree: int = 0,
    max_levels: int = 9,
    run_name: str | None = None,
    test_index: int = 60,
    frame_duration_ms: int = 1000,
    output_size: int = 1024,
) -> bytes:
    """Render each LOD level from a single test camera and return an APNG."""
    vol.reload()
    import torch
    import glob
    import io
    from PIL import Image, ImageDraw, ImageFont

    from gaussianfractallod.checkpoint import load_checkpoint
    from gaussianfractallod.data import NerfSyntheticDataset
    from gaussianfractallod.render import render_gaussians

    suffix = run_name or f"sh{sh_degree}_l{max_levels}"
    checkpoint_dir = f"/checkpoints/{scene}_{suffix}"
    ckpts = sorted(glob.glob(f"{checkpoint_dir}/phase2_level_*.pt"),
                   key=lambda p: int(p.split("_level_")[1].split(".")[0]))
    if not ckpts:
        print(f"No checkpoints found in {checkpoint_dir}")
        return b""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots, tree, meta = load_checkpoint(ckpts[-1], device=device)

    dataset = NerfSyntheticDataset(f"/app/nerf_synthetic/{scene}", split="test")
    gt_rgb, gt_alpha, camera = dataset[test_index]
    cam = {k: v.to(device) if isinstance(v, torch.Tensor) else v
           for k, v in camera.items()}
    background = torch.ones(3, device=device)

    # Try to load a larger font
    font = None
    for font_path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:
            font = ImageFont.truetype(font_path, 36)
            break
        except (OSError, IOError):
            continue

    frames = []

    # Each LOD level (no ground truth)
    with torch.no_grad():
        for depth in range(tree.depth):
            gaussians = tree.get_gaussians_at_depth(depth)
            rendered = render_gaussians(
                gaussians, cam["viewmat"], cam["K"],
                cam["width"], cam["height"], background,
            )
            img_np = (rendered.cpu().numpy() * 255).clip(0, 255).astype("uint8")
            pil_img = Image.fromarray(img_np)
            if pil_img.size != (output_size, output_size):
                pil_img = pil_img.resize((output_size, output_size), Image.LANCZOS)
            draw = ImageDraw.Draw(pil_img)
            label = f"L{depth}: {gaussians.num_gaussians:,} G"
            draw.text((20, 20), label, fill=(255, 0, 0), font=font)
            frames.append(pil_img)
            print(f"Rendered level {depth}: {gaussians.num_gaussians} Gaussians")

    buf = io.BytesIO()
    frames[0].save(
        buf, format="PNG", save_all=True, append_images=frames[1:],
        duration=frame_duration_ms, loop=0,
    )
    print(f"APNG: {len(frames)} frames, {buf.tell() / 1024:.0f} KB")
    return buf.getvalue()


@app.function(
    gpu="L4",
    timeout=1800,
    volumes={"/checkpoints": vol},
)
def render_orbit_videos(
    scene: str = "lego",
    sh_degree: int = 0,
    max_levels: int = 9,
    run_name: str | None = None,
    num_frames: int = 120,
    elevation_deg: float = 30.0,
    radius: float = 4.0,
    output_size: int = 1024,
    fps: int = 30,
) -> list[tuple[str, bytes]]:
    """Render orbit videos for each LOD level. Returns list of (filename, mp4_bytes)."""
    vol.reload()
    import torch
    import glob
    import io
    import math
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from gaussianfractallod.checkpoint import load_checkpoint
    from gaussianfractallod.data import NerfSyntheticDataset
    from gaussianfractallod.render import render_gaussians

    suffix = run_name or f"sh{sh_degree}_l{max_levels}"
    checkpoint_dir = f"/checkpoints/{scene}_{suffix}"
    ckpts = sorted(glob.glob(f"{checkpoint_dir}/phase2_level_*.pt"),
                   key=lambda p: int(p.split("_level_")[1].split(".")[0]))
    if not ckpts:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots, tree, meta = load_checkpoint(ckpts[-1], device=device)

    # Get camera intrinsics from the dataset
    dataset = NerfSyntheticDataset(f"/app/nerf_synthetic/{scene}", split="test")
    _, _, ref_cam = dataset[0]
    fov_x = 2.0 * math.atan(ref_cam["width"] / (2.0 * ref_cam["K"][0, 0].item()))

    # Build intrinsics for output size
    focal = 0.5 * output_size / math.tan(0.5 * fov_x)
    K = torch.tensor([
        [focal, 0, output_size / 2.0],
        [0, focal, output_size / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)

    background = torch.ones(3, device=device)

    # Try to load a larger font
    font = None
    for font_path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:
            font = ImageFont.truetype(font_path, 32)
            break
        except (OSError, IOError):
            continue

    # Generate orbit cameras matching NeRF synthetic convention:
    # Scene is Z-up, cameras orbit in XY plane, looking at origin.
    # Build c2w in NeRF/OpenGL convention, then convert to OpenCV for gsplat.
    elev_rad = math.radians(elevation_deg)
    cameras = []
    for i in range(num_frames):
        azimuth = 2.0 * math.pi * i / num_frames
        # Camera position (Z-up world): orbit in XY, elevated along Z
        cx = radius * math.cos(elev_rad) * math.cos(azimuth)
        cy = radius * math.cos(elev_rad) * math.sin(azimuth)
        cz = radius * math.sin(elev_rad)
        cam_pos = np.array([cx, cy, cz])

        # Look-at: forward points from camera to origin
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 0.0, 1.0])  # Z-up
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        # c2w in NeRF/OpenGL convention: columns = right, up, -forward, pos
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos

        # Convert to w2c with OpenGL→OpenCV flip (same as data.py)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        w2c[1, :] *= -1
        w2c[2, :] *= -1

        cameras.append(torch.tensor(w2c, device=device))

    results = []
    all_level_frames = []

    with torch.no_grad():
        for depth in range(tree.depth):
            gaussians = tree.get_gaussians_at_depth(depth)
            frames = []

            for frame_idx, viewmat in enumerate(cameras):
                rendered = render_gaussians(
                    gaussians, viewmat, K,
                    output_size, output_size, background,
                )
                img_np = (rendered.cpu().numpy() * 255).clip(0, 255).astype("uint8")
                pil_img = Image.fromarray(img_np)
                draw = ImageDraw.Draw(pil_img)
                label = f"L{depth}: {gaussians.num_gaussians:,} G"
                draw.text((20, 20), label, fill=(255, 0, 0), font=font)
                frames.append(pil_img)

            # Encode as MP4 via PIL/imageio
            try:
                import imageio
                buf = io.BytesIO()
                writer = imageio.get_writer(buf, format='mp4', fps=fps,
                                            codec='libx264', quality=8)
                for f in frames:
                    writer.append_data(np.array(f))
                writer.close()
                ext = "mp4"
            except ImportError:
                # Fallback: APNG
                buf = io.BytesIO()
                frames[0].save(buf, format="PNG", save_all=True,
                               append_images=frames[1:],
                               duration=1000 // fps, loop=0)
                ext = "png"

            filename = f"orbit_L{depth:02d}.{ext}"
            results.append((filename, buf.getvalue()))
            all_level_frames.append(frames)
            print(f"Rendered level {depth}: {gaussians.num_gaussians:,} G, "
                  f"{len(frames)} frames, {len(buf.getvalue()) / 1024:.0f} KB")

    # Combined video: all levels back to back, full orbit each
    try:
        import imageio
        combined_frames = []
        for level_frames in all_level_frames:
            combined_frames.extend(level_frames)

        buf = io.BytesIO()
        writer = imageio.get_writer(buf, format='mp4', fps=fps,
                                    codec='libx264', quality=8)
        for f in combined_frames:
            writer.append_data(np.array(f))
        writer.close()
        results.append((f"orbit_combined.mp4", buf.getvalue()))
        print(f"Combined video: {len(combined_frames)} frames, {len(buf.getvalue()) / 1024:.0f} KB")
    except ImportError:
        pass

    return results


@app.function(
    gpu="L4",
    timeout=1800,
    volumes={"/checkpoints": vol},
)
def render_lod_zoom(
    scene: str = "lego",
    sh_degree: int = 0,
    max_levels: int = 9,
    run_name: str | None = None,
    num_frames: int = 600,
    elevation_deg: float = 30.0,
    close_radius: float = 4.0,
    spins: float = 3.0,
    output_size: int = 1024,
    fps: int = 30,
) -> bytes:
    """Render a zoom-in video demonstrating LOD transitions.

    Camera starts far away and slowly moves closer while the model spins.
    At each distance, renders the LOD level whose training resolution
    matches the projected detail level at that distance.
    """
    vol.reload()
    import torch
    import glob
    import io
    import math
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from gaussianfractallod.checkpoint import load_checkpoint
    from gaussianfractallod.data import NerfSyntheticDataset
    from gaussianfractallod.render import render_gaussians
    from gaussianfractallod.train import _get_level_resolution

    suffix = run_name or f"sh{sh_degree}_l{max_levels}"
    checkpoint_dir = f"/checkpoints/{scene}_{suffix}"
    ckpts = sorted(glob.glob(f"{checkpoint_dir}/phase2_level_*.pt"),
                   key=lambda p: int(p.split("_level_")[1].split(".")[0]))
    if not ckpts:
        print(f"No checkpoints found in {checkpoint_dir}")
        return b""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots, tree, meta = load_checkpoint(ckpts[-1], device=device)

    dataset = NerfSyntheticDataset(f"/app/nerf_synthetic/{scene}", split="test")
    _, _, ref_cam = dataset[0]
    fov_x = 2.0 * math.atan(ref_cam["width"] / (2.0 * ref_cam["K"][0, 0].item()))

    focal = 0.5 * output_size / math.tan(0.5 * fov_x)
    K = torch.tensor([
        [focal, 0, output_size / 2.0],
        [0, focal, output_size / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)

    background = torch.ones(3, device=device)

    font = None
    for font_path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:
            font = ImageFont.truetype(font_path, 32)
            break
        except (OSError, IOError):
            continue

    # Compute distance-to-level mapping.
    # Each level was trained at a resolution. The "correct" viewing distance
    # for that level is where the object projects at that resolution.
    # distance = close_radius * (full_res / level_res)
    full_res = 800
    level_distances = {}
    for level in range(tree.depth):
        level_res = _get_level_resolution(level)
        level_distances[level] = close_radius * (full_res / level_res)

    # Far distance: where level 0 is appropriate
    far_radius = level_distances[0]
    # Extrapolate even further for dramatic start
    far_radius *= 1.5

    print(f"Distance range: {far_radius:.1f} (far) → {close_radius:.1f} (close)")
    for level in range(tree.depth):
        print(f"  Level {level}: distance={level_distances[level]:.1f}, "
              f"res={_get_level_resolution(level)}px")

    elev_rad = math.radians(elevation_deg)
    frames = []

    with torch.no_grad():
        for i in range(num_frames):
            t = i / (num_frames - 1)  # 0 to 1

            # Smooth ease-in-out for zoom
            t_smooth = 0.5 * (1 - math.cos(math.pi * t))

            # Interpolate distance (log-space for smooth zoom feel)
            log_far = math.log(far_radius)
            log_close = math.log(close_radius)
            radius = math.exp(log_far * (1 - t_smooth) + log_close * t_smooth)

            # Continuous spin
            azimuth = 2.0 * math.pi * spins * t

            # Pick LOD level based on current distance
            best_level = 0
            for level in range(tree.depth):
                if radius <= level_distances[level]:
                    best_level = level

            # Camera position (Z-up world)
            cx = radius * math.cos(elev_rad) * math.cos(azimuth)
            cy = radius * math.cos(elev_rad) * math.sin(azimuth)
            cz = radius * math.sin(elev_rad)
            cam_pos = np.array([cx, cy, cz])

            # Build c2w in OpenGL/NeRF convention (Z-up)
            forward = -cam_pos / np.linalg.norm(cam_pos)
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, forward)

            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward
            c2w[:3, 3] = cam_pos

            w2c = np.linalg.inv(c2w).astype(np.float32)
            w2c[1, :] *= -1
            w2c[2, :] *= -1
            viewmat = torch.tensor(w2c, device=device)

            gaussians = tree.get_gaussians_at_depth(best_level)
            rendered = render_gaussians(
                gaussians, viewmat, K,
                output_size, output_size, background,
            )
            img_np = (rendered.cpu().numpy() * 255).clip(0, 255).astype("uint8")
            pil_img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil_img)
            label = f"L{best_level}: {gaussians.num_gaussians:,} G"
            draw.text((20, 20), label, fill=(255, 0, 0), font=font)
            frames.append(pil_img)

            if i % 60 == 0:
                print(f"Frame {i}/{num_frames}: d={radius:.1f}, LOD={best_level}")

    import imageio
    buf = io.BytesIO()
    writer = imageio.get_writer(buf, format='mp4', fps=fps,
                                codec='libx264', quality=8)
    for f in frames:
        writer.append_data(np.array(f))
    writer.close()
    print(f"LOD zoom: {len(frames)} frames, {len(buf.getvalue()) / 1024:.0f} KB")
    return buf.getvalue()


@app.local_entrypoint()
def main(
    scene: str = "lego",
    num_roots: int = 1,
    sh_degree: int = 0,
    max_levels: int = 9,
    eval_only: bool = False,
    export: bool = False,
    y_up: bool = False,
    analyze: bool = False,
    gif: bool = False,
    orbit: bool = False,
    lod_zoom: bool = False,
    test_index: int = 60,
    resume: str | None = None,
    run_name: str | None = None,
):
    if lod_zoom:
        print(f"Rendering LOD zoom video...")
        video_bytes = render_lod_zoom.remote(
            scene=scene, sh_degree=sh_degree, max_levels=max_levels,
            run_name=run_name,
        )
        import os
        suffix = run_name or f"sh{sh_degree}_l{max_levels}"
        out_dir = f"exports/{scene}_{suffix}"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/lod_zoom.mp4"
        with open(path, "wb") as f:
            f.write(video_bytes)
        print(f"Saved {path} ({len(video_bytes)/1024:.0f} KB)")
        return

    if orbit:
        print(f"Rendering orbit videos for each LOD level...")
        videos = render_orbit_videos.remote(
            scene=scene, sh_degree=sh_degree, max_levels=max_levels,
            run_name=run_name,
        )
        import os
        suffix = run_name or f"sh{sh_degree}_l{max_levels}"
        out_dir = f"exports/{scene}_{suffix}"
        os.makedirs(out_dir, exist_ok=True)
        for filename, video_bytes in videos:
            path = f"{out_dir}/{filename}"
            with open(path, "wb") as f:
                f.write(video_bytes)
            print(f"Saved {path} ({len(video_bytes)/1024:.0f} KB)")
        return

    if gif:
        print(f"Rendering LOD GIF (test view {test_index})...")
        gif_bytes = render_lod_gif.remote(
            scene=scene, sh_degree=sh_degree, max_levels=max_levels,
            run_name=run_name, test_index=test_index,
        )
        import os
        suffix = run_name or f"sh{sh_degree}_l{max_levels}"
        out_dir = f"exports/{scene}_{suffix}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/lod_r_{test_index}.png"
        with open(out_path, "wb") as f:
            f.write(gif_bytes)
        print(f"Saved {out_path} ({len(gif_bytes)/1024:.0f} KB)")
        return

    if analyze:
        print("Analyzing residuals...")
        analyze_residuals.remote(scene=scene, sh_degree=sh_degree, max_levels=max_levels, run_name=run_name)
        return

    if export:
        print("Exporting PLY files...")
        plys = export_plys.remote(scene=scene, sh_degree=sh_degree, max_levels=max_levels, run_name=run_name, y_up=y_up)
        import os
        out_dir = f"exports/{scene}"
        os.makedirs(out_dir, exist_ok=True)
        for filename, ply_bytes in plys:
            path = f"{out_dir}/{filename}"
            with open(path, "wb") as f:
                f.write(ply_bytes)
            print(f"Saved {path} ({len(ply_bytes)/1024:.0f} KB)")
        print(f"\nExported {len(plys)} PLY files to {out_dir}/")
        return

    if eval_only:
        results = evaluate.remote(scene=scene, sh_degree=sh_degree, max_levels=max_levels, run_name=run_name)
        print("\nResults:", results)
    else:
        result = train.remote(
            scene=scene, num_roots=num_roots,
            sh_degree=sh_degree, max_levels=max_levels,
            resume_from=resume,
            auto_eval=True,
            run_name=run_name,
        )
        print("\nResult:", result)

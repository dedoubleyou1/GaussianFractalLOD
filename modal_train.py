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
                 "pyyaml", "huggingface_hub", "ninja")
    .pip_install("gsplat==1.4.0")  # 1.4 has pre-built CUDA; 1.5+ needs JIT
    .add_local_dir("gaussianfractallod", remote_path="/app/gaussianfractallod", copy=True)
    .add_local_file("setup.py", remote_path="/app/setup.py", copy=True)
    .run_commands("cd /app && pip install -e .")
    # Include lego scene in image (complete dataset from repo via LFS)
    .add_local_dir("nerf_synthetic/lego", remote_path="/app/nerf_synthetic/lego", copy=True)
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
            pos_drift = (lm.means - lm.init_means).norm(dim=-1)  # (N,)
            avg_scale = torch.exp(lm.log_scales).mean(dim=-1)  # (N,)
            pos_relative = pos_drift / (avg_scale + 1e-8)

            # Scale drift (in log-space)
            scale_drift = (lm.log_scales - lm.init_log_scales).abs().mean(dim=-1)  # (N,)

            # Quaternion drift (angle between init and trained)
            # For now approximate: init quats aren't stored, but we know
            # children inherit parent quats. Use dot product as similarity.
            # quat drift = arccos(|dot(q_trained, q_init)|)
            # Since we don't store init_quats, measure deviation from identity-like behavior
            # by checking how far from the parent direction

            # Color drift
            color_drift = (lm.sh_coeffs - lm.sh_coeffs.mean()).abs().mean(dim=-1) if hasattr(lm, 'sh_coeffs') else torch.zeros(1)

            # Opacity drift
            init_opacity_est = 0.1  # rough estimate — children start near this after reset
            opacity_drift = lm.opacities.squeeze(-1).abs().mean()

            N = lm.num_gaussians
            print(f"{level_idx:>6} {N:>8} {pos_drift.mean():.4f} {pos_relative.mean():>10.2f}σ {scale_drift.mean():>12.4f} {'N/A':>11} {color_drift.mean():>12.4f} {opacity_drift:>14.4f}")

    # Drift direction analysis: is there a systematic bias?
    print("\n--- Drift Direction Analysis ---")
    for level_idx in range(1, min(tree.depth, 8)):
        lm = tree.levels[level_idx]
        with torch.no_grad():
            drift_vec = lm.means - lm.init_means  # (N, 3)
            avg_scale = torch.exp(lm.log_scales).mean(dim=-1, keepdim=True)  # (N, 1)

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
            pos_drift = (lm.means - lm.init_means).norm(dim=-1)
            avg_scale = torch.exp(lm.log_scales).mean(dim=-1)
            relative = pos_drift / (avg_scale + 1e-8)

            # Fraction of Gaussians that stayed within 1σ of init
            within_1sigma = (relative < 1.0).float().mean().item()
            within_2sigma = (relative < 2.0).float().mean().item()

            scale_change = (lm.log_scales - lm.init_log_scales).abs()
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
        export_ply(g, local_path, sh_degree=sh_degree)
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


@app.local_entrypoint()
def main(
    scene: str = "lego",
    num_roots: int = 1,
    sh_degree: int = 0,
    max_levels: int = 9,
    eval_only: bool = False,
    export: bool = False,
    analyze: bool = False,
    gif: bool = False,
    test_index: int = 60,
    resume: str | None = None,
    run_name: str | None = None,
):
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
        plys = export_plys.remote(scene=scene, sh_degree=sh_degree, max_levels=max_levels, run_name=run_name)
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

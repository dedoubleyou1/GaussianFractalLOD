"""Train GaussianFractalLOD on Modal cloud GPUs.

Usage:
    modal run modal_train.py                    # Train with defaults
    modal run modal_train.py --max-levels 9     # Override settings
    modal run modal_train.py --resume <path>    # Resume from checkpoint
"""

import modal
import os

# Build the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "gsplat", "torchmetrics", "lpips",
                 "tensorboard", "numpy", "Pillow", "pyyaml")
    .copy_local_dir("gaussianfractallod", "/app/gaussianfractallod")
    .copy_local_dir("nerf_synthetic", "/app/nerf_synthetic")
    .copy_local_file("setup.py", "/app/setup.py")
    .run_commands("cd /app && pip install -e .")
)

app = modal.App("gaussianfractallod", image=image)

# Persistent volume for checkpoints (survives across runs)
vol = modal.Volume.from_name("gflod-checkpoints", create_if_missing=True)


@app.function(
    gpu="L4",  # L4 is cheapest with 24GB VRAM. Use "A100" for faster runs.
    timeout=3600,  # 1 hour max
    volumes={"/checkpoints": vol},
)
def train(
    scene: str = "lego",
    num_roots: int = 1,
    sh_degree: int = 0,
    max_levels: int = 9,
    resume_from: str | None = None,
):
    import torch
    import logging
    logging.basicConfig(level=logging.INFO)

    from gaussianfractallod.config import Config
    from gaussianfractallod.train import train as do_train

    checkpoint_dir = f"/checkpoints/{scene}_sh{sh_degree}_l{max_levels}"

    cfg = Config(
        data_dir=f"/app/nerf_synthetic/{scene}",
        num_roots=num_roots,
        sh_degree=sh_degree,
        max_levels=max_levels,
        checkpoint_dir=checkpoint_dir,
    )

    print(f"Training {scene}: {cfg.num_roots} root, SH{cfg.sh_degree}, {cfg.max_levels} levels")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Checkpoints: {checkpoint_dir}")

    roots, tree = do_train(cfg, resume_from=resume_from)

    print(f"\nTraining complete! {tree.depth} levels")
    for d in range(tree.depth):
        g = tree.get_gaussians_at_depth(d)
        print(f"  Level {d}: {g.num_gaussians} Gaussians")

    # Commit the volume so checkpoints persist
    vol.commit()

    return {"depth": tree.depth, "checkpoint_dir": checkpoint_dir}


@app.function(
    gpu="L4",
    timeout=600,
    volumes={"/checkpoints": vol},
)
def evaluate(
    scene: str = "lego",
    sh_degree: int = 0,
    max_levels: int = 9,
    checkpoint_path: str | None = None,
):
    import torch
    import glob
    import logging
    logging.basicConfig(level=logging.INFO)

    from gaussianfractallod.config import Config
    from gaussianfractallod.data import NerfSyntheticDataset
    from gaussianfractallod.eval import evaluate as do_eval
    from gaussianfractallod.checkpoint import load_checkpoint

    checkpoint_dir = f"/checkpoints/{scene}_sh{sh_degree}_l{max_levels}"

    if checkpoint_path is None:
        ckpts = sorted(glob.glob(f"{checkpoint_dir}/phase2_level_*.pt"))
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

    vol.commit()
    return results


@app.local_entrypoint()
def main(
    scene: str = "lego",
    num_roots: int = 1,
    sh_degree: int = 0,
    max_levels: int = 9,
    eval_only: bool = False,
    resume: str | None = None,
):
    if eval_only:
        results = evaluate.remote(scene=scene, sh_degree=sh_degree, max_levels=max_levels)
        print("\nResults:", results)
    else:
        result = train.remote(
            scene=scene, num_roots=num_roots,
            sh_degree=sh_degree, max_levels=max_levels,
            resume_from=resume,
        )
        print("\nTrain result:", result)

        # Auto-evaluate after training
        print("\n--- Evaluating ---")
        results = evaluate.remote(scene=scene, sh_degree=sh_degree, max_levels=max_levels)
        print("\nEval results:", results)

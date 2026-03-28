"""Train standard 3DGS baseline on Modal using gsplat directly.

Usage:
    modal run modal_baseline.py
    modal run --detach modal_baseline.py --sh-degree 3
"""

import modal

# Reuse the same image as our main training (has gsplat already)
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands("apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("torchmetrics", "lpips", "tensorboard", "numpy", "Pillow",
                 "pyyaml", "huggingface_hub", "ninja")
    .pip_install("gsplat==1.4.0")
    .add_local_dir("gaussianfractallod", remote_path="/app/gaussianfractallod", copy=True)
    .add_local_file("setup.py", remote_path="/app/setup.py", copy=True)
    .run_commands("cd /app && pip install -e .")
    .add_local_dir("nerf_synthetic/lego", remote_path="/app/nerf_synthetic/lego", copy=True)
)

app = modal.App("gflod-baseline", image=image)
vol = modal.Volume.from_name("gflod-checkpoints", create_if_missing=True)


@app.function(
    gpu="L4",
    timeout=86400,
    volumes={"/checkpoints": vol},
)
def train_baseline(
    scene: str = "lego",
    sh_degree: int = 0,
    max_steps: int = 30000,
    init_num_pts: int = 50000,
):
    import logging
    logging.basicConfig(level=logging.INFO)

    from gaussianfractallod.baseline import train_baseline as do_train

    data_dir = f"/app/nerf_synthetic/{scene}"
    output_dir = f"/checkpoints/baseline_{scene}_sh{sh_degree}"

    do_train(
        data_dir=data_dir,
        output_dir=output_dir,
        sh_degree=sh_degree,
        max_steps=max_steps,
        init_num_pts=init_num_pts,
    )

    vol.commit()


@app.local_entrypoint()
def main(
    scene: str = "lego",
    sh_degree: int = 0,
    max_steps: int = 30000,
    init_num_pts: int = 50000,
):
    train_baseline.remote(
        scene=scene,
        sh_degree=sh_degree,
        max_steps=max_steps,
        init_num_pts=init_num_pts,
    )

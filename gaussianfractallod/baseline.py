"""Standard 3DGS baseline using gsplat directly.

Flat (non-hierarchical) Gaussian splatting with standard densification,
pruning, and opacity reset — for comparison against hierarchical approach.
"""

import math
import torch
import torch.nn.functional as F
import random
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

from gaussianfractallod.data import NerfSyntheticDataset
from gaussianfractallod.loss import rendering_loss

logger = logging.getLogger(__name__)


def _init_gaussians(num_points: int, sh_degree: int, device: torch.device) -> dict:
    """Initialize random Gaussians in a unit cube."""
    means = (torch.rand(num_points, 3, device=device) * 2 - 1) * 1.5
    quats = torch.rand(num_points, 4, device=device)
    quats = F.normalize(quats, dim=-1)
    scales = torch.log(torch.full((num_points, 3), 0.02, device=device))
    opacities = torch.logit(torch.full((num_points,), 0.5, device=device))
    num_sh = (sh_degree + 1) ** 2
    sh_coeffs = torch.zeros(num_points, num_sh, 3, device=device)
    sh_coeffs[:, 0, :] = 0.5  # grey default

    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "quats": torch.nn.Parameter(quats),
        "scales": torch.nn.Parameter(scales),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh_coeffs[:, :1, :]),
        "shN": torch.nn.Parameter(sh_coeffs[:, 1:, :]),
    })
    return splats


def train_baseline(
    data_dir: str,
    output_dir: str,
    sh_degree: int = 0,
    max_steps: int = 30000,
    init_num_pts: int = 50000,
):
    """Train standard flat 3DGS."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    background = torch.tensor([1.0, 1.0, 1.0], device=device)

    # Load data at full resolution
    train_dataset = NerfSyntheticDataset(data_dir, split="train", scale=1.0)
    test_dataset = NerfSyntheticDataset(data_dir, split="test", scale=1.0)
    logger.info(f"Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")

    # Initialize Gaussians
    splats = _init_gaussians(init_num_pts, sh_degree, device)
    logger.info(f"Initialized {init_num_pts} Gaussians")

    # Per-parameter optimizers (matching 3DGS)
    optimizers = {
        "means": torch.optim.Adam([splats["means"]], lr=1.6e-4, eps=1e-15),
        "quats": torch.optim.Adam([splats["quats"]], lr=1e-3, eps=1e-15),
        "scales": torch.optim.Adam([splats["scales"]], lr=5e-3, eps=1e-15),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=5e-2, eps=1e-15),
        "sh0": torch.optim.Adam([splats["sh0"]], lr=2.5e-3, eps=1e-15),
        "shN": torch.optim.Adam([splats["shN"]], lr=2.5e-3 / 20, eps=1e-15),
    }

    # Densification strategy (standard 3DGS)
    strategy = DefaultStrategy(verbose=True)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(output_dir) / "logs"))

    for step in range(max_steps):
        # Random training view
        idx = random.randint(0, len(train_dataset) - 1)
        gt_image, camera = train_dataset[idx]
        gt_image = gt_image.to(device)
        viewmat = camera["viewmat"].to(device)
        K = camera["K"].to(device)
        width, height = camera["width"], camera["height"]

        for opt in optimizers.values():
            opt.zero_grad()

        # Combine SH coefficients
        if sh_degree > 0:
            colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        else:
            colors = splats["sh0"]

        renders, alphas, info = rasterization(
            means=splats["means"],
            quats=F.normalize(splats["quats"], dim=-1),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=sh_degree,
        )
        # Add camera dim BEFORE retain_grad so grad is stored on the [C,N,...] tensor
        if "radii" in info and info["radii"].dim() == 1:
            info["radii"] = info["radii"].unsqueeze(0)  # [N] -> [1, N]
        if "means2d" in info and info["means2d"].dim() == 2:
            info["means2d"] = info["means2d"].unsqueeze(0)  # [N, 2] -> [1, N, 2]
        if "means2d" in info:
            info["means2d"].retain_grad()
        info["width"] = width
        info["height"] = height
        info["n_cameras"] = 1

        rendered = renders[0]  # (H, W, 3)
        alpha = alphas[0]  # (H, W, 1)
        rendered = rendered + (1.0 - alpha) * background.view(1, 1, 3)

        loss = rendering_loss(rendered, gt_image, ssim_weight=0.2)

        # Strategy pre-backward
        strategy.step_pre_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        loss.backward()

        # Strategy post-backward (densification, pruning, opacity reset)
        strategy.step_post_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
            packed=False,
        )

        for opt in optimizers.values():
            opt.step()

        # Normalize quaternions
        with torch.no_grad():
            splats["quats"].data = F.normalize(splats["quats"].data, dim=-1)

        writer.add_scalar("train/loss", loss.item(), step)

        if step % 1000 == 0:
            n_gaussians = splats["means"].shape[0]
            logger.info(f"Step {step}: loss={loss.item():.6f}, {n_gaussians} Gaussians")

        # Eval at 7K and 30K (standard 3DGS checkpoints)
        if step in [6999, max_steps - 1]:
            _eval_baseline(splats, test_dataset, sh_degree, device, background, step, writer)
            # Save checkpoint
            torch.save(
                {k: v.data.cpu() for k, v in splats.items()},
                Path(output_dir) / f"splats_step_{step+1}.pt",
            )

    writer.close()
    n_final = splats["means"].shape[0]
    logger.info(f"Training complete. {n_final} Gaussians.")
    return splats


def _eval_baseline(splats, test_dataset, sh_degree, device, background, step, writer):
    """Evaluate on test set."""
    from gaussianfractallod.eval import psnr_fn
    import torchmetrics

    psnr_total = 0.0
    n_images = len(test_dataset)

    with torch.no_grad():
        if sh_degree > 0:
            colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        else:
            colors = splats["sh0"]

        for i in range(n_images):
            gt_image, camera = test_dataset[i]
            gt_image = gt_image.to(device)
            viewmat = camera["viewmat"].to(device)
            K = camera["K"].to(device)
            width, height = camera["width"], camera["height"]

            renders, alphas, _ = rasterization(
                means=splats["means"],
                quats=F.normalize(splats["quats"], dim=-1),
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors,
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                sh_degree=sh_degree,
            )

            rendered = renders[0]
            alpha = alphas[0]
            rendered = rendered + (1.0 - alpha) * background.view(1, 1, 3)

            mse = F.mse_loss(rendered, gt_image)
            psnr = -10.0 * math.log10(mse.item())
            psnr_total += psnr

    avg_psnr = psnr_total / n_images
    n_gaussians = splats["means"].shape[0]
    logger.info(f"Eval step {step+1}: PSNR={avg_psnr:.2f}, {n_gaussians} Gaussians")
    writer.add_scalar("eval/psnr", avg_psnr, step)
    writer.add_scalar("eval/num_gaussians", n_gaussians, step)

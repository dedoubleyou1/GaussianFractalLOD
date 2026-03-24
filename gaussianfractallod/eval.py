"""Evaluation: render test views and compute PSNR, SSIM, LPIPS."""

import torch
import logging
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import GaussianTree
from gaussianfractallod.render import render_gaussians
from gaussianfractallod.data import NerfSyntheticDataset

logger = logging.getLogger(__name__)


def evaluate(
    tree: GaussianTree,
    dataset: NerfSyntheticDataset,
    target_depth: int,
    device: torch.device,
    background: torch.Tensor | None = None,
) -> dict:
    """Evaluate model on a dataset split.

    Args:
        tree: Gaussian tree with trained levels.
        dataset: Evaluation dataset.
        target_depth: Which level to render (0 = roots).
        device: Torch device.
        background: Background color.

    Returns:
        Dict with 'psnr', 'ssim', 'lpips' (mean values).
    """
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    if background is None:
        background = torch.ones(3, device=device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    with torch.no_grad():
        gaussians = tree.get_gaussians_at_depth(target_depth)
        logger.info(f"Rendering {gaussians.num_gaussians} Gaussians at depth {target_depth}")

        for i in range(len(dataset)):
            gt_image, camera = dataset[i]
            gt_image = gt_image.to(device)
            camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera.items()}

            rendered = render_gaussians(
                gaussians,
                viewmat=camera["viewmat"],
                K=camera["K"],
                width=camera["width"],
                height=camera["height"],
                background=background,
            )

            pred_4d = rendered.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            gt_4d = gt_image.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

            psnr_values.append(psnr_metric(pred_4d, gt_4d).item())
            ssim_values.append(ssim_metric(pred_4d, gt_4d).item())
            lpips_values.append(lpips_metric(pred_4d, gt_4d).item())

            if i % 50 == 0:
                logger.info(f"Evaluated {i+1}/{len(dataset)} views")

    results = {
        "psnr": sum(psnr_values) / len(psnr_values),
        "ssim": sum(ssim_values) / len(ssim_values),
        "lpips": sum(lpips_values) / len(lpips_values),
        "num_gaussians": gaussians.num_gaussians,
        "target_depth": target_depth,
    }

    logger.info(
        f"Depth {target_depth}: PSNR={results['psnr']:.2f} "
        f"SSIM={results['ssim']:.4f} LPIPS={results['lpips']:.4f} "
        f"({gaussians.num_gaussians} Gaussians)"
    )

    return results

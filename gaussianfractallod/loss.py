"""Rendering loss: L1 + weighted SSIM."""

import torch
import torch.nn.functional as F


def _ssim_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create Gaussian window for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    return window.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute SSIM between two images.

    Args:
        pred, gt: (H, W, 3) images in [0, 1].

    Returns:
        Scalar SSIM value.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
    gt_4d = gt.permute(2, 0, 1).unsqueeze(0)

    window = _ssim_window().to(pred.device)
    window = window.expand(3, 1, -1, -1)

    mu1 = F.conv2d(pred_4d, window, padding=5, groups=3)
    mu2 = F.conv2d(gt_4d, window, padding=5, groups=3)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_4d ** 2, window, padding=5, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(gt_4d ** 2, window, padding=5, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred_4d * gt_4d, window, padding=5, groups=3) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


def rendering_loss(
    pred: torch.Tensor, gt: torch.Tensor, ssim_weight: float = 0.2,
    alpha_weight: torch.Tensor | None = None, coverage_bias: float = 0.0,
) -> torch.Tensor:
    """Combined L1 + SSIM loss.

    Args:
        alpha_weight: (H, W, 1) GT alpha. When provided with coverage_bias > 0,
            opaque pixels get higher weight: w = 1 + bias * alpha.
        coverage_bias: strength of the opaque-pixel bias (0 = uniform).
    """
    if coverage_bias > 0 and alpha_weight is not None:
        weight = 1.0 + coverage_bias * alpha_weight
        l1 = (weight * (pred - gt).abs()).mean()
    else:
        l1 = F.l1_loss(pred, gt)
    ssim_val = ssim(pred, gt)
    return (1.0 - ssim_weight) * l1 + ssim_weight * (1.0 - ssim_val)

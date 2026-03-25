"""Phase 1: Train root-level Gaussians with standard splatting loss.

Simplification: This prototype uses a fixed root count without adaptive
densification/pruning (unlike standard 3DGS). Roots are initialized
randomly and optimized via gradient descent. Densification can be added
later if root quality is insufficient.
"""

import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.render import render_gaussians
from gaussianfractallod.loss import rendering_loss


def init_roots(
    num_roots: int, sh_degree: int = 0, device: torch.device = torch.device("cpu")
) -> Gaussian:
    """Initialize root Gaussians with random positions near origin.

    Covariance is represented as quaternion + log_scales.
    Initialized as isotropic (identity rotation, scale ~0.37).
    """
    sh_dim = 3 * ((sh_degree + 1) ** 2)

    means = torch.randn(num_roots, 3, device=device) * 0.5
    means.requires_grad_(True)

    # Identity quaternions (wxyz): [1, 0, 0, 0]
    quats = torch.zeros(num_roots, 4, device=device)
    quats[:, 0] = 1.0  # w = 1
    quats.requires_grad_(True)

    # log_scales: log(0.37) ≈ -1.0, isotropic
    log_scales = torch.full((num_roots, 3), -1.0, device=device)
    log_scales.requires_grad_(True)

    opacities = torch.full((num_roots, 1), 2.0, device=device)
    opacities.requires_grad_(True)

    sh_coeffs = torch.randn(num_roots, sh_dim, device=device) * 0.1
    sh_coeffs.requires_grad_(True)

    return Gaussian(means=means, quats=quats, log_scales=log_scales,
                    opacities=opacities, sh_coeffs=sh_coeffs)


def train_roots_step(
    roots: Gaussian,
    gt_image: torch.Tensor,
    camera: dict,
    optimizer: torch.optim.Optimizer,
    ssim_weight: float = 0.2,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single training step for root Gaussians."""
    optimizer.zero_grad()
    rendered = render_gaussians(
        roots, viewmat=camera["viewmat"], K=camera["K"],
        width=camera["width"], height=camera["height"], background=background,
    )
    loss = rendering_loss(rendered, gt_image, ssim_weight=ssim_weight)
    loss.backward()
    optimizer.step()
    return loss.detach()

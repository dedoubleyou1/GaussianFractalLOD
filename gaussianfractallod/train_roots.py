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
    num_roots: int, sh_degree: int = 0, device: torch.device = torch.device("cpu"),
    dataset=None,
) -> Gaussian:
    """Initialize root Gaussians from dataset statistics.

    If a dataset is provided, computes:
    - Position: scene center (average camera look-at point, ~origin for NeRF synthetic)
    - Scale: scene extent from camera positions
    - Color: average color across training views
    - Opacity: high (fully opaque)

    Falls back to random initialization if no dataset is given.
    """
    import math
    num_sh = (sh_degree + 1) ** 2

    if dataset is not None and len(dataset) > 0:
        # Compute scene center from camera positions
        cam_positions = []
        avg_color = torch.zeros(3)
        n_pixels = 0
        for i in range(len(dataset)):
            gt_rgb, gt_alpha, camera = dataset[i]
            # cam_pos = -R^T @ t is the camera origin in world space.
            # Valid for any rigid-body w2c regardless of axis convention.
            w2c = camera["viewmat"]
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            cam_pos = -R.T @ t
            cam_positions.append(cam_pos)
            # Average color (alpha-weighted)
            rgb_masked = gt_rgb * gt_alpha
            avg_color += rgb_masked.sum(dim=(0, 1))
            n_pixels += gt_alpha.sum()

        cam_positions = torch.stack(cam_positions)
        # Scene center: where cameras look at (for NeRF synthetic, near origin)
        # Approximate as the point closest to all camera rays
        scene_center = torch.zeros(3)  # NeRF synthetic is centered at origin

        # Scene extent: radius that covers the object
        # Object is roughly within the camera orbit
        cam_radius = cam_positions.norm(dim=-1).mean()
        # Object extent is roughly 1/4 to 1/3 of camera radius
        scene_extent = cam_radius * 0.3
        log_scale = math.log(max(scene_extent.item(), 0.01))

        # Average color → SH DC coefficient
        avg_color = avg_color / n_pixels.clamp(min=1)
        C0 = 0.28209479177387814
        sh_dc_init = ((avg_color - 0.5) / C0).reshape(1, 1, 3)

        means = scene_center.unsqueeze(0).expand(num_roots, 3).clone()
        if num_roots > 1:
            means += torch.randn(num_roots, 3) * scene_extent * 0.1
        means = means.to(device).requires_grad_(True)

        log_scales = torch.full((num_roots, 3), log_scale, device=device)
        log_scales.requires_grad_(True)

        opacities = torch.full((num_roots, 1), 0.0, device=device)  # sigmoid(0) = 0.5
        opacities.requires_grad_(True)

        sh_dc = sh_dc_init.expand(num_roots, 1, 3).clone().to(device)
        sh_dc.requires_grad_(True)
    else:
        # Fallback: random initialization
        means = torch.randn(num_roots, 3, device=device) * 0.5
        means.requires_grad_(True)

        log_scales = torch.full((num_roots, 3), -1.0, device=device)
        log_scales.requires_grad_(True)

        opacities = torch.full((num_roots, 1), 2.0, device=device)
        opacities.requires_grad_(True)

        sh_dc = torch.randn(num_roots, 1, 3, device=device) * 0.1
        sh_dc.requires_grad_(True)

    # Identity quaternions (wxyz): [1, 0, 0, 0]
    quats = torch.zeros(num_roots, 4, device=device)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)

    sh_rest = torch.zeros(num_roots, num_sh - 1, 3, device=device)
    sh_rest.requires_grad_(True)

    return Gaussian(means=means, quats=quats, log_scales=log_scales,
                    opacities=opacities, sh_dc=sh_dc, sh_rest=sh_rest)


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
        width=camera["width"], height=camera["height"],
        background=background,
    )
    loss = rendering_loss(rendered, gt_image, ssim_weight=ssim_weight)
    loss.backward()
    optimizer.step()
    return loss.detach()

"""Rendering wrapper: auto-selects gsplat (CUDA) or pure PyTorch fallback.

Same API regardless of backend. gsplat is ~50-100× faster but requires CUDA.
The PyTorch fallback works on any device (CPU, MPS, CUDA).
"""

import torch
import math
from gaussianfractallod.gaussian import Gaussian

# Try to import gsplat — may not be available on non-CUDA systems
_GSPLAT_AVAILABLE = False
try:
    from gsplat import rasterization as _gsplat_rasterization
    _GSPLAT_AVAILABLE = True
except ImportError:
    pass


def render_gaussians(
    gaussians: Gaussian,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    background: torch.Tensor | None = None,
    sh_degree: int | None = None,
) -> torch.Tensor:
    """Render Gaussians to an image.

    Auto-selects backend:
      - CUDA + gsplat available → gsplat (fast)
      - Otherwise → pure PyTorch (slow but universal)

    Returns:
        (H, W, 3) rendered image.
    """
    device = gaussians.means.device

    if _GSPLAT_AVAILABLE and device.type == "cuda":
        return _render_gsplat(gaussians, viewmat, K, width, height, background, sh_degree)
    else:
        return _render_pytorch(gaussians, viewmat, K, width, height, background, sh_degree)


def _infer_sh_degree(D: int) -> int:
    if D <= 3:
        return 0
    elif D <= 12:
        return 1
    elif D <= 27:
        return 2
    return 3


def _render_gsplat(
    gaussians: Gaussian,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    background: torch.Tensor | None = None,
    sh_degree: int | None = None,
) -> torch.Tensor:
    """Fast path: gsplat CUDA rasterizer."""
    device = gaussians.means.device

    if background is None:
        background = torch.ones(3, device=device)

    N = gaussians.num_gaussians
    covars = gaussians.covariance()

    D = gaussians.sh_coeffs.shape[-1]
    if sh_degree is None:
        sh_degree = _infer_sh_degree(D)

    num_sh = (sh_degree + 1) ** 2
    expected_dim = num_sh * 3
    assert D >= expected_dim
    sh_coeffs_3 = gaussians.sh_coeffs[:, :expected_dim].reshape(N, num_sh, 3)

    renders, alphas, meta = _gsplat_rasterization(
        means=gaussians.means,
        quats=None,
        scales=None,
        opacities=torch.sigmoid(gaussians.opacities.squeeze(-1)),
        colors=sh_coeffs_3,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=sh_degree,
        covars=covars,
    )

    render_img = renders[0]
    render_alpha = alphas[0]
    render_img = render_img + (1.0 - render_alpha) * background.view(1, 1, 3)
    return render_img


def _render_pytorch(
    gaussians: Gaussian,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    background: torch.Tensor | None = None,
    sh_degree: int | None = None,
) -> torch.Tensor:
    """Fallback: pure PyTorch differentiable splatting.

    Slower than gsplat but works on any device (CPU, MPS, CUDA).
    Implements basic alpha-compositing of 2D projected Gaussians.
    """
    device = gaussians.means.device

    if background is None:
        background = torch.ones(3, device=device)

    N = gaussians.num_gaussians

    # --- Project 3D Gaussians to 2D ---

    # Transform means to camera space
    means_hom = torch.cat([gaussians.means, torch.ones(N, 1, device=device)], dim=-1)  # (N, 4)
    means_cam = (viewmat @ means_hom.T).T[:, :3]  # (N, 3)

    # Depth (z in camera space) — filter Gaussians behind camera
    depths = means_cam[:, 2]  # (N,)
    valid = depths > 0.1
    if not valid.any():
        return background.view(1, 1, 3).expand(height, width, 3)

    # Project to pixel coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    means_2d_x = fx * means_cam[:, 0] / depths + cx  # (N,)
    means_2d_y = fy * means_cam[:, 1] / depths + cy  # (N,)

    # --- Project 3D covariance to 2D ---
    # Jacobian of projection: J = [[fx/z, 0, -fx*x/z²], [0, fy/z, -fy*y/z²]]
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / depths
    J[:, 0, 2] = -fx * means_cam[:, 0] / (depths * depths)
    J[:, 1, 1] = fy / depths
    J[:, 1, 2] = -fy * means_cam[:, 1] / (depths * depths)

    # 3D covariance in camera space: R @ Σ_world @ R.T
    R = viewmat[:3, :3]  # (3, 3)
    cov_3d = gaussians.covariance()  # (N, 3, 3)
    cov_cam = R @ cov_3d @ R.T  # (N, 3, 3) via broadcast

    # 2D covariance: J @ Σ_cam @ J.T
    cov_2d = J @ cov_cam @ J.transpose(-1, -2)  # (N, 2, 2)

    # Add small value for numerical stability
    cov_2d = cov_2d + 0.3 * torch.eye(2, device=device).unsqueeze(0)

    # Inverse of 2D covariance
    det = cov_2d[:, 0, 0] * cov_2d[:, 1, 1] - cov_2d[:, 0, 1] * cov_2d[:, 1, 0]
    det = det.clamp(min=1e-8)

    cov_2d_inv = torch.zeros_like(cov_2d)
    cov_2d_inv[:, 0, 0] = cov_2d[:, 1, 1] / det
    cov_2d_inv[:, 1, 1] = cov_2d[:, 0, 0] / det
    cov_2d_inv[:, 0, 1] = -cov_2d[:, 0, 1] / det
    cov_2d_inv[:, 1, 0] = -cov_2d[:, 1, 0] / det

    # --- Sort by depth (front to back) ---
    sort_idx = depths.argsort()

    # --- Rasterize via alpha compositing ---
    opacities = torch.sigmoid(gaussians.opacities.squeeze(-1))  # (N,)

    # SH degree 0: just use DC color directly
    colors = gaussians.sh_coeffs[:, :3]  # (N, 3) — DC only for simplicity
    # Apply SH0 to RGB conversion: color = 0.5 + C0 * sh_dc
    C0 = 0.28209479177387814
    colors = 0.5 + C0 * colors
    colors = colors.clamp(0, 1)

    # Create pixel grid
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

    # Accumulate: front-to-back alpha compositing
    rendered = torch.zeros(height, width, 3, device=device)
    transmittance = torch.ones(height, width, 1, device=device)

    for idx in sort_idx:
        if not valid[idx]:
            continue
        if transmittance.max() < 1e-4:
            break  # Early exit if fully opaque

        # Compute Gaussian contribution at each pixel
        dx = xx - means_2d_x[idx]  # (H, W)
        dy = yy - means_2d_y[idx]

        # Mahalanobis distance: d = [dx, dy] @ cov_inv @ [dx, dy]
        inv = cov_2d_inv[idx]  # (2, 2)
        maha = (dx * dx * inv[0, 0] + 2 * dx * dy * inv[0, 1] + dy * dy * inv[1, 1])

        # Gaussian falloff
        gauss = torch.exp(-0.5 * maha)  # (H, W)

        # Alpha for this Gaussian
        alpha = (opacities[idx] * gauss).unsqueeze(-1).clamp(max=0.99)  # (H, W, 1)

        # Accumulate
        rendered = rendered + transmittance * alpha * colors[idx].view(1, 1, 3)
        transmittance = transmittance * (1.0 - alpha)

    # Apply background
    rendered = rendered + transmittance * background.view(1, 1, 3)

    return rendered

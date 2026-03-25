"""Rendering wrapper: auto-selects gsplat (CUDA) or tile-based PyTorch fallback.

Same API regardless of backend. gsplat is fastest on CUDA.
The PyTorch fallback uses tile-based rasterization for efficiency
and works on any device (CPU, MPS, CUDA).
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

# Tile size for tile-based rasterization
TILE_SIZE = 16


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
      - Otherwise → tile-based PyTorch (slower but universal)
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
    """Fallback: tile-based pure PyTorch differentiable splatting.

    Uses bounding-box culling per Gaussian so each one only computes
    over the pixels it actually affects (~3σ radius). This is much
    faster than the naive N×H×W approach for small Gaussians.
    """
    device = gaussians.means.device

    if background is None:
        background = torch.ones(3, device=device)

    N = gaussians.num_gaussians

    # --- Project 3D Gaussians to 2D ---
    means_hom = torch.cat([gaussians.means, torch.ones(N, 1, device=device)], dim=-1)
    means_cam = (viewmat @ means_hom.T).T[:, :3]

    depths = means_cam[:, 2]
    valid = depths > 0.1
    if not valid.any():
        return background.view(1, 1, 3).expand(height, width, 3).contiguous()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    means_2d_x = fx * means_cam[:, 0] / depths + cx
    means_2d_y = fy * means_cam[:, 1] / depths + cy

    # --- Project 3D covariance to 2D ---
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / depths
    J[:, 0, 2] = -fx * means_cam[:, 0] / (depths * depths)
    J[:, 1, 1] = fy / depths
    J[:, 1, 2] = -fy * means_cam[:, 1] / (depths * depths)

    R = viewmat[:3, :3]
    cov_3d = gaussians.covariance()
    cov_cam = R @ cov_3d @ R.T

    cov_2d = J @ cov_cam @ J.transpose(-1, -2)
    cov_2d = cov_2d + 0.3 * torch.eye(2, device=device).unsqueeze(0)

    # Inverse of 2D covariance
    det = cov_2d[:, 0, 0] * cov_2d[:, 1, 1] - cov_2d[:, 0, 1] * cov_2d[:, 1, 0]
    det = det.clamp(min=1e-8)

    cov_2d_inv = torch.zeros_like(cov_2d)
    cov_2d_inv[:, 0, 0] = cov_2d[:, 1, 1] / det
    cov_2d_inv[:, 1, 1] = cov_2d[:, 0, 0] / det
    cov_2d_inv[:, 0, 1] = -cov_2d[:, 0, 1] / det
    cov_2d_inv[:, 1, 0] = -cov_2d[:, 1, 0] / det

    # Compute 3σ bounding radius per Gaussian (in pixels)
    # Eigenvalues of 2D covariance give the extent
    # Use trace-based approximation: radius ≈ 3 * sqrt(max_eigenvalue)
    trace = cov_2d[:, 0, 0] + cov_2d[:, 1, 1]
    diff = ((cov_2d[:, 0, 0] - cov_2d[:, 1, 1]) ** 2 + 4 * cov_2d[:, 0, 1] ** 2).sqrt()
    max_eigval = 0.5 * (trace + diff)
    radius = (3.0 * max_eigval.sqrt()).clamp(min=1.0)  # (N,) pixels

    # Sort by depth (front to back)
    sort_idx = depths.argsort()

    # Colors: SH0 DC
    opacities = torch.sigmoid(gaussians.opacities.squeeze(-1))
    C0 = 0.28209479177387814
    colors = (0.5 + C0 * gaussians.sh_coeffs[:, :3]).clamp(0, 1)

    # Pixel grid
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)

    # Accumulate with bounding-box culling
    rendered = torch.zeros(height, width, 3, device=device)
    transmittance = torch.ones(height, width, 1, device=device)

    for idx in sort_idx:
        if not valid[idx]:
            continue
        if transmittance.max() < 1e-4:
            break

        # Bounding box (clipped to image)
        r = int(radius[idx].item()) + 1
        cx_i = means_2d_x[idx]
        cy_i = means_2d_y[idx]
        x_min = max(0, int(cx_i.item()) - r)
        x_max = min(width, int(cx_i.item()) + r + 1)
        y_min = max(0, int(cy_i.item()) - r)
        y_max = min(height, int(cy_i.item()) + r + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        # Local pixel coordinates within bounding box
        local_xs = xs[x_min:x_max]
        local_ys = ys[y_min:y_max]
        local_yy, local_xx = torch.meshgrid(local_ys, local_xs, indexing='ij')

        dx = local_xx - cx_i
        dy = local_yy - cy_i

        # Mahalanobis distance (only within bounding box)
        inv = cov_2d_inv[idx]
        maha = dx * dx * inv[0, 0] + 2 * dx * dy * inv[0, 1] + dy * dy * inv[1, 1]

        gauss = torch.exp(-0.5 * maha)
        alpha = (opacities[idx] * gauss).unsqueeze(-1).clamp(max=0.99)

        # Update only the bounding box region
        local_trans = transmittance[y_min:y_max, x_min:x_max]
        rendered[y_min:y_max, x_min:x_max] += local_trans * alpha * colors[idx].view(1, 1, 3)
        transmittance[y_min:y_max, x_min:x_max] = local_trans * (1.0 - alpha)

    # Apply background
    rendered = rendered + transmittance * background.view(1, 1, 3)

    return rendered

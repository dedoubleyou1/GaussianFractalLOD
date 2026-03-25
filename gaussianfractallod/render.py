"""Rendering wrapper: auto-selects gsplat (CUDA) or tile-based PyTorch fallback.

Same API regardless of backend. gsplat is fastest on CUDA.
The PyTorch fallback uses bounding-box culling for efficiency
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
      - Otherwise → bounding-box PyTorch (slower but universal)
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
    """Fallback: bounding-box PyTorch differentiable splatting.

    Each Gaussian only computes within its 3σ bounding box.
    Transmittance is detached to allow in-place updates while
    still providing gradients through alpha and color.
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

    # 3σ bounding radius per Gaussian
    trace = cov_2d[:, 0, 0] + cov_2d[:, 1, 1]
    diff = ((cov_2d[:, 0, 0] - cov_2d[:, 1, 1]) ** 2 + 4 * cov_2d[:, 0, 1] ** 2).sqrt()
    max_eigval = 0.5 * (trace + diff)
    radius = (3.0 * max_eigval.sqrt()).clamp(min=1.0)

    sort_idx = depths.argsort()

    opacities = torch.sigmoid(gaussians.opacities.squeeze(-1))
    C0 = 0.28209479177387814
    colors = (0.5 + C0 * gaussians.sh_coeffs[:, :3]).clamp(0, 1)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij',
    )

    # Two-pass approach:
    # Pass 1: compute per-Gaussian alpha maps (with bounding box) — builds grad graph
    # Pass 2: composite front-to-back with detached transmittance — no in-place issue

    # Collect per-Gaussian contributions
    contributions = []  # list of (y_min, y_max, x_min, x_max, alpha_local, color_idx)

    for idx in sort_idx:
        if not valid[idx]:
            continue

        r = int(radius[idx].detach().item()) + 1
        cx_i = int(means_2d_x[idx].detach().item())
        cy_i = int(means_2d_y[idx].detach().item())
        x_min = max(0, cx_i - r)
        x_max = min(width, cx_i + r + 1)
        y_min = max(0, cy_i - r)
        y_max = min(height, cy_i + r + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        local_dx = xx[y_min:y_max, x_min:x_max] - means_2d_x[idx]
        local_dy = yy[y_min:y_max, x_min:x_max] - means_2d_y[idx]

        inv = cov_2d_inv[idx]
        maha = (local_dx * local_dx * inv[0, 0]
                + 2 * local_dx * local_dy * inv[0, 1]
                + local_dy * local_dy * inv[1, 1])

        gauss = torch.exp(-0.5 * maha)
        local_alpha = (opacities[idx] * gauss).clamp(max=0.99)  # (h, w)

        contributions.append((y_min, y_max, x_min, x_max, local_alpha, idx))

    # Pass 2: composite using full-image sparse contributions
    # Each Gaussian produces a full-image contribution (zero outside bbox)
    # Transmittance tracked separately (detached) for weighting
    transmittance = torch.ones(height, width, device=device)

    # Accumulate weighted contributions into a list, then sum
    all_contributions = []

    for y_min, y_max, x_min, x_max, local_alpha, idx in contributions:
        local_t = transmittance[y_min:y_max, x_min:x_max].clone()

        # Create full-image contribution (sparse — only bbox is nonzero)
        weight = local_t * local_alpha  # (h, w) — has gradient through local_alpha
        full_contrib = torch.zeros(height, width, 3, device=device)
        full_contrib[y_min:y_max, x_min:x_max] = (
            weight.unsqueeze(-1) * colors[idx].view(1, 1, 3)
        )
        all_contributions.append(full_contrib)

        # Update transmittance (detached, in-place ok)
        with torch.no_grad():
            transmittance[y_min:y_max, x_min:x_max] = (
                local_t * (1.0 - local_alpha.detach())
            )

    # Sum all contributions + background
    if all_contributions:
        rendered = torch.stack(all_contributions).sum(dim=0)
    else:
        rendered = torch.zeros(height, width, 3, device=device)

    rendered = rendered + transmittance.unsqueeze(-1) * background.view(1, 1, 3)

    return rendered

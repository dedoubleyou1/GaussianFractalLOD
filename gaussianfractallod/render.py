"""gsplat rendering wrapper: Gaussian batch -> rendered image."""

import torch
import torch.nn.functional as F
from gsplat import rasterization
from gaussianfractallod.gaussian import Gaussian


def render_gaussians(
    gaussians: Gaussian,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    background: torch.Tensor | None = None,
    sh_degree: int | None = None,
) -> torch.Tensor:
    """Render Gaussians to an image using gsplat.

    Passes quaternions and scales directly to gsplat, which is
    faster than passing covariance matrices.
    """
    device = gaussians.means.device

    if background is None:
        background = torch.ones(3, device=device)

    N = gaussians.num_gaussians

    # Normalized quaternions and scales for gsplat
    quats = F.normalize(gaussians.quats, dim=-1)  # (N, 4) wxyz
    scales = gaussians.scales()  # (N, 3)

    # Infer SH degree from coefficient count
    D = gaussians.sh_coeffs.shape[-1]
    if sh_degree is None:
        if D <= 3:
            sh_degree = 0
        elif D <= 12:
            sh_degree = 1
        elif D <= 27:
            sh_degree = 2
        else:
            sh_degree = 3

    # Reshape SH coeffs for gsplat: (N, num_sh, 3)
    num_sh = (sh_degree + 1) ** 2
    expected_dim = num_sh * 3
    assert gaussians.sh_coeffs.shape[-1] >= expected_dim, (
        f"SH coeffs dim {gaussians.sh_coeffs.shape[-1]} < expected {expected_dim} "
        f"for sh_degree={sh_degree}"
    )
    sh_coeffs_3 = gaussians.sh_coeffs[:, :expected_dim].reshape(N, num_sh, 3)

    renders, alphas, meta = rasterization(
        means=gaussians.means,
        quats=quats,
        scales=scales,
        opacities=torch.sigmoid(gaussians.opacities.squeeze(-1)),
        colors=sh_coeffs_3,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=sh_degree,
    )

    # Apply background manually
    render_img = renders[0]   # (H, W, 3)
    render_alpha = alphas[0]  # (H, W, 1)
    render_img = render_img + (1.0 - render_alpha) * background.view(1, 1, 3)

    return render_img

"""gsplat rendering wrapper: Gaussian batch → rendered image."""

import torch
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

    Passes covariance matrices (Σ = L @ L.T) directly to gsplat,
    which is fully differentiable — gradients flow cleanly back
    to the Cholesky factor L_flat without eigendecomposition.
    """
    device = gaussians.means.device

    if background is None:
        background = torch.ones(3, device=device)

    N = gaussians.num_gaussians

    # Compute covariance directly from Cholesky: Σ = L @ L.T
    covars = gaussians.covariance()  # (N, 3, 3)

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

    # Apply background manually
    render_img = renders[0]   # (H, W, 3)
    render_alpha = alphas[0]  # (H, W, 1)
    render_img = render_img + (1.0 - render_alpha) * background.view(1, 1, 3)

    return render_img

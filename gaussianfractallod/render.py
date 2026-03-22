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

    Args:
        gaussians: Batch of N Gaussians.
        viewmat: (4, 4) world-to-camera matrix.
        K: (3, 3) camera intrinsics.
        width, height: Image dimensions.
        background: (3,) background color. Defaults to white.
        sh_degree: SH degree for color. If None, inferred from sh_coeffs.

    Returns:
        (H, W, 3) rendered image.
    """
    device = gaussians.means.device

    if background is None:
        background = torch.ones(3, device=device)

    N = gaussians.num_gaussians

    # gsplat expects quaternions (N, 4) for rotation. We use identity
    # (axis-aligned) since the split tree operates in parent local frames
    # via diagonal scale approximation. Full rotation support can be added
    # later if the per-axis variance approximation proves insufficient.
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0  # identity quaternion

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
        scales=torch.exp(gaussians.scales),
        opacities=torch.sigmoid(gaussians.opacities.squeeze(-1)),
        colors=sh_coeffs_3,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=sh_degree,
        backgrounds=background.unsqueeze(0),
    )

    return renders[0]  # (H, W, 3)

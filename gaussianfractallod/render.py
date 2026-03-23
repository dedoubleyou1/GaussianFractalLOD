"""gsplat rendering wrapper: Gaussian batch → rendered image."""

import torch
from gsplat import rasterization
from gaussianfractallod.gaussian import Gaussian


def _covariance_to_quat_scale(L_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Cholesky L_flat (N, 6) to quaternion (N, 4) + scale (N, 3).

    Decomposes Σ = L @ L.T into R @ diag(s²) @ R.T via eigendecomposition,
    then converts R to quaternion.
    """
    N = L_flat.shape[0]
    device = L_flat.device

    # Build L matrix
    L = torch.zeros(N, 3, 3, device=device, dtype=L_flat.dtype)
    L[:, 0, 0] = torch.exp(L_flat[:, 0])
    L[:, 1, 0] = L_flat[:, 1]
    L[:, 1, 1] = torch.exp(L_flat[:, 2])
    L[:, 2, 0] = L_flat[:, 3]
    L[:, 2, 1] = L_flat[:, 4]
    L[:, 2, 2] = torch.exp(L_flat[:, 5])

    # Covariance
    cov = L @ L.transpose(-1, -2)

    # Eigendecomposition: cov = V @ diag(eigenvalues) @ V.T
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=1e-8)

    # Scales = sqrt(eigenvalues)
    scales = torch.sqrt(eigvals)  # (N, 3)

    # Rotation matrix to quaternion
    quats = _rotation_matrix_to_quaternion(eigvecs)  # (N, 4)

    return quats, scales


def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 3) rotation matrices to (N, 4) quaternions [w, x, y, z].

    Ensures proper rotation (det=+1) and numerical stability.
    """
    # Ensure proper rotation (det > 0)
    det = torch.linalg.det(R)
    R = R * det.sign().unsqueeze(-1).unsqueeze(-1)

    batch_size = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    quat = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    s = torch.sqrt((trace + 1.0).clamp(min=1e-8)) * 2  # s = 4*w
    mask = trace > 0
    if mask.any():
        quat[mask, 0] = 0.25 * s[mask]
        quat[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
        quat[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
        quat[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

    # Case 2: R[0,0] is largest diagonal
    mask2 = ~mask & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s2 = torch.sqrt((1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]).clamp(min=1e-8)) * 2
        quat[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
        quat[mask2, 1] = 0.25 * s2
        quat[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
        quat[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    # Case 3: R[1,1] is largest diagonal
    mask3 = ~mask & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s3 = torch.sqrt((1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]).clamp(min=1e-8)) * 2
        quat[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
        quat[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
        quat[mask3, 2] = 0.25 * s3
        quat[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    # Case 4: R[2,2] is largest diagonal
    mask4 = ~mask & ~mask2 & ~mask3
    if mask4.any():
        s4 = torch.sqrt((1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]).clamp(min=1e-8)) * 2
        quat[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
        quat[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
        quat[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
        quat[mask4, 3] = 0.25 * s4

    # Normalize
    quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)

    return quat


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

    Decomposes Cholesky-factored covariance into quaternion + scale
    for gsplat's rasterizer.
    """
    device = gaussians.means.device

    if background is None:
        background = torch.ones(3, device=device)

    N = gaussians.num_gaussians

    # Decompose Cholesky L into quaternion + scale for gsplat
    quats, scales = _covariance_to_quat_scale(gaussians.L_flat)

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

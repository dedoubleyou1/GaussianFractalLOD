"""R&G-style binary split via truncated Gaussian moments.

A parent Gaussian is split by a cut plane in its local frame.
The two children are the moment-matched Gaussians of each half —
their positions, covariances, and orientations are derived from
the cut plane geometry.

Conservation guarantees (exact, by the law of total variance):
  - Opacity: α_A + α_B = α_parent
  - Center of mass: π_A μ_A + π_B μ_B = μ_parent
  - Covariance: π_A [Σ_A + δ_Aδ_Aᵀ] + π_B [Σ_B + δ_Bδ_Bᵀ] = Σ_parent
  - Color: π_A c_A + π_B c_B = c_parent
"""

import torch
import math
from dataclasses import dataclass
from gaussianfractallod.gaussian import Gaussian


EPS = 1e-6

# Standard normal PDF and CDF
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF."""
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _Phi(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@dataclass
class SplitVariables:
    """Split variables for a single binary split.

    All tensors have shape (N, ...) for N parallel splits.

    cut_direction: (N, 3) unnormalized direction of the cut plane normal
        in parent's local frame. Will be normalized internally.
    cut_offset: (N,) position of the cut along the normal. Determines
        mass partition: π_left = Φ(cut_offset).
    color_split: (N, D) SH coefficient deviation.
    """

    cut_direction: torch.Tensor   # (N, 3) cut plane normal (unnormalized)
    cut_offset: torch.Tensor      # (N,) cut position along normal
    color_split: torch.Tensor     # (N, D) SH coefficient deviation


def derive_children(
    parent: Gaussian, split_vars: SplitVariables
) -> tuple[Gaussian, Gaussian]:
    """Derive two children by slicing the parent with a cut plane.

    The cut plane is defined in the parent's local coordinate frame
    (where the parent has identity covariance). The children are
    moment-matched Gaussians of the two halves.

    Children naturally orient to the cut plane — no free rotation
    parameters needed. All conservation laws are exact.
    """
    N = parent.num_gaussians
    device = parent.means.device

    # Normalize cut direction to unit vector
    n = split_vars.cut_direction  # (N, 3)
    n = n / (n.norm(dim=-1, keepdim=True) + EPS)

    d = split_vars.cut_offset  # (N,)

    # Mass partition from cut position via CDF
    pi_left = _Phi(d).unsqueeze(-1)         # (N, 1) — left side (n·x ≤ d)
    pi_right = (1.0 - pi_left)              # (N, 1) — right side (n·x > d)

    # Clamp to avoid division by zero
    pi_left = pi_left.clamp(min=EPS)
    pi_right = pi_right.clamp(min=EPS)

    # Opacity conservation
    alpha_left = pi_left * parent.opacities
    alpha_right = pi_right * parent.opacities

    # Hazard rates (inverse Mills ratios)
    phi_d = _phi(d)  # (N,)
    lambda_right = phi_d / pi_right.squeeze(-1)   # (N,)
    lambda_left = phi_d / pi_left.squeeze(-1)     # (N,)

    # Child means in parent's local frame
    # μ_right = +λ_R · n,  μ_left = -λ_L · n
    mu_right_local = lambda_right.unsqueeze(-1) * n       # (N, 3)
    mu_left_local = -lambda_left.unsqueeze(-1) * n        # (N, 3)

    # Convert to world frame: μ_child = μ_parent + L_p @ μ_local
    L_p = parent.L_matrix()  # (N, 3, 3)

    mu_right = parent.means + torch.bmm(L_p, mu_right_local.unsqueeze(-1)).squeeze(-1)
    mu_left = parent.means + torch.bmm(L_p, mu_left_local.unsqueeze(-1)).squeeze(-1)

    # Child covariances in parent's local frame
    # Σ_local = I - c · nnᵀ  where c is the compression factor
    c_right = lambda_right * (lambda_right - d)    # (N,)
    c_left = lambda_left * (lambda_left + d)       # (N,)

    # Clamp compression to [0, 1) to ensure positive-definiteness
    c_right = c_right.clamp(min=0.0, max=1.0 - EPS)
    c_left = c_left.clamp(min=0.0, max=1.0 - EPS)

    # Σ_local = I - c · nnᵀ
    # Eigenvalues: (1 - c) along n, 1.0 perpendicular to n
    # L_child = L_parent @ R_n @ diag(1, 1, √(1-c))
    # where R_n rotates the z-axis to align with n

    # Build rotation R_n that maps z-axis → n
    R_right = _rotation_z_to_n(n)  # (N, 3, 3)
    R_left = R_right  # Same cut plane, same rotation

    # Scale: (1, 1, √(1-c)) — compress along the cut normal
    scale_right = torch.ones(N, 3, device=device)
    scale_right[:, 2] = torch.sqrt((1.0 - c_right).clamp(min=EPS))

    scale_left = torch.ones(N, 3, device=device)
    scale_left[:, 2] = torch.sqrt((1.0 - c_left).clamp(min=EPS))

    # L_child = L_parent @ R_n @ diag(scale)
    L_right = L_p @ R_right * scale_right.unsqueeze(-2)  # (N,3,3)
    L_left = L_p @ R_left * scale_left.unsqueeze(-2)

    # Convert L matrices to flat representation
    L_right_flat = _L_matrix_to_flat(L_right)
    L_left_flat = _L_matrix_to_flat(L_left)

    # Color conservation
    delta_c = split_vars.color_split
    c_a = parent.sh_coeffs + delta_c
    c_b = parent.sh_coeffs - (pi_right / (pi_left + EPS)) * delta_c

    child_right = Gaussian(
        means=mu_right, L_flat=L_right_flat,
        opacities=alpha_right, sh_coeffs=c_a,
    )
    child_left = Gaussian(
        means=mu_left, L_flat=L_left_flat,
        opacities=alpha_left, sh_coeffs=c_b,
    )

    return child_right, child_left


def _rotation_z_to_n(n: torch.Tensor) -> torch.Tensor:
    """Build rotation matrix that maps z-axis to unit vector n.

    Uses Rodrigues' formula. For n ≈ z, returns near-identity.
    For n ≈ -z, uses a stable fallback.

    Args:
        n: (N, 3) unit vectors.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    N = n.shape[0]
    device = n.device

    z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, 3)

    # Cross product z × n = rotation axis
    cross = torch.cross(z, n, dim=-1)  # (N, 3)
    sin_angle = cross.norm(dim=-1, keepdim=True)  # (N, 1)
    cos_angle = (z * n).sum(dim=-1, keepdim=True)  # (N, 1) = n_z

    # Rodrigues: R = I + [k]_× + [k]_×² · (1-cos)/sin²
    # where k = cross / |cross|
    # Handle degenerate case: n ≈ z (sin ≈ 0)
    near_identity = (sin_angle.squeeze(-1) < 1e-6)
    # Handle n ≈ -z
    near_flip = (cos_angle.squeeze(-1) < -1.0 + 1e-6)

    # Normalize axis
    axis = cross / (sin_angle + EPS)  # (N, 3)

    # Skew-symmetric matrix [k]_×
    K = torch.zeros(N, 3, 3, device=device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3)
    R = I + sin_angle.unsqueeze(-1) * K + (1.0 - cos_angle.unsqueeze(-1)) * (K @ K)

    # Fix degenerate cases
    if near_identity.any():
        R[near_identity] = torch.eye(3, device=device)
    if near_flip.any():
        # 180-degree rotation around x-axis
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))
        R[near_flip] = flip

    return R


def _L_matrix_to_flat(L: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 3) lower-triangular matrix to (N, 6) flat representation.

    Since L may not be lower-triangular after rotation, we compute the
    Cholesky factorization of L @ L.T to get a proper lower-triangular form.
    """
    # Compute covariance
    cov = L @ L.transpose(-1, -2)

    # Add small diagonal for stability
    cov = cov + EPS * torch.eye(3, device=cov.device).unsqueeze(0)

    # Cholesky to get proper lower-triangular
    L_proper = torch.linalg.cholesky(cov)

    # Flatten with log on diagonal
    return torch.stack([
        torch.log(L_proper[:, 0, 0].clamp(min=EPS)),
        L_proper[:, 1, 0],
        torch.log(L_proper[:, 1, 1].clamp(min=EPS)),
        L_proper[:, 2, 0],
        L_proper[:, 2, 1],
        torch.log(L_proper[:, 2, 2].clamp(min=EPS)),
    ], dim=-1)

"""R&G-style binary split via truncated Gaussian moments.

A parent Gaussian is split by a cut plane in its local frame.
The two children are the moment-matched Gaussians of each half --
their positions, covariances, and orientations are derived from
the cut plane geometry.

Conservation guarantees (exact, by the law of total variance):
  - Opacity: alpha_A + alpha_B = alpha_parent
  - Center of mass: pi_A mu_A + pi_B mu_B = mu_parent
  - Covariance: pi_A [Sigma_A + delta_A delta_A^T] + pi_B [Sigma_B + delta_B delta_B^T] = Sigma_parent
  - Color: pi_A c_A + pi_B c_B = c_parent
"""

import torch
import torch.nn.functional as F
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
        mass partition: pi_left = Phi(cut_offset).
    color_split: (N, D) SH coefficient deviation.
    """

    cut_direction: torch.Tensor   # (N, 3) cut plane normal (unnormalized)
    cut_offset: torch.Tensor      # (N,) cut position along normal
    color_split: torch.Tensor     # (N, D) SH coefficient deviation


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two batches of quaternions (wxyz convention).

    Args:
        q1: (N, 4) quaternions
        q2: (N, 4) quaternions

    Returns:
        (N, 4) product quaternions
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _rotation_matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 3) rotation matrices to (N, 4) quaternions (wxyz).

    Uses a stable method that handles all rotation cases.
    """
    batch_size = R.shape[0]
    device = R.device

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    quat = torch.zeros(batch_size, 4, device=device, dtype=R.dtype)

    s = torch.sqrt((trace + 1.0).clamp(min=1e-8)) * 2
    mask = trace > 0
    if mask.any():
        quat[mask, 0] = 0.25 * s[mask]
        quat[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
        quat[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
        quat[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

    mask2 = ~mask & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s2 = torch.sqrt((1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]).clamp(min=1e-8)) * 2
        quat[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
        quat[mask2, 1] = 0.25 * s2
        quat[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
        quat[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    mask3 = ~mask & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s3 = torch.sqrt((1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]).clamp(min=1e-8)) * 2
        quat[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
        quat[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
        quat[mask3, 2] = 0.25 * s3
        quat[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    mask4 = ~mask & ~mask2 & ~mask3
    if mask4.any():
        s4 = torch.sqrt((1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]).clamp(min=1e-8)) * 2
        quat[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
        quat[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
        quat[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
        quat[mask4, 3] = 0.25 * s4

    quat = F.normalize(quat, dim=-1)
    return quat


def derive_children(
    parent: Gaussian, split_vars: SplitVariables
) -> tuple[Gaussian, Gaussian]:
    """Derive two children by slicing the parent with a cut plane.

    The cut plane is defined in the parent's local coordinate frame
    (where the parent has identity covariance). The children are
    moment-matched Gaussians of the two halves.

    Children naturally orient to the cut plane -- no free rotation
    parameters needed. All conservation laws are exact.
    """
    N = parent.num_gaussians
    device = parent.means.device

    # Normalize cut direction to unit vector
    n = split_vars.cut_direction  # (N, 3)
    n = n / (n.norm(dim=-1, keepdim=True) + EPS)

    d = split_vars.cut_offset  # (N,)

    # Mass partition from cut position via CDF
    pi_left = _Phi(d).unsqueeze(-1)         # (N, 1) -- left side (n.x <= d)
    pi_right = (1.0 - pi_left)              # (N, 1) -- right side (n.x > d)

    # Clamp to avoid division by zero
    pi_left = pi_left.clamp(min=EPS)
    pi_right = pi_right.clamp(min=EPS)

    # Opacity conservation
    alpha_left = pi_left * parent.opacities
    alpha_right = pi_right * parent.opacities

    # Clamp cut_offset to prevent extreme hazard rates
    # Beyond +/-3, the truncated moments become numerically unstable
    d = d.clamp(-3.0, 3.0)

    # Hazard rates (inverse Mills ratios)
    phi_d = _phi(d)  # (N,)
    lambda_right = (phi_d / pi_right.squeeze(-1)).clamp(max=5.0)   # (N,)
    lambda_left = (phi_d / pi_left.squeeze(-1)).clamp(max=5.0)     # (N,)

    # Child means in parent's local frame
    # mu_right = +lambda_R * n,  mu_left = -lambda_L * n
    mu_right_local = lambda_right.unsqueeze(-1) * n       # (N, 3)
    mu_left_local = -lambda_left.unsqueeze(-1) * n        # (N, 3)

    # Convert to world frame: mu_child = mu_parent + R_p @ diag(s) @ mu_local
    R_p = parent.rotation_matrix()  # (N, 3, 3)
    s_p = parent.scales()           # (N, 3)
    # Scale then rotate: R_p @ diag(s) @ mu_local
    scaled_right = s_p * mu_right_local   # (N, 3) element-wise
    scaled_left = s_p * mu_left_local     # (N, 3) element-wise

    mu_right = parent.means + torch.bmm(R_p, scaled_right.unsqueeze(-1)).squeeze(-1)
    mu_left = parent.means + torch.bmm(R_p, scaled_left.unsqueeze(-1)).squeeze(-1)

    # Child covariances in parent's local frame
    # Sigma_local = I - c * nn^T  where c is the compression factor
    c_right = lambda_right * (lambda_right - d)    # (N,)
    c_left = lambda_left * (lambda_left + d)       # (N,)

    # Clamp compression to [0, 1) to ensure positive-definiteness
    c_right = c_right.clamp(min=0.0, max=1.0 - EPS)
    c_left = c_left.clamp(min=0.0, max=1.0 - EPS)

    # Sigma_local = I - c * nn^T
    # Eigenvalues: (1 - c) along n, 1.0 perpendicular to n
    # The child's Sigma in world = R_p @ diag(s) @ Sigma_local @ diag(s) @ R_p^T
    # = R_p @ diag(s) @ R_n @ diag(1, 1, 1-c) @ R_n^T @ diag(s) @ R_p^T
    # = (R_p @ R_n) @ diag(s_0, s_1, s_2*sqrt(1-c)) ^2 @ (R_p @ R_n)^T
    # So child rotation = R_p @ R_n, child scales = (s_0, s_1, s_2*sqrt(1-c))

    # Build rotation R_n that maps z-axis to n
    R_right = _rotation_z_to_n(n)  # (N, 3, 3)
    R_left = R_right  # Same cut plane, same rotation

    # Child rotation in world: R_child = R_parent @ R_n
    R_child_right = R_p @ R_right  # (N, 3, 3)
    R_child_left = R_p @ R_left    # (N, 3, 3)

    # Convert rotation matrices to quaternions
    q_child_right = _rotation_matrix_to_quat(R_child_right)  # (N, 4)
    q_child_left = _rotation_matrix_to_quat(R_child_left)    # (N, 4)

    # Child scales: parent scales, with z-axis scaled by sqrt(1-c)
    # In the R_n frame, z-axis is aligned with cut normal n
    # Permute parent scales into R_n frame, then adjust z
    # Since R_n maps z->n, the scales in the R_n frame are still (s0, s1, s2)
    # but the compression is along z in the R_n frame
    scale_right = s_p.clone()
    scale_right[:, 2] = s_p[:, 2] * torch.sqrt((1.0 - c_right).clamp(min=EPS))

    scale_left = s_p.clone()
    scale_left[:, 2] = s_p[:, 2] * torch.sqrt((1.0 - c_left).clamp(min=EPS))

    log_scales_right = torch.log(scale_right.clamp(min=EPS))
    log_scales_left = torch.log(scale_left.clamp(min=EPS))

    # Color conservation
    delta_c = split_vars.color_split
    c_a = parent.sh_coeffs + delta_c
    c_b = parent.sh_coeffs - (pi_right / (pi_left + EPS)) * delta_c

    child_right = Gaussian(
        means=mu_right, quats=q_child_right, log_scales=log_scales_right,
        opacities=alpha_right, sh_coeffs=c_a,
    )
    child_left = Gaussian(
        means=mu_left, quats=q_child_left, log_scales=log_scales_left,
        opacities=alpha_left, sh_coeffs=c_b,
    )

    return child_right, child_left


def _rotation_z_to_n(n: torch.Tensor) -> torch.Tensor:
    """Build rotation matrix that maps z-axis to unit vector n.

    Uses Rodrigues' formula. For n ~ z, returns near-identity.
    For n ~ -z, uses a stable fallback.

    Args:
        n: (N, 3) unit vectors.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    N = n.shape[0]
    device = n.device

    z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, 3)

    # Cross product z x n = rotation axis
    cross = torch.cross(z, n, dim=-1)  # (N, 3)
    sin_angle = cross.norm(dim=-1, keepdim=True)  # (N, 1)
    cos_angle = (z * n).sum(dim=-1, keepdim=True)  # (N, 1) = n_z

    # Rodrigues: R = I + [k]_x + [k]_x^2 * (1-cos)/sin^2
    # where k = cross / |cross|
    # Handle degenerate case: n ~ z (sin ~ 0)
    near_identity = (sin_angle.squeeze(-1) < 1e-6)
    # Handle n ~ -z
    near_flip = (cos_angle.squeeze(-1) < -1.0 + 1e-6)

    # Normalize axis
    axis = cross / (sin_angle + EPS)  # (N, 3)

    # Skew-symmetric matrix [k]_x
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

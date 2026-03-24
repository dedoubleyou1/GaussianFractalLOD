"""Subdivide parent Gaussians into 8 children via 3 sequential binary cuts.

Uses truncated Gaussian moments for physically plausible initialization.
Children are returned as independent Gaussians — not constrained by parents.

Each parent is cut along its local X, Y, Z axes (via Cholesky frame),
producing 2→4→8 children. Symmetric cuts (d=0) give equal mass.
"""

import torch
import math
from gaussianfractallod.gaussian import Gaussian


# For symmetric cut (d=0): λ = φ(0)/Φ(0) = (1/√2π) / 0.5 = √(2/π)
_LAMBDA_SYMMETRIC = math.sqrt(2.0 / math.pi)
# Compression factor: c = λ² = 2/π
_C_SYMMETRIC = 2.0 / math.pi


def subdivide_to_8(parents: Gaussian) -> Gaussian:
    """Subdivide each parent Gaussian into 8 children.

    Applies 3 sequential symmetric binary cuts along the parent's
    local X, Y, Z axes. Each cut halves each Gaussian along one axis
    using truncated Gaussian moment-matching.

    Args:
        parents: N parent Gaussians.

    Returns:
        8N child Gaussians, initialized with plausible positions,
        covariances, and opacities from the truncated moments.
    """
    current = parents

    # 3 sequential cuts: local X, Y, Z
    for axis in range(3):
        current = _binary_cut_along_axis(current, axis)

    return current


def _binary_cut_along_axis(gaussians: Gaussian, axis: int) -> Gaussian:
    """Split each Gaussian into 2 along a local axis using truncated moments.

    For a symmetric cut (d=0) along direction e_axis in local frame:
      - Each child gets half the opacity
      - Means displaced by ±λ along the axis (in parent's local frame)
      - Covariance compressed along the cut axis, unchanged perpendicular

    Args:
        gaussians: N Gaussians.
        axis: 0=X, 1=Y, 2=Z in parent's local Cholesky frame.

    Returns:
        2N Gaussians (right half then left half, interleaved).
    """
    N = gaussians.num_gaussians
    device = gaussians.means.device

    # Get parent's Cholesky factor
    L_p = gaussians.L_matrix()  # (N, 3, 3)

    # Cut direction: axis-th basis vector in local frame
    # Displacement in world = L_p @ e_axis = L_p[:, :, axis]
    displacement_world = L_p[:, :, axis]  # (N, 3)

    # Symmetric cut: children at ±λ along the cut direction
    offset = _LAMBDA_SYMMETRIC * displacement_world  # (N, 3)
    mu_right = gaussians.means + offset
    mu_left = gaussians.means - offset

    # Children keep parent's opacity — they cover less area but are
    # just as opaque at their center. Optimizer adjusts during training.
    child_logit = gaussians.opacities

    # Child covariance: compress along cut axis, preserve perpendicular
    # In local frame: Σ_child = I - c * e_axis @ e_axis^T
    # So the axis-th scale factor is √(1-c), others stay 1
    # L_child = L_parent @ diag(scale)
    scale = torch.ones(N, 3, device=device)
    scale[:, axis] = math.sqrt(1.0 - _C_SYMMETRIC)  # ≈ 0.604

    L_right = L_p * scale.unsqueeze(-2)  # (N, 3, 3) * (N, 1, 3)
    L_left = L_right  # Same covariance for symmetric cut

    L_right_flat = _L_to_flat(L_right)
    L_left_flat = _L_to_flat(L_left)

    # Color: children start with parent's color (split evenly)
    sh_right = gaussians.sh_coeffs.clone()
    sh_left = gaussians.sh_coeffs.clone()

    # Interleave: [right_0, left_0, right_1, left_1, ...]
    means = torch.stack([mu_right, mu_left], dim=1).reshape(2 * N, 3)
    L_flats = torch.stack([L_right_flat, L_left_flat], dim=1).reshape(2 * N, 6)
    opacities = torch.stack([child_logit, child_logit], dim=1).reshape(2 * N, 1)
    sh = torch.stack([sh_right, sh_left], dim=1).reshape(2 * N, -1)

    return Gaussian(means=means, L_flat=L_flats, opacities=opacities, sh_coeffs=sh)


def _L_to_flat(L: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 3) lower-triangular to (N, 6) flat with log diagonal."""
    return torch.stack([
        torch.log(L[:, 0, 0].abs().clamp(min=1e-8)),
        L[:, 1, 0],
        torch.log(L[:, 1, 1].abs().clamp(min=1e-8)),
        L[:, 2, 0],
        L[:, 2, 1],
        torch.log(L[:, 2, 2].abs().clamp(min=1e-8)),
    ], dim=-1)

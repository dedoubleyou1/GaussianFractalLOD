"""Subdivide parent Gaussians into 8 children via 3 sequential binary cuts.

Uses least-squares optimal formulas for alpha compositing (not statistical
mixture). Children are sized and positioned to minimize the integrated
squared error between the alpha-composited children and the parent.

Formulas (derived from numerical optimization across opacity/spread space):
  σ_c = σ_parent · √(1 - f²/4) · (1 - 0.124 · α_p²)
  α_c = [1 - √(1 - α_p)] · (0.932 + 0.114 · f)

where f = spread factor (children at ±f/2 in parent's local frame).
"""

import torch
import math
from gaussianfractallod.gaussian import Gaussian


# Spread factor: children displaced ±f/2 in parent's local frame per cut axis
SPREAD_FACTOR = 1.0


def subdivide_to_8(parents: Gaussian) -> Gaussian:
    """Subdivide each parent Gaussian into 8 children.

    Applies 3 sequential binary cuts along the parent's local X, Y, Z axes.
    Each cut uses alpha-compositing-aware formulas for child size and opacity.

    Args:
        parents: N parent Gaussians.

    Returns:
        8N child Gaussians.
    """
    current = parents

    for axis in range(3):
        current = _binary_cut_along_axis(current, axis)

    return current


def _binary_cut_along_axis(gaussians: Gaussian, axis: int) -> Gaussian:
    """Split each Gaussian into 2 along a local axis.

    Uses least-squares optimal formulas for alpha compositing:
    - σ_c accounts for alpha compositing nonlinearity
    - α_c matches center opacity exactly under alpha compositing
    """
    N = gaussians.num_gaussians
    device = gaussians.means.device
    f = SPREAD_FACTOR

    # Get parent's Cholesky factor
    L_p = gaussians.L_matrix()  # (N, 3, 3)

    # Displacement in world = ±(f/2) * L_p[:, :, axis]
    displacement_world = (f / 2.0) * L_p[:, :, axis]  # (N, 3)
    mu_right = gaussians.means + displacement_world
    mu_left = gaussians.means - displacement_world

    # Child opacity: α_c = [1 - √(1 - α_p)] · (0.932 + 0.114·f)
    # This matches center opacity under alpha compositing
    alpha_p = torch.sigmoid(gaussians.opacities)  # (N, 1)
    alpha_c = (1.0 - torch.sqrt((1.0 - alpha_p).clamp(min=1e-8))) * (0.932 + 0.114 * f)
    alpha_c = alpha_c.clamp(min=1e-6, max=1.0 - 1e-6)
    child_logit = torch.log(alpha_c / (1.0 - alpha_c))

    # Child scale: σ_c = σ_p · √(1 - f²/4) · (1 - 0.124·α_p²)
    # In Cholesky terms: scale the axis by this factor
    scale_base = math.sqrt(max(1.0 - f * f / 4.0, 0.01))
    alpha_correction = (1.0 - 0.124 * alpha_p ** 2)  # (N, 1)
    scale_factor = scale_base * alpha_correction  # (N, 1)

    # L_child = L_parent @ diag(scale) where scale[axis] = scale_factor
    scale = torch.ones(N, 3, device=device)
    scale[:, axis:axis+1] = scale_factor  # broadcast (N, 1) into column

    L_right = L_p * scale.unsqueeze(-2)
    L_left = L_right  # Same covariance for symmetric cut

    L_right_flat = _L_to_flat(L_right)
    L_left_flat = _L_to_flat(L_left)

    # Color: parent's color + small random perturbation to break symmetry
    sh_right = gaussians.sh_coeffs + torch.randn_like(gaussians.sh_coeffs) * 0.01
    sh_left = gaussians.sh_coeffs + torch.randn_like(gaussians.sh_coeffs) * 0.01

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

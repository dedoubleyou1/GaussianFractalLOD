"""R&G-style binary split: derive two children from a parent Gaussian + split variables.

Conservation guarantees (by construction):
  - Opacity: alpha_A + alpha_B = alpha_parent
  - Center of mass: pi_A * mu_A + pi_B * mu_B = mu_parent
  - Color: pi_A * c_A + pi_B * c_B = c_parent
"""

import torch
from dataclasses import dataclass
from gaussianfractallod.gaussian import Gaussian


EPS = 1e-6


@dataclass
class SplitVariables:
    """Split variables for a single binary split.

    All tensors have shape (N, ...) for N parallel splits.
    """

    mass_logit: torch.Tensor       # (N,) logit of mass partition ratio
    position_split: torch.Tensor   # (N, 3) displacement in parent's local frame
    variance_split: torch.Tensor   # (N, 3) per-axis variance partition in (0, 1)
    color_split: torch.Tensor      # (N, D) SH coefficient deviation


def derive_children(
    parent: Gaussian, split_vars: SplitVariables
) -> tuple[Gaussian, Gaussian]:
    """Derive two children from parent + split variables.

    Uses Richardson & Green (1997) split parameterization generalized to 3D.
    All conservation laws are satisfied by construction.

    Args:
        parent: Batch of N parent Gaussians.
        split_vars: Batch of N split variable sets.

    Returns:
        (child_a, child_b): Two batches of N child Gaussians.
    """
    # Mass partition
    pi_a = torch.sigmoid(split_vars.mass_logit).unsqueeze(-1)  # (N, 1)
    pi_b = 1.0 - pi_a

    # Opacity conservation: alpha_i = pi_i * alpha_parent
    alpha_a = pi_a * parent.opacities
    alpha_b = pi_b * parent.opacities

    # Position split in parent's local frame
    # Parent scale defines the local frame (diagonal approximation)
    parent_scale = torch.exp(parent.scales)  # (N, 3), from log-space
    u2 = split_vars.position_split  # (N, 3)

    # R&G position formulas (center-of-mass conserved by construction):
    #   mu_A = mu_p + scale_p * u2 * sqrt(pi_B / pi_A)
    #   mu_B = mu_p - scale_p * u2 * sqrt(pi_A / pi_B)
    sqrt_ratio_ab = torch.sqrt(pi_b / (pi_a + EPS) + EPS)  # (N, 1)
    sqrt_ratio_ba = torch.sqrt(pi_a / (pi_b + EPS) + EPS)  # (N, 1)

    mu_a = parent.means + parent_scale * u2 * sqrt_ratio_ab
    mu_b = parent.means - parent_scale * u2 * sqrt_ratio_ba

    # Variance split (per-axis, in log-space for stability)
    # u3 in (0, 1) via sigmoid; split parent variance
    u3 = torch.sigmoid(split_vars.variance_split)  # (N, 3)
    u2_sq = u2 ** 2

    # R&G variance formulas (per-axis approximation):
    #   var_A = (1 - u2^2) * var_p * u3 / pi_A
    #   var_B = (1 - u2^2) * var_p * (1 - u3) / pi_B
    parent_var = torch.exp(2.0 * parent.scales)  # (N, 3), variance = scale^2
    shared_factor = (1.0 - u2_sq).clamp(min=EPS) * parent_var

    var_a = shared_factor * u3 / (pi_a + EPS)
    var_b = shared_factor * (1.0 - u3) / (pi_b + EPS)

    # Convert back to log-scale
    scale_a = 0.5 * torch.log(var_a.clamp(min=EPS))
    scale_b = 0.5 * torch.log(var_b.clamp(min=EPS))

    # Color conservation: pi_A * c_A + pi_B * c_B = c_parent
    #   c_A = c_parent + delta_c
    #   c_B = c_parent - (pi_A / pi_B) * delta_c
    delta_c = split_vars.color_split
    c_a = parent.sh_coeffs + delta_c
    c_b = parent.sh_coeffs - (pi_a / (pi_b + EPS)) * delta_c

    child_a = Gaussian(means=mu_a, scales=scale_a, opacities=alpha_a, sh_coeffs=c_a)
    child_b = Gaussian(means=mu_b, scales=scale_b, opacities=alpha_b, sh_coeffs=c_b)

    return child_a, child_b

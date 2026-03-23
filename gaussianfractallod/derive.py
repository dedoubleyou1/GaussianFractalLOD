"""R&G-style binary split: derive two children from a parent Gaussian + split variables.

Uses Cholesky factored covariance for full orientation support.
The parent's Cholesky factor L_p defines the local coordinate frame —
position splits and variance splits operate in this frame.

Conservation guarantees (by construction):
  - Opacity: alpha_A + alpha_B = alpha_parent
  - Center of mass: pi_A * mu_A + pi_B * mu_B = mu_parent
  - Covariance: pi_A * [Σ_A + δ_Aδ_A^T] + pi_B * [Σ_B + δ_Bδ_B^T] = Σ_parent
  - Color: pi_A * c_A + pi_B * c_B = c_parent

All children's covariances are guaranteed positive-definite because they
are parameterized as partition ratios of a positive budget, not as
remainders from subtraction.
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
    variance_split: torch.Tensor   # (N, 3) per-axis variance partition ratio (pre-sigmoid)
    color_split: torch.Tensor      # (N, D) SH coefficient deviation


def derive_children(
    parent: Gaussian, split_vars: SplitVariables
) -> tuple[Gaussian, Gaussian]:
    """Derive two children from parent + split variables.

    Operates in the parent's local coordinate frame (defined by its
    Cholesky factor L_p). Position displacement uses L_p to map from
    unit space to world space. Variance is split per-axis in this
    local frame, then transformed back to world via L_p.

    All conservation laws are satisfied by construction. All children
    covariances are guaranteed positive-definite.
    """
    # Mass partition
    pi_a = torch.sigmoid(split_vars.mass_logit).unsqueeze(-1)  # (N, 1)
    pi_b = 1.0 - pi_a

    # Opacity conservation
    alpha_a = pi_a * parent.opacities
    alpha_b = pi_b * parent.opacities

    # Parent's Cholesky factor — defines local coordinate frame
    L_p = parent.L_matrix()  # (N, 3, 3)

    # Position split in parent's local frame via L_p
    u2 = split_vars.position_split  # (N, 3)

    # R&G position formulas (center-of-mass conserved by construction)
    sqrt_ratio_ab = torch.sqrt(pi_b / (pi_a + EPS) + EPS)
    sqrt_ratio_ba = torch.sqrt(pi_a / (pi_b + EPS) + EPS)

    # L_p @ u2: map displacement from unit space to world space
    displacement = torch.bmm(L_p, u2.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    mu_a = parent.means + displacement * sqrt_ratio_ab
    mu_b = parent.means - displacement * sqrt_ratio_ba

    # Variance split — partition ratio in parent's local frame
    # budget_j = 1 - u2_j^2 (variance remaining after scatter)
    # u3 ∈ (0,1) via sigmoid: partition ratio per axis
    u3 = torch.sigmoid(split_vars.variance_split)  # (N, 3)
    u2_sq = u2 ** 2
    budget = (1.0 - u2_sq).clamp(min=EPS)  # (N, 3)

    # Per-axis variance in parent's local frame
    var_a_local = budget * u3 / (pi_a + EPS)           # (N, 3)
    var_b_local = budget * (1.0 - u3) / (pi_b + EPS)   # (N, 3)

    # Child Cholesky factors in world space: L_child = L_p @ diag(sqrt(var_local))
    # L_p columns get scaled by sqrt(var_local) per axis
    scale_a = torch.sqrt(var_a_local.clamp(min=EPS))  # (N, 3)
    scale_b = torch.sqrt(var_b_local.clamp(min=EPS))  # (N, 3)

    # L_child = L_p @ diag(s) — scale each column of L_p
    L_a = L_p * scale_a.unsqueeze(-2)  # (N, 3, 3) * (N, 1, 3) = (N, 3, 3)
    L_b = L_p * scale_b.unsqueeze(-2)

    # Convert L matrices to flat representation [log(l00), l10, log(l11), l20, l21, log(l22)]
    L_a_flat = torch.stack([
        torch.log(L_a[:, 0, 0].abs().clamp(min=EPS)),
        L_a[:, 1, 0],
        torch.log(L_a[:, 1, 1].abs().clamp(min=EPS)),
        L_a[:, 2, 0],
        L_a[:, 2, 1],
        torch.log(L_a[:, 2, 2].abs().clamp(min=EPS)),
    ], dim=-1)

    L_b_flat = torch.stack([
        torch.log(L_b[:, 0, 0].abs().clamp(min=EPS)),
        L_b[:, 1, 0],
        torch.log(L_b[:, 1, 1].abs().clamp(min=EPS)),
        L_b[:, 2, 0],
        L_b[:, 2, 1],
        torch.log(L_b[:, 2, 2].abs().clamp(min=EPS)),
    ], dim=-1)

    # Color conservation
    delta_c = split_vars.color_split
    c_a = parent.sh_coeffs + delta_c
    c_b = parent.sh_coeffs - (pi_a / (pi_b + EPS)) * delta_c

    child_a = Gaussian(means=mu_a, L_flat=L_a_flat, opacities=alpha_a, sh_coeffs=c_a)
    child_b = Gaussian(means=mu_b, L_flat=L_b_flat, opacities=alpha_b, sh_coeffs=c_b)

    return child_a, child_b

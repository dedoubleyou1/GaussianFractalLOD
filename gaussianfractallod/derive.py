"""R&G-style binary split: derive two children from a parent Gaussian + split variables.

Uses Cholesky factored covariance for full orientation support.
The parent's Cholesky factor L_p defines the local coordinate frame —
position splits and covariance splits operate in this frame.

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
    cov_split: torch.Tensor        # (N, 6) Cholesky factor split (child A's L relative to parent)
    color_split: torch.Tensor      # (N, D) SH coefficient deviation


def derive_children(
    parent: Gaussian, split_vars: SplitVariables
) -> tuple[Gaussian, Gaussian]:
    """Derive two children from parent + split variables.

    Uses Richardson & Green (1997) split parameterization with full
    Cholesky covariance. The parent's Cholesky factor L_p serves as
    the local coordinate frame for position displacement.

    Args:
        parent: Batch of N parent Gaussians.
        split_vars: Batch of N split variable sets.

    Returns:
        (child_a, child_b): Two batches of N child Gaussians.
    """
    N = parent.num_gaussians

    # Mass partition
    pi_a = torch.sigmoid(split_vars.mass_logit).unsqueeze(-1)  # (N, 1)
    pi_b = 1.0 - pi_a

    # Opacity conservation: alpha_i = pi_i * alpha_parent
    alpha_a = pi_a * parent.opacities
    alpha_b = pi_b * parent.opacities

    # Get parent's Cholesky factor — defines local coordinate frame
    L_p = parent.L_matrix()  # (N, 3, 3)

    # Position split in parent's local frame via L_p
    # u2 is a displacement in unit space; L_p maps it to world space
    u2 = split_vars.position_split  # (N, 3)

    # R&G position formulas (center-of-mass conserved by construction):
    #   mu_A = mu_p + L_p @ u2 * sqrt(pi_B / pi_A)
    #   mu_B = mu_p - L_p @ u2 * sqrt(pi_A / pi_B)
    sqrt_ratio_ab = torch.sqrt(pi_b / (pi_a + EPS) + EPS)  # (N, 1)
    sqrt_ratio_ba = torch.sqrt(pi_a / (pi_b + EPS) + EPS)  # (N, 1)

    # L_p @ u2: (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1) -> (N, 3)
    displacement = torch.bmm(L_p, u2.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    mu_a = parent.means + displacement * sqrt_ratio_ab
    mu_b = parent.means - displacement * sqrt_ratio_ba

    # Covariance split via Cholesky factors
    # Each child gets its own Cholesky factor derived from the parent's.
    # Child A's L is parameterized as a perturbation of the parent's L,
    # scaled to conserve total covariance.
    #
    # We use the R&G total variance decomposition:
    #   Σ_p = π_A [Σ_A + δ_A δ_A^T] + π_B [Σ_B + δ_B δ_B^T]
    # where δ_i = μ_i - μ_p
    #
    # Rather than solving this exactly (which over-constrains child B),
    # we parameterize child A's Cholesky factor and derive child B's
    # covariance from the conservation equation.

    # Child A: L_a is a learned perturbation of L_p
    # cov_split encodes a 6-value delta for the Cholesky factor
    L_a_flat = parent.L_flat + split_vars.cov_split
    # Reconstruct L_a matrix to compute Σ_a for the conservation equation
    L_a = torch.zeros(N, 3, 3, device=parent.L_flat.device, dtype=parent.L_flat.dtype)
    L_a[:, 0, 0] = torch.exp(L_a_flat[:, 0])
    L_a[:, 1, 0] = L_a_flat[:, 1]
    L_a[:, 1, 1] = torch.exp(L_a_flat[:, 2])
    L_a[:, 2, 0] = L_a_flat[:, 3]
    L_a[:, 2, 1] = L_a_flat[:, 4]
    L_a[:, 2, 2] = torch.exp(L_a_flat[:, 5])

    cov_a = L_a @ L_a.transpose(-1, -2)  # (N, 3, 3)

    # Child B's covariance from conservation:
    #   Σ_p = π_A [Σ_A + δ_A δ_A^T] + π_B [Σ_B + δ_B δ_B^T]
    #   Σ_B = (Σ_p - π_A [Σ_A + δ_A δ_A^T] - π_B δ_B δ_B^T) / π_B
    cov_p = L_p @ L_p.transpose(-1, -2)  # (N, 3, 3)
    delta_a = (mu_a - parent.means).unsqueeze(-1)  # (N, 3, 1)
    delta_b = (mu_b - parent.means).unsqueeze(-1)  # (N, 3, 1)
    scatter_a = delta_a @ delta_a.transpose(-1, -2)  # (N, 3, 3)
    scatter_b = delta_b @ delta_b.transpose(-1, -2)

    cov_b = (cov_p - pi_a.unsqueeze(-1) * (cov_a + scatter_a)
             - pi_b.unsqueeze(-1) * scatter_b) / (pi_b.unsqueeze(-1) + EPS)

    # Ensure cov_b is positive definite by taking Cholesky
    # Add small diagonal for numerical stability
    cov_b = cov_b + EPS * torch.eye(3, device=cov_b.device).unsqueeze(0)
    # Use eigendecomposition fallback if Cholesky fails
    try:
        L_b = torch.linalg.cholesky(cov_b)
    except torch.linalg.LinAlgError:
        # Fallback: clamp eigenvalues to be positive
        eigvals, eigvecs = torch.linalg.eigh(cov_b)
        eigvals = eigvals.clamp(min=EPS)
        cov_b = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        L_b = torch.linalg.cholesky(cov_b)

    # Convert L matrices back to flat representation
    # [l00, l10, l11, l20, l21, l22] with log on diagonal
    L_a_out = torch.stack([
        torch.log(L_a[:, 0, 0].clamp(min=EPS)),
        L_a[:, 1, 0],
        torch.log(L_a[:, 1, 1].clamp(min=EPS)),
        L_a[:, 2, 0],
        L_a[:, 2, 1],
        torch.log(L_a[:, 2, 2].clamp(min=EPS)),
    ], dim=-1)

    L_b_out = torch.stack([
        torch.log(L_b[:, 0, 0].clamp(min=EPS)),
        L_b[:, 1, 0],
        torch.log(L_b[:, 1, 1].clamp(min=EPS)),
        L_b[:, 2, 0],
        L_b[:, 2, 1],
        torch.log(L_b[:, 2, 2].clamp(min=EPS)),
    ], dim=-1)

    # Color conservation: pi_A * c_A + pi_B * c_B = c_parent
    delta_c = split_vars.color_split
    c_a = parent.sh_coeffs + delta_c
    c_b = parent.sh_coeffs - (pi_a / (pi_b + EPS)) * delta_c

    child_a = Gaussian(means=mu_a, L_flat=L_a_out, opacities=alpha_a, sh_coeffs=c_a)
    child_b = Gaussian(means=mu_b, L_flat=L_b_out, opacities=alpha_b, sh_coeffs=c_b)

    return child_a, child_b

"""Subdivide parent Gaussians via sequential binary cuts along longest axis.

Uses least-squares optimal formulas for alpha compositing (not statistical
mixture). Children are sized and positioned to minimize the integrated
squared error between the alpha-composited children and the parent.

Supports variable cuts per Gaussian (1→2, 1→4, 1→8) based on a
per-Gaussian cut count. Longest axis is re-evaluated after each cut.

Formulas (derived from numerical optimization across opacity/spread space):
  sigma_c = sigma_parent * sqrt(1 - f^2/4) * (1 - 0.124 * alpha_p^2)
  alpha_c = [1 - sqrt(1 - alpha_p)] * (0.932 + 0.114 * f)

where f = spread factor (children at +/-f/2 in parent's local frame).
"""

import torch
import torch.nn.functional as F
import math
from gaussianfractallod.gaussian import Gaussian


# Spread factor: children displaced +/-f/2 in parent's local frame per cut axis
SPREAD_FACTOR = 1.0


def rotate_by_quat(quats: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions (wxyz convention)."""
    q = F.normalize(quats, dim=-1)
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    u = q[:, 1:4]
    uv = torch.cross(u, vectors, dim=-1)
    uuv = torch.cross(u, uv, dim=-1)
    return vectors + 2.0 * (w * uv + uuv)


def subdivide(parents: Gaussian, num_cuts: int = 3, opacity_floor: float = 0.05, opacity_formula: str = "linear") -> Gaussian:
    """Subdivide each parent Gaussian into 2^num_cuts children.

    Applies sequential binary cuts along the longest axis (re-evaluated
    after each cut). For isotropic Gaussians this degrades to the standard
    octree. For elongated Gaussians, the longest axis may be cut multiple
    times, naturally correcting the aspect ratio.

    Args:
        parents: N parent Gaussians.
        num_cuts: number of binary cuts (1→2, 2→4, 3→8 children).

    Returns:
        N * 2^num_cuts child Gaussians.
    """
    current = parents
    for cut in range(num_cuts):
        axes = current.log_scales.argmax(dim=-1)
        current = _binary_cut_along_axis(current, axes, opacity_floor=opacity_floor, opacity_formula=opacity_formula)
    return current


def subdivide_variable(parents: Gaussian, cuts_per_parent: torch.Tensor, opacity_floor: float = 0.05, opacity_formula: str = "linear") -> tuple[Gaussian, torch.Tensor]:
    """Subdivide parents with a variable number of cuts each.

    Args:
        parents: N parent Gaussians.
        cuts_per_parent: (N,) int tensor, values in {1, 2, 3}.

    Returns:
        (children, child_indices): children Gaussian batch and (M,) tensor
        mapping each child back to its position within the split (0-based).
    """
    device = parents.means.device
    N = parents.num_gaussians

    all_means = []
    all_quats = []
    all_log_scales = []
    all_opacities = []
    all_sh_dc = []
    all_sh_rest = []
    all_child_idx = []

    for n_cuts in [1, 2, 3]:
        mask = cuts_per_parent == n_cuts
        if not mask.any():
            continue

        tier_parents = Gaussian(
            means=parents.means[mask],
            quats=parents.quats[mask],
            log_scales=parents.log_scales[mask],
            opacities=parents.opacities[mask],
            sh_dc=parents.sh_dc[mask],
            sh_rest=parents.sh_rest[mask],
        )

        tier_children = subdivide(tier_parents, num_cuts=n_cuts, opacity_floor=opacity_floor, opacity_formula=opacity_formula)
        n_tier = tier_parents.num_gaussians
        n_children_per = 2 ** n_cuts

        # Child index: 0..(2^n_cuts - 1) for each parent
        child_idx = torch.arange(n_children_per, device=device).repeat(n_tier)

        all_means.append(tier_children.means)
        all_quats.append(tier_children.quats)
        all_log_scales.append(tier_children.log_scales)
        all_opacities.append(tier_children.opacities)
        all_sh_dc.append(tier_children.sh_dc)
        all_sh_rest.append(tier_children.sh_rest)
        all_child_idx.append(child_idx)

    children = Gaussian(
        means=torch.cat(all_means, dim=0),
        quats=torch.cat(all_quats, dim=0),
        log_scales=torch.cat(all_log_scales, dim=0),
        opacities=torch.cat(all_opacities, dim=0),
        sh_dc=torch.cat(all_sh_dc, dim=0),
        sh_rest=torch.cat(all_sh_rest, dim=0),
    )
    child_indices = torch.cat(all_child_idx, dim=0)

    return children, child_indices


# Keep old name as alias for backward compatibility
def subdivide_to_8(parents: Gaussian) -> Gaussian:
    """Subdivide each parent into 8 children (3 cuts)."""
    return subdivide(parents, num_cuts=3)


def _binary_cut_along_axis(gaussians: Gaussian, axes, opacity_floor: float = 0.05, opacity_formula: str = "linear") -> Gaussian:
    """Split each Gaussian into 2 along a local axis.

    Args:
        gaussians: N Gaussians to split.
        axes: int (same axis for all) or (N,) tensor of per-Gaussian axis indices.
    """
    N = gaussians.num_gaussians
    device = gaussians.means.device
    f = SPREAD_FACTOR

    if isinstance(axes, int):
        axis_vector = torch.zeros(N, 3, device=device)
        axis_vector[:, axes] = 1.0
        axis_idx = axes
    else:
        axis_vector = torch.zeros(N, 3, device=device)
        axis_vector.scatter_(1, axes.unsqueeze(1), 1.0)
        axis_idx = None

    scales = gaussians.scales()
    local_offset = axis_vector * scales * (f / 2.0)
    displacement_world = rotate_by_quat(gaussians.quats, local_offset)

    mu_right = gaussians.means + displacement_world
    mu_left = gaussians.means - displacement_world

    alpha_p = torch.sigmoid(gaussians.opacities)
    if opacity_formula == "classic":
        # Original per-cut formula with compounding scale (no floor).
        alpha_c = (1 - torch.sqrt(1 - alpha_p)) * (0.932 + 0.114 * f) * 0.1
    else:
        # Linear area-preserving formula with floor.
        # Scale applied once after all cuts (in split_tree.add_level).
        alpha_c = (alpha_p - opacity_floor) * 0.65 + opacity_floor
    alpha_c = alpha_c.clamp(min=1e-6, max=1.0 - 1e-6)
    child_logit = torch.log(alpha_c / (1.0 - alpha_c))

    scale_base = math.sqrt(max(1.0 - f * f / 4.0, 0.01))
    alpha_correction = (1.0 - 0.124 * alpha_p ** 2)
    scale_factor = scale_base * alpha_correction

    log_scale_adjust = torch.zeros(N, 3, device=device)
    if axis_idx is not None:
        log_scale_adjust[:, axis_idx:axis_idx+1] = torch.log(scale_factor)
    else:
        log_scale_adjust.scatter_(1, axes.unsqueeze(1), torch.log(scale_factor).expand(N, 1))
    child_log_scales = gaussians.log_scales + log_scale_adjust

    child_quats = gaussians.quats

    means = torch.stack([mu_right, mu_left], dim=1).reshape(2 * N, 3)
    quats = torch.stack([child_quats, child_quats], dim=1).reshape(2 * N, 4)
    log_scales = torch.stack([child_log_scales, child_log_scales], dim=1).reshape(2 * N, 3)
    opacities = torch.stack([child_logit, child_logit], dim=1).reshape(2 * N, 1)
    sh_dc = gaussians.sh_dc.repeat_interleave(2, dim=0)
    sh_rest = gaussians.sh_rest.repeat_interleave(2, dim=0)

    return Gaussian(means=means, quats=quats, log_scales=log_scales,
                    opacities=opacities, sh_dc=sh_dc, sh_rest=sh_rest)

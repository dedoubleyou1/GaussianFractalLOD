"""Pruning: remove splits that don't contribute meaningful detail.

Three pruning signals:
  1. Mass fade-out: one child's mass share (π) drops below threshold
  2. Split convergence: split variables are near-zero, meaning children
     are essentially identical to the parent (the split adds nothing)
  3. Low opacity: child's effective opacity is too low to be visible

When a split is pruned, both children are removed and the parent
renders in their place — which is exactly the LoD behavior we want.
"""

import torch
from gaussianfractallod.split_tree import SplitTree


def prune_level(
    tree: SplitTree,
    level_idx: int,
    mass_threshold: float = 0.05,
    split_magnitude_threshold: float = 0.01,
    opacity_threshold: float = 0.005,
    parent_opacities: torch.Tensor | None = None,
) -> dict:
    """Prune splits that don't contribute meaningful detail.

    Args:
        tree: The split tree.
        level_idx: Which level to prune.
        mass_threshold: Prune child if its mass share π < threshold.
        split_magnitude_threshold: Prune entire split if the total
            magnitude of split variables is below this. Means children
            are essentially the same as the parent.
        opacity_threshold: Prune child if its effective opacity
            (parent_opacity * π) is below this.
        parent_opacities: (N,) effective opacities of parent Gaussians
            at this level. Needed for opacity-based pruning.

    Returns:
        Dict with counts: {'mass': n, 'convergence': n, 'opacity': n, 'total': n}
    """
    level = tree.levels[level_idx]
    mass_logit = level.mass_logit.detach()
    pi_a = torch.sigmoid(mass_logit)  # (N,)
    pi_b = 1.0 - pi_a

    counts = {"mass": 0, "convergence": 0, "opacity": 0, "total": 0}

    with torch.no_grad():
        # --- 1. Split convergence: entire split is useless ---
        # If all split variables are near zero, children ≈ parent
        pos_mag = level.position_split.detach().norm(dim=-1)      # (N,)
        var_mag = level.variance_split.detach().abs().sum(dim=-1)  # (N,) deviation from 0 (uniform)
        color_mag = level.color_split.detach().norm(dim=-1)    # (N,)
        total_mag = pos_mag + var_mag + color_mag

        converged = total_mag < split_magnitude_threshold
        if converged.any():
            level.occupancy[converged, 0] = False
            level.occupancy[converged, 1] = False
            counts["convergence"] = int(converged.sum().item())

        # --- 2. Mass fade-out: one child wants to vanish ---
        # Only check nodes not already pruned by convergence
        active = level.occupancy[:, 0] | level.occupancy[:, 1]

        prune_a = active & (pi_a < mass_threshold)
        if prune_a.any():
            level.occupancy[prune_a, 0] = False
            counts["mass"] += int(prune_a.sum().item())

        prune_b = active & (pi_b < mass_threshold)
        if prune_b.any():
            level.occupancy[prune_b, 1] = False
            counts["mass"] += int(prune_b.sum().item())

        # --- 3. Low opacity: child is invisible ---
        if parent_opacities is not None:
            eff_opacity_a = torch.sigmoid(parent_opacities) * pi_a
            eff_opacity_b = torch.sigmoid(parent_opacities) * pi_b

            low_opacity_a = level.occupancy[:, 0] & (eff_opacity_a < opacity_threshold)
            if low_opacity_a.any():
                level.occupancy[low_opacity_a, 0] = False
                counts["opacity"] += int(low_opacity_a.sum().item())

            low_opacity_b = level.occupancy[:, 1] & (eff_opacity_b < opacity_threshold)
            if low_opacity_b.any():
                level.occupancy[low_opacity_b, 1] = False
                counts["opacity"] += int(low_opacity_b.sum().item())

        # --- Clean up: zero dead split vars, renormalize mass ---
        only_a = level.occupancy[:, 0] & ~level.occupancy[:, 1]
        only_b = ~level.occupancy[:, 0] & level.occupancy[:, 1]
        neither = ~level.occupancy[:, 0] & ~level.occupancy[:, 1]
        dead = only_a | only_b | neither

        if dead.any():
            level.position_split[dead] = 0.0
            level.variance_split[dead] = 0.0
            level.color_split[dead] = 0.0

        if only_a.any():
            level.mass_logit[only_a] = 10.0
        if only_b.any():
            level.mass_logit[only_b] = -10.0
        if neither.any():
            level.mass_logit[neither] = 0.0

    counts["total"] = counts["mass"] + counts["convergence"] * 2 + counts["opacity"]
    return counts

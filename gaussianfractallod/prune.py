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
    counts = {"mass": 0, "convergence": 0, "opacity": 0, "total": 0}

    with torch.no_grad():
        # --- 1. Split convergence: entire split is useless ---
        # If all split variables are near zero, children ≈ parent
        # Split magnitude: how much does this split deviate from "do nothing"?
        # cut_offset ≈ 0 means symmetric split (minimal change)
        # color_split ≈ 0 means children share parent's color
        offset_mag = level.cut_offset.detach().abs()           # (N,)
        color_mag = level.color_split.detach().norm(dim=-1)    # (N,)
        total_mag = offset_mag + color_mag

        converged = total_mag < split_magnitude_threshold
        if converged.any():
            level.occupancy[converged, 0] = False
            level.occupancy[converged, 1] = False
            counts["convergence"] = int(converged.sum().item())

        # --- 2. Mass fade-out: one child wants to vanish ---
        # Mass partition is derived from cut_offset via CDF
        from gaussianfractallod.derive import _Phi
        pi_left = _Phi(level.cut_offset.detach())
        pi_right = 1.0 - pi_left

        active = level.occupancy[:, 0] | level.occupancy[:, 1]

        prune_right = active & (pi_right < mass_threshold)
        if prune_right.any():
            level.occupancy[prune_right, 0] = False
            counts["mass"] += int(prune_right.sum().item())

        prune_left = active & (pi_left < mass_threshold)
        if prune_left.any():
            level.occupancy[prune_left, 1] = False
            counts["mass"] += int(prune_left.sum().item())

        # --- 3. Low opacity: child is invisible ---
        if parent_opacities is not None:
            eff_opacity_right = torch.sigmoid(parent_opacities) * pi_right
            eff_opacity_left = torch.sigmoid(parent_opacities) * pi_left

            low_opacity_right = level.occupancy[:, 0] & (eff_opacity_right < opacity_threshold)
            if low_opacity_right.any():
                level.occupancy[low_opacity_right, 0] = False
                counts["opacity"] += int(low_opacity_right.sum().item())

            low_opacity_left = level.occupancy[:, 1] & (eff_opacity_left < opacity_threshold)
            if low_opacity_left.any():
                level.occupancy[low_opacity_left, 1] = False
                counts["opacity"] += int(low_opacity_left.sum().item())

        # --- Clean up: zero dead split vars, renormalize mass ---
        only_a = level.occupancy[:, 0] & ~level.occupancy[:, 1]
        only_b = ~level.occupancy[:, 0] & level.occupancy[:, 1]
        neither = ~level.occupancy[:, 0] & ~level.occupancy[:, 1]
        dead = only_a | only_b | neither

        if dead.any():
            level.cut_offset[dead] = 0.0
            level.color_split[dead] = 0.0

        # Renormalize: push all mass to the surviving child via cut_offset
        # Large positive offset → π_left ≈ 1 (left child gets all mass)
        # Large negative offset → π_right ≈ 1 (right child gets all mass)
        if only_a.any():  # only right child survives
            level.cut_offset[only_a] = -10.0
        if only_b.any():  # only left child survives
            level.cut_offset[only_b] = 10.0
        if neither.any():
            level.cut_offset[neither] = 0.0

    counts["total"] = counts["mass"] + counts["convergence"] * 2 + counts["opacity"]
    return counts

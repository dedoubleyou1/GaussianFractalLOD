"""Reconstruct flat Gaussian list from roots + split tree at a target depth.

Iteratively applies split derivation level-by-level, respecting occupancy masks.
Uses iterative (not recursive) approach for GPU-friendly batched operations.

Supports caching: when training level L, levels 0..L-1 are frozen and their
output is constant. Pass cached_parents to skip recomputing frozen levels.
"""

import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.derive import derive_children


def _apply_one_level(current: Gaussian, tree: SplitTree, level_idx: int) -> Gaussian:
    """Apply one level of split derivation with occupancy masking."""
    split_vars = tree.get_level_split_vars(level_idx)
    occupancy = tree.get_occupancy(level_idx)

    child_a, child_b = derive_children(current, split_vars)

    parts_means = []
    parts_L_flat = []
    parts_opacities = []
    parts_sh = []

    mask_a = occupancy[:, 0]
    mask_b = occupancy[:, 1]

    if mask_a.any():
        parts_means.append(child_a.means[mask_a])
        parts_L_flat.append(child_a.L_flat[mask_a])
        parts_opacities.append(child_a.opacities[mask_a])
        parts_sh.append(child_a.sh_coeffs[mask_a])

    if mask_b.any():
        parts_means.append(child_b.means[mask_b])
        parts_L_flat.append(child_b.L_flat[mask_b])
        parts_opacities.append(child_b.opacities[mask_b])
        parts_sh.append(child_b.sh_coeffs[mask_b])

    if not parts_means:
        return current

    return Gaussian(
        means=torch.cat(parts_means, dim=0),
        L_flat=torch.cat(parts_L_flat, dim=0),
        opacities=torch.cat(parts_opacities, dim=0),
        sh_coeffs=torch.cat(parts_sh, dim=0),
    )


def reconstruct(
    roots: Gaussian,
    tree: SplitTree,
    target_depth: int,
    cached_parents: Gaussian | None = None,
    cache_depth: int = 0,
) -> Gaussian:
    """Reconstruct Gaussians at a target binary depth.

    Args:
        roots: The root-level Gaussians (N_roots,).
        tree: The split tree containing split variables.
        target_depth: How many binary split levels to apply.
            0 = return roots, 1 = first split, etc.
        cached_parents: Pre-computed Gaussians at cache_depth. If provided,
            skips levels 0..cache_depth-1 (the frozen levels).
        cache_depth: The depth that cached_parents represents.

    Returns:
        Gaussian batch containing all leaf Gaussians at the target depth.
    """
    if target_depth == 0 or tree.depth == 0:
        return roots

    actual_depth = min(target_depth, tree.depth)

    # Start from cache if available
    if cached_parents is not None and cache_depth > 0:
        current = cached_parents
        start_level = cache_depth
    else:
        current = roots
        start_level = 0

    for level_idx in range(start_level, actual_depth):
        current = _apply_one_level(current, tree, level_idx)

    return current


def build_cache(roots: Gaussian, tree: SplitTree, depth: int) -> Gaussian:
    """Build a detached cache of Gaussians at a given depth.

    Used during training: compute the frozen levels once, cache the result,
    and reuse it for every training step at the current level.
    """
    with torch.no_grad():
        cached = reconstruct(roots, tree, depth)
    # Detach to ensure no gradient graph is retained
    return Gaussian(
        means=cached.means.detach(),
        scales=cached.scales.detach(),
        opacities=cached.opacities.detach(),
        sh_coeffs=cached.sh_coeffs.detach(),
    )

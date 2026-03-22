"""Reconstruct flat Gaussian list from roots + split tree at a target depth.

Iteratively applies split derivation level-by-level, respecting occupancy masks.
Uses iterative (not recursive) approach for GPU-friendly batched operations.
"""

import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.derive import derive_children


def reconstruct(
    roots: Gaussian, tree: SplitTree, target_depth: int
) -> Gaussian:
    """Reconstruct Gaussians at a target binary depth.

    Args:
        roots: The root-level Gaussians (N_roots,).
        tree: The split tree containing split variables.
        target_depth: How many binary split levels to apply.
            0 = return roots, 1 = first split, etc.

    Returns:
        Gaussian batch containing all leaf Gaussians at the target depth.
    """
    if target_depth == 0 or tree.depth == 0:
        return roots

    current = roots
    actual_depth = min(target_depth, tree.depth)

    for level_idx in range(actual_depth):
        split_vars = tree.get_level_split_vars(level_idx)
        occupancy = tree.get_occupancy(level_idx)  # (num_nodes, 2)

        child_a, child_b = derive_children(current, split_vars)

        # Collect active children
        parts_means = []
        parts_scales = []
        parts_opacities = []
        parts_sh = []

        mask_a = occupancy[:, 0]
        mask_b = occupancy[:, 1]

        if mask_a.any():
            parts_means.append(child_a.means[mask_a])
            parts_scales.append(child_a.scales[mask_a])
            parts_opacities.append(child_a.opacities[mask_a])
            parts_sh.append(child_a.sh_coeffs[mask_a])

        if mask_b.any():
            parts_means.append(child_b.means[mask_b])
            parts_scales.append(child_b.scales[mask_b])
            parts_opacities.append(child_b.opacities[mask_b])
            parts_sh.append(child_b.sh_coeffs[mask_b])

        if not parts_means:
            return current

        current = Gaussian(
            means=torch.cat(parts_means, dim=0),
            scales=torch.cat(parts_scales, dim=0),
            opacities=torch.cat(parts_opacities, dim=0),
            sh_coeffs=torch.cat(parts_sh, dim=0),
        )

    return current

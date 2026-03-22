"""Mass-based child pruning: remove children with negligible opacity share.

When one child is pruned, the surviving child effectively becomes the
parent — split variables are zeroed so the derivation produces a child
identical to the parent (mass_logit is set to a large positive/negative
value so sigmoid gives ~1.0 or ~0.0, assigning all mass to the survivor).
"""

import torch
from gaussianfractallod.split_tree import SplitTree


def prune_level(
    tree: SplitTree, level_idx: int, threshold: float = 0.01
) -> int:
    """Prune children whose mass partition falls below threshold."""
    level = tree.levels[level_idx]
    mass_logit = level.mass_logit.detach()
    pi_a = torch.sigmoid(mass_logit)
    pi_b = 1.0 - pi_a

    pruned_count = 0

    with torch.no_grad():
        prune_a = pi_a < threshold
        if prune_a.any():
            level.occupancy[prune_a, 0] = False
            pruned_count += prune_a.sum().item()

        prune_b = pi_b < threshold
        if prune_b.any():
            level.occupancy[prune_b, 1] = False
            pruned_count += prune_b.sum().item()

        only_a = level.occupancy[:, 0] & ~level.occupancy[:, 1]
        only_b = ~level.occupancy[:, 0] & level.occupancy[:, 1]
        dead = only_a | only_b | (~level.occupancy[:, 0] & ~level.occupancy[:, 1])

        if dead.any():
            level.position_split[dead] = 0.0
            level.variance_split[dead] = 0.0
            level.color_split[dead] = 0.0

        if only_a.any():
            level.mass_logit[only_a] = 10.0
        if only_b.any():
            level.mass_logit[only_b] = -10.0

    return pruned_count

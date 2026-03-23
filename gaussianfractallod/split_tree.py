"""Binary split tree: stores split variables for each level of the hierarchy.

The tree is organized by levels. Level 0 splits the root Gaussians.
Level 1 splits the children from level 0. Each level stores split
variables for all nodes at that depth.

Node count at level L: num_roots * 2^L (before pruning).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from gaussianfractallod.derive import SplitVariables


class SplitLevel(nn.Module):
    """Split variables for all nodes at one level of the tree."""

    def __init__(self, num_nodes: int, sh_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.sh_dim = sh_dim

        # Learnable split variables
        self.mass_logit = nn.Parameter(torch.zeros(num_nodes))
        # Random direction initialization — no cardinal axis bias
        self.position_split = nn.Parameter(torch.randn(num_nodes, 3) * 0.1)
        # Variance partition ratio (pre-sigmoid): 3 values per axis
        # sigmoid(0) = 0.5 → uniform split (each child gets half the budget)
        self.variance_split = nn.Parameter(torch.zeros(num_nodes, 3))
        self.color_split = nn.Parameter(torch.zeros(num_nodes, sh_dim))

        # Occupancy: (num_nodes, 2) — [child_a_active, child_b_active]
        self.register_buffer(
            "occupancy", torch.ones(num_nodes, 2, dtype=torch.bool)
        )

    def get_split_vars(self) -> SplitVariables:
        return SplitVariables(
            mass_logit=self.mass_logit,
            position_split=self.position_split,
            variance_split=self.variance_split,
            color_split=self.color_split,
        )


class SplitTree(nn.Module):
    """Binary split tree organized by levels."""

    def __init__(self, num_roots: int, sh_dim: int):
        super().__init__()
        self.num_roots = num_roots
        self.sh_dim = sh_dim
        self.levels = nn.ModuleList()

    @property
    def depth(self) -> int:
        return len(self.levels)

    @property
    def num_splits(self) -> int:
        return sum(level.num_nodes for level in self.levels)

    def add_level(self) -> None:
        """Add a new split level at the bottom of the tree.

        Node count: each active child at the previous level becomes a
        node to split at this level. occupancy is (num_nodes, 2) bools,
        so sum() counts total active children across both slots.
        """
        if self.depth == 0:
            num_nodes = self.num_roots
        else:
            prev_level = self.levels[-1]
            # Count active children: each True in occupancy (num_nodes, 2)
            # is one child that becomes a node at the next level
            num_nodes = int(prev_level.occupancy.sum().item())

        level = SplitLevel(num_nodes, self.sh_dim)
        self.levels.append(level)

    def get_level_split_vars(self, level_idx: int) -> SplitVariables:
        return self.levels[level_idx].get_split_vars()

    def level_parameters(self, level_idx: int):
        """Yield learnable parameters for a specific level."""
        level = self.levels[level_idx]
        yield level.mass_logit
        yield level.position_split
        yield level.variance_split
        yield level.color_split

    def get_occupancy(self, level_idx: int) -> torch.Tensor:
        return self.levels[level_idx].occupancy

    def set_occupancy(
        self, level: int, node_idx: int, child_a: bool = True, child_b: bool = True
    ) -> None:
        self.levels[level].occupancy[node_idx, 0] = child_a
        self.levels[level].occupancy[node_idx, 1] = child_b

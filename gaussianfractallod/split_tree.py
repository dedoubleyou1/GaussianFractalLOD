"""Hierarchical Gaussian tree: each level stores independently trainable Gaussians.

Level 0: root Gaussians (trained in Phase 1)
Level L: 8× more Gaussians than level L-1 (initialized via subdivision)

Each level is independently renderable for LoD.
Children are initialized from parent subdivision but train freely.
"""

import torch
import torch.nn as nn
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.subdivide import subdivide_to_8


class GaussianLevel(nn.Module):
    """Trainable Gaussians at one level of the hierarchy."""

    def __init__(self, means: torch.Tensor, L_flat: torch.Tensor,
                 opacities: torch.Tensor, sh_coeffs: torch.Tensor):
        super().__init__()
        self.means = nn.Parameter(means)
        self.L_flat = nn.Parameter(L_flat)
        self.opacities = nn.Parameter(opacities)
        self.sh_coeffs = nn.Parameter(sh_coeffs)

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def get_gaussians(self) -> Gaussian:
        return Gaussian(
            means=self.means,
            L_flat=self.L_flat,
            opacities=self.opacities,
            sh_coeffs=self.sh_coeffs,
        )

    def parameters_list(self):
        """Return list of parameters for optimizer."""
        return [self.means, self.L_flat, self.opacities, self.sh_coeffs]


class GaussianTree(nn.Module):
    """Hierarchical tree of Gaussian levels.

    Each level is independently renderable. Children are initialized
    from parent subdivision (8× per parent) but train freely.
    """

    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleList()

    @property
    def depth(self) -> int:
        return len(self.levels)

    def set_root_level(self, roots: Gaussian) -> None:
        """Set level 0 from trained root Gaussians (frozen)."""
        level = GaussianLevel(
            means=roots.means.detach().clone(),
            L_flat=roots.L_flat.detach().clone(),
            opacities=roots.opacities.detach().clone(),
            sh_coeffs=roots.sh_coeffs.detach().clone(),
        )
        # Freeze root level
        for p in level.parameters():
            p.requires_grad_(False)
        self.levels.append(level)

    def add_level(self) -> None:
        """Add a new level by subdividing the current finest level.

        Each parent Gaussian produces 8 children via 3 sequential
        binary cuts. Children are detached and set as trainable parameters.
        """
        assert self.depth > 0, "Must set root level first"

        parent_level = self.levels[-1]
        parents = parent_level.get_gaussians()

        # Subdivide: N parents → 8N children
        with torch.no_grad():
            children = subdivide_to_8(parents)

        # Create trainable level from subdivision
        new_level = GaussianLevel(
            means=children.means.detach().clone(),
            L_flat=children.L_flat.detach().clone(),
            opacities=children.opacities.detach().clone(),
            sh_coeffs=children.sh_coeffs.detach().clone(),
        )
        self.levels.append(new_level)

    def get_level_gaussians(self, level: int) -> Gaussian:
        """Get Gaussians at a specific level."""
        return self.levels[level].get_gaussians()

    def level_parameters(self, level: int):
        """Yield trainable parameters for a specific level."""
        return self.levels[level].parameters_list()

    def get_gaussians_at_depth(self, target_depth: int) -> Gaussian:
        """Get Gaussians for rendering at a target depth.

        Returns the Gaussians at the specified level (0-indexed).
        Level 0 = roots, level 1 = first subdivision, etc.
        """
        actual_depth = min(target_depth, self.depth - 1)
        return self.levels[actual_depth].get_gaussians()

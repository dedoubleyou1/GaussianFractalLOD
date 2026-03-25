"""Hierarchical Gaussian tree: each level stores independently trainable Gaussians.

Level 0: root Gaussians (trained in Phase 1)
Level L: children of selected parents from level L-1 (adaptive splitting)

Each level is independently renderable for LoD.
Children are initialized from parent subdivision but train freely.
"""

import torch
import torch.nn as nn
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.subdivide import subdivide_to_8


class GaussianLevel(nn.Module):
    """Trainable Gaussians at one level of the hierarchy."""

    def __init__(self, means: torch.Tensor, quats: torch.Tensor,
                 log_scales: torch.Tensor, opacities: torch.Tensor,
                 sh_coeffs: torch.Tensor):
        super().__init__()
        self.means = nn.Parameter(means)
        self.quats = nn.Parameter(quats)
        self.log_scales = nn.Parameter(log_scales)
        self.opacities = nn.Parameter(opacities)
        self.sh_coeffs = nn.Parameter(sh_coeffs)

        # Store initial values for regularization
        self.register_buffer("init_means", means.detach().clone())
        self.register_buffer("init_log_scales", log_scales.detach().clone())

        # Gradient accumulator for adaptive splitting (not a parameter)
        self.register_buffer("grad_accum", torch.zeros(means.shape[0]))
        self.register_buffer("grad_count", torch.zeros(means.shape[0]))

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def get_gaussians(self) -> Gaussian:
        return Gaussian(
            means=self.means,
            quats=self.quats,
            log_scales=self.log_scales,
            opacities=self.opacities,
            sh_coeffs=self.sh_coeffs,
        )

    def accumulate_grad(self) -> None:
        """Accumulate position gradient magnitude for split decisions."""
        if self.means.grad is not None:
            self.grad_accum += self.means.grad.detach().norm(dim=-1)
            self.grad_count += 1

    def avg_grad(self) -> torch.Tensor:
        """Average gradient magnitude per Gaussian."""
        return self.grad_accum / (self.grad_count + 1e-8)

    def reset_opacity(self, value: float = -2.2) -> None:
        """Reset all opacities to a low value (inverse_sigmoid(0.1) ~ -2.2)."""
        with torch.no_grad():
            self.opacities.fill_(value)


class GaussianTree(nn.Module):
    """Hierarchical tree of Gaussian levels."""

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
            quats=roots.quats.detach().clone(),
            log_scales=roots.log_scales.detach().clone(),
            opacities=roots.opacities.detach().clone(),
            sh_coeffs=roots.sh_coeffs.detach().clone(),
        )
        for p in level.parameters():
            p.requires_grad_(False)
        self.levels.append(level)

    def add_level(self, split_mask: torch.Tensor | None = None) -> None:
        """Add a new level by subdividing selected parents.

        Args:
            split_mask: (N,) bool tensor. If provided, only parents where
                mask is True are subdivided into 8 children. Parents where
                mask is False are carried forward as-is (kept at this level).
                If None, all parents are subdivided.
        """
        assert self.depth > 0, "Must set root level first"

        parent_level = self.levels[-1]
        parents = parent_level.get_gaussians()

        N_parents = parents.num_gaussians

        with torch.no_grad():
            if split_mask is None or split_mask.all():
                # Subdivide all parents
                children = subdivide_to_8(parents)
                parent_means_repeated = parents.means.repeat_interleave(8, dim=0)
                # Parent index: [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, ...]
                parent_indices = torch.arange(N_parents, device=parents.means.device).repeat_interleave(8)
            else:
                # Split only selected parents, keep the rest as-is
                split_indices = torch.where(split_mask)[0]
                keep_indices = torch.where(~split_mask)[0]

                split_parents = Gaussian(
                    means=parents.means[split_mask],
                    quats=parents.quats[split_mask],
                    log_scales=parents.log_scales[split_mask],
                    opacities=parents.opacities[split_mask],
                    sh_coeffs=parents.sh_coeffs[split_mask],
                )
                keep_parents = Gaussian(
                    means=parents.means[~split_mask],
                    quats=parents.quats[~split_mask],
                    log_scales=parents.log_scales[~split_mask],
                    opacities=parents.opacities[~split_mask],
                    sh_coeffs=parents.sh_coeffs[~split_mask],
                )

                if split_parents.num_gaussians > 0:
                    split_children = subdivide_to_8(split_parents)
                    split_parent_means = split_parents.means.repeat_interleave(8, dim=0)
                else:
                    split_children = None

                # Combine: kept parents + new children
                parts_means = []
                parts_quats = []
                parts_log_scales = []
                parts_op = []
                parts_sh = []
                parts_parent_means = []
                parts_parent_idx = []

                if keep_parents.num_gaussians > 0:
                    parts_means.append(keep_parents.means)
                    parts_quats.append(keep_parents.quats)
                    parts_log_scales.append(keep_parents.log_scales)
                    parts_op.append(keep_parents.opacities)
                    parts_sh.append(keep_parents.sh_coeffs)
                    parts_parent_means.append(keep_parents.means)
                    parts_parent_idx.append(keep_indices)

                if split_children is not None:
                    parts_means.append(split_children.means)
                    parts_quats.append(split_children.quats)
                    parts_log_scales.append(split_children.log_scales)
                    parts_op.append(split_children.opacities)
                    parts_sh.append(split_children.sh_coeffs)
                    parts_parent_means.append(split_parent_means)
                    parts_parent_idx.append(split_indices.repeat_interleave(8))

                children = Gaussian(
                    means=torch.cat(parts_means, dim=0),
                    quats=torch.cat(parts_quats, dim=0),
                    log_scales=torch.cat(parts_log_scales, dim=0),
                    opacities=torch.cat(parts_op, dim=0),
                    sh_coeffs=torch.cat(parts_sh, dim=0),
                )
                parent_means_repeated = torch.cat(parts_parent_means, dim=0)
                parent_indices = torch.cat(parts_parent_idx, dim=0)

            expected_offset = (children.means - parent_means_repeated).norm(dim=-1)
            expected_offset = expected_offset.clamp(min=1e-4)

        new_level = GaussianLevel(
            means=children.means.detach().clone(),
            quats=children.quats.detach().clone(),
            log_scales=children.log_scales.detach().clone(),
            opacities=children.opacities.detach().clone(),
            sh_coeffs=children.sh_coeffs.detach().clone(),
        )
        new_level.register_buffer("expected_offset", expected_offset.detach().clone())
        new_level.register_buffer("parent_indices", parent_indices.detach().clone())
        self.levels.append(new_level)

    def get_level_gaussians(self, level: int) -> Gaussian:
        return self.levels[level].get_gaussians()

    def level_parameters(self, level: int):
        return [
            self.levels[level].means,
            self.levels[level].quats,
            self.levels[level].log_scales,
            self.levels[level].opacities,
            self.levels[level].sh_coeffs,
        ]

    def get_gaussians_at_depth(self, target_depth: int) -> Gaussian:
        actual_depth = min(target_depth, self.depth - 1)
        return self.levels[actual_depth].get_gaussians()

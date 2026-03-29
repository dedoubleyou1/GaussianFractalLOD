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
                 sh_dc: torch.Tensor, sh_rest: torch.Tensor):
        super().__init__()
        self.means = nn.Parameter(means)
        self.quats = nn.Parameter(quats)
        self.log_scales = nn.Parameter(log_scales)
        self.opacities = nn.Parameter(opacities)
        self.sh_dc = nn.Parameter(sh_dc)
        self.sh_rest = nn.Parameter(sh_rest)

        # Store initial values for regularization
        self.register_buffer("init_means", means.detach().clone())
        self.register_buffer("init_log_scales", log_scales.detach().clone())

        # Gradient tracking for adaptive splitting
        self.register_buffer("grad_accum", torch.zeros(means.shape[0]))
        self.register_buffer("grad_count", torch.zeros(means.shape[0]))
        self.register_buffer("grad_max", torch.zeros(means.shape[0]))

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def get_gaussians(self) -> Gaussian:
        return Gaussian(
            means=self.means,
            quats=self.quats,
            log_scales=self.log_scales,
            opacities=self.opacities,
            sh_dc=self.sh_dc,
            sh_rest=self.sh_rest,
        )

    def accumulate_grad(self) -> None:
        """Accumulate position gradient magnitude for split decisions."""
        if self.means.grad is not None:
            grad_norm = self.means.grad.detach().norm(dim=-1)
            self.grad_accum += grad_norm
            self.grad_count += 1
            self.grad_max = torch.max(self.grad_max, grad_norm)

    def split_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Scores for split decisions.

        Returns:
            (max_grad, mean_grad) — two independent signals:
            - max_grad: peak gradient from any view (catches under-observed regions)
            - mean_grad: average gradient across views (catches coverage gaps)
        """
        mean_grad = self.grad_accum / self.grad_count.clamp(min=1)
        return self.grad_max, mean_grad

    def reset_opacity(self, value: float = -2.2, keep_above: float = 0.5) -> None:
        """Reset low-opacity Gaussians. High-opacity ones keep their values."""
        with torch.no_grad():
            current_alpha = torch.sigmoid(self.opacities)
            reset_mask = current_alpha < keep_above
            self.opacities[reset_mask] = value

    def prune(self, keep_mask: torch.Tensor) -> None:
        """Remove Gaussians where keep_mask is False."""
        self.means = nn.Parameter(self.means.data[keep_mask])
        self.quats = nn.Parameter(self.quats.data[keep_mask])
        self.log_scales = nn.Parameter(self.log_scales.data[keep_mask])
        self.opacities = nn.Parameter(self.opacities.data[keep_mask])
        self.sh_dc = nn.Parameter(self.sh_dc.data[keep_mask])
        self.sh_rest = nn.Parameter(self.sh_rest.data[keep_mask])

        self.register_buffer("init_means", self.init_means[keep_mask])
        self.register_buffer("init_log_scales", self.init_log_scales[keep_mask])
        self.register_buffer("grad_accum", self.grad_accum[keep_mask])
        self.register_buffer("grad_count", self.grad_count[keep_mask])
        self.register_buffer("grad_max", self.grad_max[keep_mask])
        if hasattr(self, 'expected_offset') and self.expected_offset is not None:
            self.register_buffer("expected_offset", self.expected_offset[keep_mask])
        if hasattr(self, 'parent_indices') and self.parent_indices is not None:
            self.register_buffer("parent_indices", self.parent_indices[keep_mask])
        if hasattr(self, 'octant_indices') and self.octant_indices is not None:
            self.register_buffer("octant_indices", self.octant_indices[keep_mask])


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
            sh_dc=roots.sh_dc.detach().clone(),
            sh_rest=roots.sh_rest.detach().clone(),
        )
        for p in level.parameters():
            p.requires_grad_(False)
        self.levels.append(level)

    def add_level(
        self,
        split_mask: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
        child_opacity_scale: float = 1.0,
    ) -> None:
        """Add a new level by subdividing, keeping, or excluding parents."""
        assert self.depth > 0, "Must set root level first"

        parent_level = self.levels[-1]
        parents = parent_level.get_gaussians()
        N_parents = parents.num_gaussians
        device = parents.means.device

        with torch.no_grad():
            if exclude_mask is None:
                exclude_mask = torch.zeros(N_parents, dtype=torch.bool, device=device)
            if split_mask is None:
                split_mask = ~exclude_mask

            split_mask = split_mask & ~exclude_mask
            keep_mask = ~split_mask & ~exclude_mask

            if split_mask.all() and not exclude_mask.any():
                children = subdivide_to_8(parents)
                parent_means_repeated = parents.means.repeat_interleave(8, dim=0)
                parent_indices = torch.arange(N_parents, device=device).repeat_interleave(8)
                octant_indices = torch.arange(8, device=device).repeat(N_parents)
            else:
                split_indices = torch.where(split_mask)[0]
                keep_indices = torch.where(keep_mask)[0]

                split_parents = Gaussian(
                    means=parents.means[split_mask],
                    quats=parents.quats[split_mask],
                    log_scales=parents.log_scales[split_mask],
                    opacities=parents.opacities[split_mask],
                    sh_dc=parents.sh_dc[split_mask],
                    sh_rest=parents.sh_rest[split_mask],
                )
                keep_parents = Gaussian(
                    means=parents.means[keep_mask],
                    quats=parents.quats[keep_mask],
                    log_scales=parents.log_scales[keep_mask],
                    opacities=parents.opacities[keep_mask],
                    sh_dc=parents.sh_dc[keep_mask],
                    sh_rest=parents.sh_rest[keep_mask],
                )

                if split_parents.num_gaussians > 0:
                    split_children = subdivide_to_8(split_parents)
                    split_parent_means = split_parents.means.repeat_interleave(8, dim=0)
                else:
                    split_children = None

                parts_means = []
                parts_quats = []
                parts_log_scales = []
                parts_op = []
                parts_sh_dc = []
                parts_sh_rest = []
                parts_parent_means = []
                parts_parent_idx = []
                parts_octant_idx = []

                if keep_parents.num_gaussians > 0:
                    parts_means.append(keep_parents.means)
                    parts_quats.append(keep_parents.quats)
                    parts_log_scales.append(keep_parents.log_scales)
                    parts_op.append(keep_parents.opacities)
                    parts_sh_dc.append(keep_parents.sh_dc)
                    parts_sh_rest.append(keep_parents.sh_rest)
                    parts_parent_means.append(keep_parents.means)
                    parts_parent_idx.append(keep_indices)
                    parts_octant_idx.append(torch.full(
                        (keep_parents.num_gaussians,), -1,
                        dtype=torch.long, device=device))

                if split_children is not None:
                    n_split = split_parents.num_gaussians
                    parts_means.append(split_children.means)
                    parts_quats.append(split_children.quats)
                    parts_log_scales.append(split_children.log_scales)
                    parts_op.append(split_children.opacities)
                    parts_sh_dc.append(split_children.sh_dc)
                    parts_sh_rest.append(split_children.sh_rest)
                    parts_parent_means.append(split_parent_means)
                    parts_parent_idx.append(split_indices.repeat_interleave(8))
                    parts_octant_idx.append(
                        torch.arange(8, device=device).repeat(n_split))

                children = Gaussian(
                    means=torch.cat(parts_means, dim=0),
                    quats=torch.cat(parts_quats, dim=0),
                    log_scales=torch.cat(parts_log_scales, dim=0),
                    opacities=torch.cat(parts_op, dim=0),
                    sh_dc=torch.cat(parts_sh_dc, dim=0),
                    sh_rest=torch.cat(parts_sh_rest, dim=0),
                )
                parent_means_repeated = torch.cat(parts_parent_means, dim=0)
                parent_indices = torch.cat(parts_parent_idx, dim=0)
                octant_indices = torch.cat(parts_octant_idx, dim=0)

            expected_offset = (children.means - parent_means_repeated).norm(dim=-1)
            expected_offset = expected_offset.clamp(min=1e-4)

            if child_opacity_scale < 1.0:
                alpha = torch.sigmoid(children.opacities)
                alpha_scaled = (alpha * child_opacity_scale).clamp(min=1e-6, max=1.0 - 1e-6)
                children = Gaussian(
                    means=children.means,
                    quats=children.quats,
                    log_scales=children.log_scales,
                    opacities=torch.log(alpha_scaled / (1.0 - alpha_scaled)),
                    sh_dc=children.sh_dc,
                    sh_rest=children.sh_rest,
                )

        new_level = GaussianLevel(
            means=children.means.detach().clone(),
            quats=children.quats.detach().clone(),
            log_scales=children.log_scales.detach().clone(),
            opacities=children.opacities.detach().clone(),
            sh_dc=children.sh_dc.detach().clone(),
            sh_rest=children.sh_rest.detach().clone(),
        )
        new_level.register_buffer("expected_offset", expected_offset.detach().clone())
        new_level.register_buffer("parent_indices", parent_indices.detach().clone())
        new_level.register_buffer("octant_indices", octant_indices.detach().clone())
        self.levels.append(new_level)

    def get_level_gaussians(self, level: int) -> Gaussian:
        return self.levels[level].get_gaussians()

    def level_parameters(self, level: int):
        return [
            self.levels[level].means,
            self.levels[level].quats,
            self.levels[level].log_scales,
            self.levels[level].opacities,
            self.levels[level].sh_dc,
            self.levels[level].sh_rest,
        ]

    def get_gaussians_at_depth(self, target_depth: int) -> Gaussian:
        actual_depth = min(target_depth, self.depth - 1)
        return self.levels[actual_depth].get_gaussians()

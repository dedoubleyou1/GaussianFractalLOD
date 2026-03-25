"""Checkpoint save/load for root Gaussians and Gaussian tree."""

import torch
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import GaussianTree, GaussianLevel


def save_checkpoint(
    path: str | Path,
    roots: Gaussian,
    tree: GaussianTree,
    phase: int,
    level: int,
    **extra_meta,
) -> None:
    """Save training state to disk."""
    # Save per-level sizes so we can reconstruct without subdividing
    level_sizes = [tree.levels[i].num_gaussians for i in range(tree.depth)]
    sh_dims = [tree.levels[i].sh_coeffs.shape[-1] for i in range(tree.depth)]

    state = {
        "roots": {
            "means": roots.means.detach().cpu(),
            "quats": roots.quats.detach().cpu(),
            "log_scales": roots.log_scales.detach().cpu(),
            "opacities": roots.opacities.detach().cpu(),
            "sh_coeffs": roots.sh_coeffs.detach().cpu(),
        },
        "tree": tree.state_dict(),
        "tree_depth": tree.depth,
        "level_sizes": level_sizes,
        "sh_dims": sh_dims,
        "meta": {"phase": phase, "level": level, **extra_meta},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path, device: torch.device | None = None
) -> tuple[Gaussian, GaussianTree, dict]:
    """Load training state from disk."""
    state = torch.load(path, map_location="cpu", weights_only=False)

    roots = Gaussian(
        means=state["roots"]["means"],
        quats=state["roots"]["quats"],
        log_scales=state["roots"]["log_scales"],
        opacities=state["roots"]["opacities"],
        sh_coeffs=state["roots"]["sh_coeffs"],
    )

    tree = GaussianTree()

    # Reconstruct tree with correct sizes (not by subdividing)
    level_sizes = state.get("level_sizes")
    sh_dims = state.get("sh_dims")

    if level_sizes is not None:
        # New format: create levels with exact sizes
        for i, (n, sh_dim) in enumerate(zip(level_sizes, sh_dims)):
            level = GaussianLevel(
                means=torch.zeros(n, 3),
                L_flat=torch.zeros(n, 6),
                opacities=torch.zeros(n, 1),
                sh_coeffs=torch.zeros(n, sh_dim),
            )
            # Register placeholder buffers that may exist in saved state
            if not hasattr(level, 'expected_offset'):
                level.register_buffer("expected_offset", torch.zeros(n))
            if not hasattr(level, 'parent_indices'):
                level.register_buffer("parent_indices", torch.zeros(n, dtype=torch.long))
            if i == 0:
                for p in level.parameters():
                    p.requires_grad_(False)
            tree.levels.append(level)
    else:
        # Legacy format: reconstruct by subdividing (won't work with adaptive split)
        tree.set_root_level(roots)
        for _ in range(state["tree_depth"] - 1):
            tree.add_level()

    tree.load_state_dict(state["tree"], strict=False)

    if device is not None:
        roots = roots.to(device)
        tree = tree.to(device)

    return roots, tree, state["meta"]

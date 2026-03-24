"""Checkpoint save/load for root Gaussians and Gaussian tree."""

import torch
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import GaussianTree


def save_checkpoint(
    path: str | Path,
    roots: Gaussian,
    tree: GaussianTree,
    phase: int,
    level: int,
    **extra_meta,
) -> None:
    """Save training state to disk."""
    state = {
        "roots": {
            "means": roots.means.detach().cpu(),
            "L_flat": roots.L_flat.detach().cpu(),
            "opacities": roots.opacities.detach().cpu(),
            "sh_coeffs": roots.sh_coeffs.detach().cpu(),
        },
        "tree": tree.state_dict(),
        "tree_depth": tree.depth,
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
        L_flat=state["roots"]["L_flat"],
        opacities=state["roots"]["opacities"],
        sh_coeffs=state["roots"]["sh_coeffs"],
    )

    tree = GaussianTree()
    # Reconstruct tree structure: set root, then add levels
    tree.set_root_level(roots)
    for _ in range(state["tree_depth"] - 1):
        tree.add_level()
    tree.load_state_dict(state["tree"])

    if device is not None:
        roots = roots.to(device)
        tree = tree.to(device)

    return roots, tree, state["meta"]

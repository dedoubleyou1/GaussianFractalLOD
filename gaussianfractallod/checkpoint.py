"""Checkpoint save/load for root Gaussians and split tree."""

import torch
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree


def save_checkpoint(
    path: str | Path,
    roots: Gaussian,
    tree: SplitTree,
    phase: int,
    level: int,
    **extra_meta,
) -> None:
    """Save training state to disk."""
    state = {
        "roots": {
            "means": roots.means.detach().cpu(),
            "scales": roots.scales.detach().cpu(),
            "opacities": roots.opacities.detach().cpu(),
            "sh_coeffs": roots.sh_coeffs.detach().cpu(),
        },
        "tree": tree.state_dict(),
        "tree_meta": {
            "num_roots": tree.num_roots,
            "sh_dim": tree.sh_dim,
            "depth": tree.depth,
        },
        "meta": {"phase": phase, "level": level, **extra_meta},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path, device: torch.device | None = None
) -> tuple[Gaussian, SplitTree, dict]:
    """Load training state from disk."""
    state = torch.load(path, map_location="cpu", weights_only=False)

    roots = Gaussian(
        means=state["roots"]["means"],
        scales=state["roots"]["scales"],
        opacities=state["roots"]["opacities"],
        sh_coeffs=state["roots"]["sh_coeffs"],
    )

    tm = state["tree_meta"]
    tree = SplitTree(num_roots=tm["num_roots"], sh_dim=tm["sh_dim"])
    for _ in range(tm["depth"]):
        tree.add_level()
    tree.load_state_dict(state["tree"])

    if device is not None:
        roots = roots.to(device)
        tree = tree.to(device)

    return roots, tree, state["meta"]

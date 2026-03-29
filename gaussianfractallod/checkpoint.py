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
    level_sizes = [tree.levels[i].num_gaussians for i in range(tree.depth)]
    # Store sh_rest shape for reconstruction: (K-1) where K=(degree+1)²
    sh_rest_dims = [tree.levels[i].sh_rest.shape[1] for i in range(tree.depth)]

    state = {
        "roots": {
            "means": roots.means.detach().cpu(),
            "quats": roots.quats.detach().cpu(),
            "log_scales": roots.log_scales.detach().cpu(),
            "opacities": roots.opacities.detach().cpu(),
            "sh_dc": roots.sh_dc.detach().cpu(),
            "sh_rest": roots.sh_rest.detach().cpu(),
        },
        "tree": tree.state_dict(),
        "tree_depth": tree.depth,
        "level_sizes": level_sizes,
        "sh_rest_dims": sh_rest_dims,
        "meta": {"phase": phase, "level": level, **extra_meta},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path, device: torch.device | None = None
) -> tuple[Gaussian, GaussianTree, dict]:
    """Load training state from disk."""
    state = torch.load(path, map_location="cpu", weights_only=False)

    root_data = state["roots"]
    # Handle both old (sh_coeffs) and new (sh_dc + sh_rest) formats
    if "sh_dc" in root_data:
        roots = Gaussian(
            means=root_data["means"],
            quats=root_data["quats"],
            log_scales=root_data["log_scales"],
            opacities=root_data["opacities"],
            sh_dc=root_data["sh_dc"],
            sh_rest=root_data["sh_rest"],
        )
    else:
        # Legacy: sh_coeffs is flat (N, D)
        sh = root_data["sh_coeffs"]
        N = sh.shape[0]
        roots = Gaussian(
            means=root_data["means"],
            quats=root_data["quats"],
            log_scales=root_data["log_scales"],
            opacities=root_data["opacities"],
            sh_dc=sh[:, :3].reshape(N, 1, 3),
            sh_rest=sh[:, 3:].reshape(N, -1, 3) if sh.shape[1] > 3 else torch.zeros(N, 0, 3),
        )

    tree = GaussianTree()

    level_sizes = state.get("level_sizes")
    sh_rest_dims = state.get("sh_rest_dims")
    sh_dims = state.get("sh_dims")  # legacy

    if level_sizes is not None:
        for i, n in enumerate(level_sizes):
            # Determine sh_rest K-1 dimension
            if sh_rest_dims is not None:
                k_minus_1 = sh_rest_dims[i]
            elif sh_dims is not None:
                # Legacy: sh_dims stored total flat dim, convert
                k_minus_1 = max(0, sh_dims[i] // 3 - 1)
            else:
                k_minus_1 = 0

            level = GaussianLevel(
                means=torch.zeros(n, 3),
                quats=torch.zeros(n, 4),
                log_scales=torch.zeros(n, 3),
                opacities=torch.zeros(n, 1),
                sh_dc=torch.zeros(n, 1, 3),
                sh_rest=torch.zeros(n, k_minus_1, 3),
            )
            if not hasattr(level, 'expected_offset'):
                level.register_buffer("expected_offset", torch.zeros(n))
            if not hasattr(level, 'parent_indices'):
                level.register_buffer("parent_indices", torch.zeros(n, dtype=torch.long))
            if not hasattr(level, 'octant_indices'):
                level.register_buffer("octant_indices", torch.zeros(n, dtype=torch.long))
            if i == 0:
                for p in level.parameters():
                    p.requires_grad_(False)
            tree.levels.append(level)
    else:
        tree.set_root_level(roots)
        for _ in range(state["tree_depth"] - 1):
            tree.add_level()

    tree.load_state_dict(state["tree"], strict=False)

    if device is not None:
        roots = roots.to(device)
        tree = tree.to(device)

    return roots, tree, state["meta"]

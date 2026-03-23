"""Phase 2: Train split variables level-by-level with frozen parents."""

import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.reconstruct import reconstruct
from gaussianfractallod.render import render_gaussians
from gaussianfractallod.loss import rendering_loss


def train_split_level_step(
    roots: Gaussian,
    tree: SplitTree,
    target_depth: int,
    gt_image: torch.Tensor,
    camera: dict,
    optimizer: torch.optim.Optimizer,
    ssim_weight: float = 0.2,
    background: torch.Tensor | None = None,
    cached_parents: Gaussian | None = None,
    cache_depth: int = 0,
) -> torch.Tensor:
    """Single training step for split variables at target depth.

    Args:
        cached_parents: Pre-computed Gaussians from frozen levels.
            Avoids recomputing levels 0..cache_depth-1 every step.
        cache_depth: Depth of the cached_parents.
    """
    optimizer.zero_grad()
    gaussians = reconstruct(
        roots, tree, target_depth,
        cached_parents=cached_parents,
        cache_depth=cache_depth,
    )
    rendered = render_gaussians(
        gaussians, viewmat=camera["viewmat"], K=camera["K"],
        width=camera["width"], height=camera["height"], background=background,
    )
    loss = rendering_loss(rendered, gt_image, ssim_weight=ssim_weight)
    loss.backward()
    optimizer.step()
    return loss.detach()

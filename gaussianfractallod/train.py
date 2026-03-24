"""Full training orchestrator: Phase 1 (roots) + Phase 2 (levels)."""

import torch
import random
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from gaussianfractallod.config import Config
from gaussianfractallod.data import NerfSyntheticDataset
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import GaussianTree
from gaussianfractallod.train_roots import init_roots, train_roots_step
from gaussianfractallod.render import render_gaussians
from gaussianfractallod.loss import rendering_loss
from gaussianfractallod.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


def _train_level_step(
    tree: GaussianTree,
    level: int,
    gt_image: torch.Tensor,
    camera: dict,
    optimizer: torch.optim.Optimizer,
    ssim_weight: float = 0.2,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single training step for a level's Gaussians."""
    optimizer.zero_grad()

    gaussians = tree.get_level_gaussians(level)

    rendered = render_gaussians(
        gaussians,
        viewmat=camera["viewmat"],
        K=camera["K"],
        width=camera["width"],
        height=camera["height"],
        background=background,
    )

    loss = rendering_loss(rendered, gt_image, ssim_weight=ssim_weight)
    loss.backward()
    optimizer.step()

    # Clamp L_flat to prevent degenerate Gaussians
    # Diagonal (log-scale): min=-5 (no needles), max=5 (no giants)
    # Off-diagonal: clamp to prevent extreme shear
    level_module = tree.levels[level]
    with torch.no_grad():
        L = level_module.L_flat
        L[:, 0].clamp_(min=-5.0, max=5.0)  # log(l00)
        L[:, 1].clamp_(min=-5.0, max=5.0)  # l10
        L[:, 2].clamp_(min=-5.0, max=5.0)  # log(l11)
        L[:, 3].clamp_(min=-5.0, max=5.0)  # l20
        L[:, 4].clamp_(min=-5.0, max=5.0)  # l21
        L[:, 5].clamp_(min=-5.0, max=5.0)  # log(l22)
        # Clamp opacity to reasonable range
        level_module.opacities.clamp_(min=-5.0, max=10.0)

    return loss.detach()


def train(cfg: Config, resume_from: str | None = None) -> tuple[Gaussian, GaussianTree]:
    """Run full training pipeline.

    Returns:
        (roots, tree): Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = NerfSyntheticDataset(cfg.data_dir, split="train", scale=cfg.image_scale)
    logger.info(f"Loaded {len(dataset)} training images")

    sh_degree = cfg.sh_degree
    background = torch.tensor(cfg.background_color, device=device)

    writer = SummaryWriter(log_dir=str(Path(cfg.checkpoint_dir) / "logs"))

    start_phase = 1
    start_level = 0

    if resume_from:
        roots, tree, meta = load_checkpoint(resume_from, device=device)
        start_phase = meta["phase"]
        start_level = meta.get("level", 0)
        logger.info(f"Resumed from phase {start_phase}, level {start_level}")
    else:
        roots = None
        tree = None

    # ========================
    # Phase 1: Root fitting
    # ========================
    if start_phase <= 1:
        logger.info(f"Phase 1: Fitting {cfg.num_roots} root Gaussians")
        roots = init_roots(cfg.num_roots, sh_degree=sh_degree, device=device)

        optimizer = torch.optim.Adam(
            [roots.means, roots.L_flat, roots.opacities, roots.sh_coeffs],
            lr=cfg.root_lr,
        )

        best_loss = float("inf")
        plateau_count = 0

        for step in range(cfg.root_iterations):
            idx = random.randint(0, len(dataset) - 1)
            gt_image, camera = dataset[idx]
            gt_image = gt_image.to(device)
            camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera.items()}

            loss = train_roots_step(
                roots, gt_image, camera, optimizer,
                ssim_weight=cfg.ssim_weight, background=background,
            )

            writer.add_scalar("phase1/loss", loss.item(), step)

            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= cfg.root_convergence_window:
                logger.info(f"Phase 1 converged at step {step}, loss={best_loss:.6f}")
                break

            if step % 500 == 0:
                logger.info(f"Phase 1 step {step}: loss={loss.item():.6f}")

        # Freeze roots
        roots = Gaussian(
            means=roots.means.detach(),
            L_flat=roots.L_flat.detach(),
            opacities=roots.opacities.detach(),
            sh_coeffs=roots.sh_coeffs.detach(),
        )

        tree = GaussianTree().to(device)
        tree.set_root_level(roots)

        save_checkpoint(
            Path(cfg.checkpoint_dir) / "phase1_roots.pt",
            roots, tree, phase=2, level=0,
        )
        logger.info("Phase 1 complete. Roots saved.")

    # ========================
    # Phase 2: Level-by-level training
    # ========================
    for level_idx in range(start_level, cfg.max_levels):
        logger.info(f"Phase 2: Training level {level_idx + 1}")

        # Add new level (subdivides current finest into 8× children)
        if tree.depth <= level_idx + 1:
            tree.add_level()
        tree = tree.to(device)

        current_level = level_idx + 1  # level 0 = roots, level 1 = first subdivision
        num_gaussians = tree.levels[current_level].num_gaussians

        # Scale iterations: deeper levels get more training time
        level_iters = cfg.level_iterations * current_level
        logger.info(
            f"Level {current_level}: {num_gaussians} Gaussians, "
            f"{level_iters} iterations (initialized from subdivision)"
        )

        optimizer = torch.optim.Adam(
            tree.level_parameters(current_level),
            lr=cfg.level_lr,
        )

        best_loss = float("inf")
        plateau_count = 0

        for step in range(level_iters):
            idx = random.randint(0, len(dataset) - 1)
            gt_image, camera = dataset[idx]
            gt_image = gt_image.to(device)
            camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera.items()}

            loss = _train_level_step(
                tree, current_level,
                gt_image, camera, optimizer,
                ssim_weight=cfg.ssim_weight, background=background,
            )

            writer.add_scalar(f"phase2/level_{current_level}/loss", loss.item(), step)

            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= cfg.level_convergence_window:
                logger.info(
                    f"Level {current_level} converged at step {step}, loss={best_loss:.6f}"
                )
                break

            if step % 500 == 0:
                logger.info(f"Level {current_level} step {step}: loss={loss.item():.6f}")

        # Freeze this level
        for param in tree.level_parameters(current_level):
            param.requires_grad_(False)

        save_checkpoint(
            Path(cfg.checkpoint_dir) / f"phase2_level_{current_level}.pt",
            roots, tree, phase=2, level=level_idx + 1,
        )
        logger.info(f"Level {current_level} complete. {num_gaussians} Gaussians. Saved.")

    writer.close()
    return roots, tree

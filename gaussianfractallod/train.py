"""Full training orchestrator: Phase 1 (roots) + Phase 2 (splits)."""

import torch
import random
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from gaussianfractallod.config import Config
from gaussianfractallod.data import NerfSyntheticDataset
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.train_roots import init_roots, train_roots_step
from gaussianfractallod.train_splits import train_split_level_step
from gaussianfractallod.reconstruct import build_cache
from gaussianfractallod.prune import prune_level
from gaussianfractallod.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


def train(cfg: Config, resume_from: str | None = None) -> tuple[Gaussian, SplitTree]:
    """Run full training pipeline.

    Args:
        cfg: Training configuration.
        resume_from: Path to checkpoint to resume from.

    Returns:
        (roots, tree): Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = NerfSyntheticDataset(cfg.data_dir, split="train", scale=cfg.image_scale)
    logger.info(f"Loaded {len(dataset)} training images")

    sh_degree = cfg.sh_degree
    sh_dim = 3 * ((sh_degree + 1) ** 2)
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

        tree = SplitTree(num_roots=cfg.num_roots, sh_dim=sh_dim).to(device)

        save_checkpoint(
            Path(cfg.checkpoint_dir) / "phase1_roots.pt",
            roots, tree, phase=2, level=0,
        )
        logger.info("Phase 1 complete. Roots saved.")

    # ========================
    # Phase 2: Level-by-level split fitting
    # ========================
    for level in range(start_level, cfg.max_binary_depth):
        logger.info(f"Phase 2: Training level {level}")
        # Only add a new level if the tree doesn't already have it
        # (it might if we resumed from a checkpoint)
        if tree.depth <= level:
            tree.add_level()
        tree = tree.to(device)

        optimizer = torch.optim.Adam(
            tree.level_parameters(level), lr=cfg.split_lr,
        )

        target_depth = level + 1

        # Cache frozen levels — only recompute current level each step
        if level > 0:
            cached_parents = build_cache(roots, tree, level)
            logger.info(
                f"Cached {cached_parents.num_gaussians} parent Gaussians "
                f"(skipping {level} frozen levels per step)"
            )
        else:
            cached_parents = None

        best_loss = float("inf")
        plateau_count = 0

        for step in range(cfg.split_iterations_per_level):
            idx = random.randint(0, len(dataset) - 1)
            gt_image, camera = dataset[idx]
            gt_image = gt_image.to(device)
            camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera.items()}

            loss = train_split_level_step(
                roots, tree, target_depth,
                gt_image, camera, optimizer,
                ssim_weight=cfg.ssim_weight, background=background,
                cached_parents=cached_parents,
                cache_depth=level,
            )

            writer.add_scalar(f"phase2/level_{level}/loss", loss.item(), step)

            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= cfg.split_convergence_window:
                logger.info(
                    f"Level {level} converged at step {step}, loss={best_loss:.6f}"
                )
                break

            if step % 500 == 0:
                logger.info(f"Level {level} step {step}: loss={loss.item():.6f}")

        # Prune — check mass fade-out, split convergence, and low opacity
        prune_stats = prune_level(
            tree, level,
            mass_threshold=cfg.prune_mass_threshold,
        )
        logger.info(
            f"Level {level}: pruned {prune_stats['total']} "
            f"(convergence={prune_stats['convergence']}, "
            f"mass={prune_stats['mass']}, opacity={prune_stats['opacity']})"
        )

        # Freeze this level's parameters
        for param in tree.level_parameters(level):
            param.requires_grad_(False)

        save_checkpoint(
            Path(cfg.checkpoint_dir) / f"phase2_level_{level}.pt",
            roots, tree, phase=2, level=level + 1,
        )
        logger.info(f"Level {level} complete. Checkpoint saved.")

    writer.close()
    return roots, tree

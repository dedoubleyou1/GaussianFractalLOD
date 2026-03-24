"""Full training orchestrator: Phase 1 (roots) + Phase 2 (levels).

Implements 3DGS-style training techniques:
- Per-parameter learning rates
- Exponential position LR decay
- Periodic opacity reset
- Adaptive splitting based on gradient accumulation
- Scale and position regularization
- Exponentially increasing iterations per level
"""

import math
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


def _get_position_lr(cfg: Config, step: int, max_steps: int) -> float:
    """Exponential decay for position learning rate (3DGS style)."""
    if max_steps <= 1:
        return cfg.lr_means
    t = min(step / max_steps, 1.0)
    log_lr = math.log(cfg.lr_means) * (1 - t) + math.log(cfg.lr_means_final) * t
    return math.exp(log_lr)


def _make_optimizer(cfg: Config, level_module) -> torch.optim.Adam:
    """Create per-parameter-group optimizer (3DGS style)."""
    return torch.optim.Adam([
        {"params": [level_module.means], "lr": cfg.lr_means, "name": "means"},
        {"params": [level_module.L_flat], "lr": cfg.lr_L_flat, "name": "L_flat"},
        {"params": [level_module.opacities], "lr": cfg.lr_opacities, "name": "opacities"},
        {"params": [level_module.sh_coeffs], "lr": cfg.lr_sh_coeffs, "name": "sh_coeffs"},
    ], eps=1e-15)


def _update_position_lr(optimizer: torch.optim.Adam, new_lr: float) -> None:
    """Update position learning rate in optimizer."""
    for param_group in optimizer.param_groups:
        if param_group.get("name") == "means":
            param_group["lr"] = new_lr


def _train_level_step(
    tree: GaussianTree,
    level: int,
    gt_image: torch.Tensor,
    camera: dict,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
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

    loss = rendering_loss(rendered, gt_image, ssim_weight=cfg.ssim_weight)

    # Regularization
    level_module = tree.levels[level]

    # Position: drift as fraction of expected offset from parent
    if hasattr(level_module, 'expected_offset'):
        drift = (level_module.means - level_module.init_means).norm(dim=-1)
        normalized_drift = drift / (level_module.expected_offset + 1e-8)
        pos_reg = normalized_drift.pow(2).mean()
    else:
        pos_reg = torch.tensor(0.0, device=loss.device)

    # Scale: exponential cost for deviating from initial (log-ratio)
    diag_idx = [0, 2, 5]
    log_ratio = gaussians.L_flat[:, diag_idx] - level_module.init_L_flat[:, diag_idx]
    scale_reg = torch.exp(log_ratio.abs()).mean()

    total_loss = (
        loss
        + cfg.reg_scale_weight * scale_reg
        + cfg.reg_position_weight * pos_reg
    )
    total_loss.backward()

    # Accumulate gradients for adaptive splitting decisions
    level_module.accumulate_grad()

    optimizer.step()

    return loss.detach()


def train(cfg: Config, resume_from: str | None = None) -> tuple[Gaussian, GaussianTree]:
    """Run full training pipeline."""
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
            lr=cfg.root_lr, eps=1e-15,
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
        current_level = level_idx + 1

        # --- Adaptive splitting: use gradient signal from previous level ---
        split_mask = None
        prev_level = tree.levels[current_level - 1]
        if current_level > 1 and prev_level.grad_count.sum() > 0:
            avg_grads = prev_level.avg_grad()
            split_mask = avg_grads > cfg.split_grad_threshold
            n_split = split_mask.sum().item()
            n_total = split_mask.shape[0]
            logger.info(
                f"Adaptive split: {n_split}/{n_total} parents selected "
                f"(grad threshold={cfg.split_grad_threshold:.4f}, "
                f"mean grad={avg_grads.mean():.4f})"
            )

        # Add new level
        if tree.depth <= current_level:
            tree.add_level(split_mask=split_mask)
        tree = tree.to(device)

        level_module = tree.levels[current_level]
        num_gaussians = level_module.num_gaussians

        # Exponential iterations: base * 2^(level-1)
        level_iters = cfg.level_base_iterations * (2 ** (current_level - 1))
        logger.info(
            f"Level {current_level}: {num_gaussians} Gaussians, "
            f"{level_iters} iterations"
        )

        # Per-parameter optimizer
        optimizer = _make_optimizer(cfg, level_module)

        best_loss = float("inf")
        plateau_count = 0

        for step in range(level_iters):
            # Decay position LR
            pos_lr = _get_position_lr(cfg, step, level_iters)
            _update_position_lr(optimizer, pos_lr)

            idx = random.randint(0, len(dataset) - 1)
            gt_image, camera = dataset[idx]
            gt_image = gt_image.to(device)
            camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera.items()}

            loss = _train_level_step(
                tree, current_level,
                gt_image, camera, optimizer,
                cfg=cfg, background=background,
            )

            writer.add_scalar(f"phase2/level_{current_level}/loss", loss.item(), step)

            # Opacity reset
            if (step + 1) % cfg.opacity_reset_interval == 0 and step < level_iters - 100:
                level_module.reset_opacity(cfg.opacity_reset_value)
                # Reset optimizer state for opacity
                for pg in optimizer.param_groups:
                    if pg.get("name") == "opacities":
                        for p in pg["params"]:
                            state = optimizer.state.get(p)
                            if state:
                                state["exp_avg"].zero_()
                                state["exp_avg_sq"].zero_()
                logger.info(f"Level {current_level} step {step}: opacity reset")

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

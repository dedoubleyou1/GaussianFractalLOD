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
import torch.nn.functional as F
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
        {"params": [level_module.quats], "lr": cfg.lr_quats, "name": "quats"},
        {"params": [level_module.log_scales], "lr": cfg.lr_log_scales, "name": "log_scales"},
        {"params": [level_module.opacities], "lr": cfg.lr_opacities, "name": "opacities"},
        {"params": [level_module.sh_coeffs], "lr": cfg.lr_sh_coeffs, "name": "sh_coeffs"},
    ], eps=1e-15)


def _update_position_lr(optimizer: torch.optim.Adam, new_lr: float) -> None:
    """Update position learning rate in optimizer."""
    for param_group in optimizer.param_groups:
        if param_group.get("name") == "means":
            param_group["lr"] = new_lr


def _normalize_quaternions(level_module) -> None:
    """Normalize quaternions after optimizer step."""
    with torch.no_grad():
        level_module.quats.data = F.normalize(level_module.quats.data, dim=-1)


def _train_level_step(
    tree: GaussianTree,
    level: int,
    gt_image: torch.Tensor,
    gt_image_hires: torch.Tensor | None,
    camera: dict,
    camera_hires: dict | None,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single training step for a level's Gaussians.

    If gt_image_hires and camera_hires are provided, also renders
    hypothetical children (on-the-fly subdivision, no extra params)
    at higher resolution for additional gradient signal.
    """
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

    # Hypothetical children: subdivide on-the-fly and render at higher res
    # Gradients flow through subdivision back to parent parameters
    if gt_image_hires is not None and camera_hires is not None:
        from gaussianfractallod.subdivide import subdivide_to_8
        hypothetical = subdivide_to_8(gaussians)
        rendered_hires = render_gaussians(
            hypothetical,
            viewmat=camera_hires["viewmat"],
            K=camera_hires["K"],
            width=camera_hires["width"],
            height=camera_hires["height"],
            background=background,
        )
        loss_children = rendering_loss(rendered_hires, gt_image_hires, ssim_weight=cfg.ssim_weight)
        loss = loss + loss_children

    # Regularization
    level_module = tree.levels[level]

    # Position: drift as fraction of expected offset from parent
    if hasattr(level_module, 'expected_offset'):
        drift = (level_module.means - level_module.init_means).norm(dim=-1)
        normalized_drift = drift / (level_module.expected_offset + 1e-8)
        pos_reg = normalized_drift.pow(2).mean()
    else:
        pos_reg = torch.tensor(0.0, device=loss.device)

    # Scale: penalize volume change (mean-of-axes), free to change shape
    mean_log_ratio = (gaussians.log_scales - level_module.init_log_scales).mean(dim=-1)
    scale_reg = torch.exp(mean_log_ratio * mean_log_ratio).mean()

    # Aspect ratio: absolute dead-zone + exp wall. No penalty up to dead_zone
    # aspect ratio, then exp(x²) ramps up. Anchored to 1:1, not parent shape.
    import math
    spread = (gaussians.log_scales.max(dim=-1).values
              - gaussians.log_scales.min(dim=-1).values)
    dead_zone = math.log(cfg.aspect_dead_zone)
    delta_spread = torch.clamp(spread - dead_zone, min=0.0)
    aspect_reg = torch.exp(delta_spread * delta_spread).mean()

    total_loss = (
        loss
        + cfg.reg_scale_weight * scale_reg
        + cfg.reg_position_weight * pos_reg
        + cfg.reg_aspect_weight * aspect_reg
    )
    total_loss.backward()

    # Accumulate gradients for adaptive splitting decisions
    level_module.accumulate_grad()

    optimizer.step()

    # Normalize quaternions after optimizer step
    _normalize_quaternions(level_module)

    # Hard clamp aspect ratio — trivial with log_scales (no eigendecomposition)
    if cfg.max_aspect_ratio > 0:
        import math
        max_log_ratio = math.log(cfg.max_aspect_ratio)
        with torch.no_grad():
            ls = level_module.log_scales
            max_ls = ls.max(dim=-1, keepdim=True).values
            ls.clamp_(min=max_ls - max_log_ratio)

    return loss.detach()


def _get_level_resolution(level: int) -> int:
    """Training resolution for a level. Anchored from bottom (level 0 = 32px).

    Growth: √2× per level (doubling pixel count). Derived from observed
    Gaussian scale shrinkage (~0.64× per level) and minimum projection
    target (~3σ pixels). Caps at 800px (dataset resolution).

    Every even level is a power of 2: 32, 64, 128, 256, 512.
    """
    res = round(32.0 * (2 ** 0.5) ** level)
    return min(res, 800)


def _load_dataset_for_level(cfg: Config, level: int) -> NerfSyntheticDataset:
    """Load dataset at appropriate resolution for this level."""
    res = _get_level_resolution(level)
    scale = res / 800.0
    dataset = NerfSyntheticDataset(cfg.data_dir, split="train", scale=scale)
    return dataset


def train(cfg: Config, resume_from: str | None = None) -> tuple[Gaussian, GaussianTree]:
    """Run full training pipeline."""
    # Seed all RNGs for reproducibility
    import numpy as np
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device} (seed={cfg.seed})")

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
        root_res = _get_level_resolution(0)
        dataset_root = _load_dataset_for_level(cfg, 0)
        logger.info(
            f"Phase 1: Fitting {cfg.num_roots} root Gaussians "
            f"(resolution={root_res}px)"
        )
        roots = init_roots(cfg.num_roots, sh_degree=sh_degree, device=device)

        optimizer = torch.optim.Adam(
            [roots.means, roots.quats, roots.log_scales,
             roots.opacities, roots.sh_coeffs],
            lr=cfg.root_lr, eps=1e-15,
        )

        best_loss = float("inf")
        plateau_count = 0

        for step in range(cfg.root_iterations):
            idx = random.randint(0, len(dataset_root) - 1)
            gt_rgb, gt_alpha, camera = dataset_root[idx]
            gt_rgb = gt_rgb.to(device)
            gt_alpha = gt_alpha.to(device)
            camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera.items()}

            rand_bg = torch.rand(3, device=device)
            gt_image = gt_rgb * gt_alpha + (1.0 - gt_alpha) * rand_bg.view(1, 1, 3)

            loss = train_roots_step(
                roots, gt_image, camera, optimizer,
                ssim_weight=cfg.ssim_weight, background=rand_bg,
            )

            # Normalize quaternions after optimizer step
            with torch.no_grad():
                roots.quats.data = F.normalize(roots.quats.data, dim=-1)

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
            quats=roots.quats.detach(),
            log_scales=roots.log_scales.detach(),
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
        exclude_mask = None
        prev_level = tree.levels[current_level - 1]
        if current_level > 1 and prev_level.grad_count.sum() > 0:
            max_grad, mean_grad = prev_level.split_scores()
            parent_opacity = torch.sigmoid(prev_level.opacities).squeeze(-1)

            # Gradient OR: split if any view needs detail OR consistent coverage gap
            split_mask = (max_grad > cfg.split_max_threshold) | (mean_grad > cfg.split_mean_threshold)
            # Opacity: exclude near-transparent parents entirely (no children)
            exclude_mask = parent_opacity < cfg.split_min_opacity

            n_total = split_mask.shape[0]
            n_split = (split_mask & ~exclude_mask).sum().item()
            n_keep = (~split_mask & ~exclude_mask).sum().item()
            n_exclude = exclude_mask.sum().item()

            # Gradient statistics for threshold calibration
            alive = ~exclude_mask
            logger.info(
                f"Adaptive split: {n_split} subdivide, {n_keep} keep, "
                f"{n_exclude} exclude (of {n_total} parents)"
            )
            logger.info(
                f"  Grad stats (alive parents): "
                f"max_grad: mean={max_grad[alive].mean():.5f}, "
                f"median={max_grad[alive].median():.5f}, "
                f"p90={max_grad[alive].quantile(0.9):.5f}, "
                f"max={max_grad[alive].max():.5f} | "
                f"mean_grad: mean={mean_grad[alive].mean():.6f}, "
                f"median={mean_grad[alive].median():.6f}, "
                f"max={mean_grad[alive].max():.6f}"
            )

        # Add new level
        if tree.depth <= current_level:
            tree.add_level(
                split_mask=split_mask,
                exclude_mask=exclude_mask,
                child_opacity_scale=cfg.child_opacity_scale,
            )
        tree = tree.to(device)

        level_module = tree.levels[current_level]
        num_gaussians = level_module.num_gaussians

        # Load dataset at appropriate resolution for this level
        level_res = _get_level_resolution(current_level)
        dataset = _load_dataset_for_level(cfg, current_level)
        num_views = len(dataset)

        # Load higher-res dataset for hypothetical children rendering
        dataset_hires = None
        if current_level < cfg.max_levels:
            dataset_hires = _load_dataset_for_level(cfg, current_level + 1)

        # Epoch-based iterations
        level_iters = cfg.level_epochs * num_views
        logger.info(
            f"Level {current_level}: {num_gaussians} Gaussians, "
            f"{level_res}px, {cfg.level_epochs} epochs ({level_iters} steps)"
        )

        # Per-parameter optimizer
        optimizer = _make_optimizer(cfg, level_module)

        best_loss = float("inf")
        best_loss_epoch = 0
        epoch_losses = []
        epoch_loss_accum = 0.0
        epoch_loss_count = 0

        for step in range(level_iters):
            # Decay position LR
            pos_lr = _get_position_lr(cfg, step, level_iters)
            _update_position_lr(optimizer, pos_lr)

            idx = random.randint(0, num_views - 1)
            gt_rgb, gt_alpha, camera = dataset[idx]
            gt_rgb = gt_rgb.to(device)
            gt_alpha = gt_alpha.to(device)
            camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera.items()}

            # Random background forces proper opacity learning:
            # renderer fills transparent regions with rand_bg, GT composited
            # with same rand_bg, so the only way to match is correct opacity
            rand_bg = torch.rand(3, device=device)
            gt_image = gt_rgb * gt_alpha + (1.0 - gt_alpha) * rand_bg.view(1, 1, 3)

            # Higher-res image for hypothetical children (same view, same background)
            gt_hires, cam_hires = None, None
            if dataset_hires is not None:
                gt_rgb_hr, gt_alpha_hr, cam_hires = dataset_hires[idx]
                gt_rgb_hr = gt_rgb_hr.to(device)
                gt_alpha_hr = gt_alpha_hr.to(device)
                gt_hires = gt_rgb_hr * gt_alpha_hr + (1.0 - gt_alpha_hr) * rand_bg.view(1, 1, 3)
                cam_hires = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in cam_hires.items()}

            loss = _train_level_step(
                tree, current_level,
                gt_image, gt_hires,
                camera, cam_hires,
                optimizer,
                cfg=cfg, background=rand_bg,
            )

            writer.add_scalar(f"phase2/level_{current_level}/loss", loss.item(), step)

            epoch_loss_accum += loss.item()
            epoch_loss_count += 1

            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                best_loss_epoch = step / num_views

            if (step + 1) % num_views == 0:
                epoch = (step + 1) / num_views
                avg_loss = epoch_loss_accum / epoch_loss_count
                epoch_losses.append(avg_loss)
                logger.info(
                    f"Level {current_level} epoch {epoch:.0f}: "
                    f"loss={avg_loss:.6f}"
                )
                epoch_loss_accum = 0.0
                epoch_loss_count = 0

        # Freeze this level
        for param in tree.level_parameters(current_level):
            param.requires_grad_(False)

        convergence_info = {
            "best_loss": best_loss,
            "best_loss_epoch": best_loss_epoch,
            "total_epochs": cfg.level_epochs,
            "epoch_losses": epoch_losses,
            "num_gaussians": num_gaussians,
            "resolution": level_res,
        }
        save_checkpoint(
            Path(cfg.checkpoint_dir) / f"phase2_level_{current_level}.pt",
            roots, tree, phase=2, level=level_idx + 1,
            convergence=convergence_info,
        )
        epochs_since_best = cfg.level_epochs - best_loss_epoch
        logger.info(
            f"Level {current_level} complete. {num_gaussians} Gaussians. "
            f"Best loss={best_loss:.6f} at epoch {best_loss_epoch:.1f} "
            f"({epochs_since_best:.1f} epochs ago). Saved."
        )

    writer.close()
    return roots, tree

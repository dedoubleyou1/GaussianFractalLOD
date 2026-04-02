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
from gaussianfractallod.train_roots import init_roots, train_roots_step, fit_roots_lbfgs, fit_roots_silhouette
from gaussianfractallod.geometric_gaussian_fit import fit_gaussian_to_views
from gaussianfractallod.render import render_gaussians
from gaussianfractallod.loss import rendering_loss
from gaussianfractallod.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


def _get_position_lr(cfg: Config, step: int, max_steps: int) -> float:
    """Exponential decay with cosine warmup for position LR (3DGS style).

    Matches the reference 3DGS implementation:
    - Cosine warmup from delay_mult * lr to lr over first 1% of steps
    - Then log-linear decay from lr_means to lr_means_final
    """
    if max_steps <= 1:
        return cfg.lr_means
    t = min(step / max_steps, 1.0)
    log_lr = math.log(cfg.lr_means) * (1 - t) + math.log(cfg.lr_means_final) * t
    lr = math.exp(log_lr)

    # Cosine warmup over first 1% of steps (delay_mult=0.01 per 3DGS)
    delay_steps = max(max_steps // 100, 1)
    delay_mult = 0.01
    if step < delay_steps:
        warmup = delay_mult + (1 - delay_mult) * math.sin(
            0.5 * math.pi * step / delay_steps
        )
        lr *= warmup

    return lr


def _make_optimizer(cfg: Config, level_module) -> torch.optim.Adam:
    """Create per-parameter-group optimizer (3DGS style).

    SH DC and rest have separate learning rates per standard 3DGS:
    rest LR is 20× lower to prevent view-dependent overfitting.
    """
    return torch.optim.Adam([
        {"params": [level_module.delta_means], "lr": cfg.lr_means, "name": "means"},
        {"params": [level_module.quats], "lr": cfg.lr_quats, "name": "quats"},
        {"params": [level_module.delta_log_scales], "lr": cfg.lr_log_scales, "name": "log_scales"},
        {"params": [level_module.delta_opacities], "lr": cfg.lr_opacities, "name": "opacities"},
        {"params": [level_module.delta_sh_dc], "lr": cfg.lr_sh_dc, "name": "sh_dc"},
        {"params": [level_module.delta_sh_rest], "lr": cfg.lr_sh_rest, "name": "sh_rest"},
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
    active_sh_degree: int | None = None,
    gt_moments: dict | None = None,
) -> torch.Tensor:
    """Single training step for a level's Gaussians.

    If gt_image_hires and camera_hires are provided, also renders
    hypothetical children (on-the-fly subdivision, no extra params)
    at higher resolution for additional gradient signal.
    """
    optimizer.zero_grad()

    gaussians = tree.get_level_gaussians(level)

    use_moments = gt_moments is not None and (cfg.reg_centroid_weight > 0 or cfg.reg_covariance_weight > 0)

    render_result = render_gaussians(
        gaussians,
        viewmat=camera["viewmat"],
        K=camera["K"],
        width=camera["width"],
        height=camera["height"],
        background=background,
        return_alpha=use_moments,
    )

    if use_moments:
        rendered, render_alpha = render_result
    else:
        rendered = render_result

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

    # Regularization (delta parameterization: deltas are the parameters directly)
    level_module = tree.levels[level]

    # Position: delta drift as fraction of expected offset from parent
    if hasattr(level_module, 'expected_offset'):
        drift = level_module.delta_means.norm(dim=-1)
        normalized_drift = drift / (level_module.expected_offset + 1e-8)
        pos_reg = normalized_drift.pow(2).mean()
    else:
        pos_reg = torch.tensor(0.0, device=loss.device)

    # Scale: penalize volume change (mean-of-axes delta)
    mean_log_ratio = level_module.delta_log_scales.mean(dim=-1)
    scale_reg = torch.exp(mean_log_ratio * mean_log_ratio).mean()

    # Aspect ratio: absolute dead-zone + exp wall
    spread = (gaussians.log_scales.max(dim=-1).values
              - gaussians.log_scales.min(dim=-1).values)
    dead_zone = math.log(cfg.aspect_dead_zone)
    delta_spread = torch.clamp(spread - dead_zone, min=0.0)
    aspect_reg = torch.exp(delta_spread * delta_spread).mean()

    # Moment regularization: encourage rendered silhouette to match GT shape
    centroid_reg = torch.tensor(0.0, device=loss.device)
    cov_reg = torch.tensor(0.0, device=loss.device)
    if use_moments:
        from gaussianfractallod.metrics import moment_loss
        # render_alpha is (H, W) from gsplat
        alpha_2d = render_alpha.squeeze()
        centroid_reg, cov_reg = moment_loss(
            alpha_2d,
            gt_moments["centroid"], gt_moments["covariance"],
            gt_moments["yy"], gt_moments["xx"],
            gt_moments["diagonal"],
        )

    total_loss = (
        loss
        + cfg.reg_scale_weight * scale_reg
        + cfg.reg_position_weight * pos_reg
        + cfg.reg_aspect_weight * aspect_reg
        + cfg.reg_centroid_weight * centroid_reg
        + cfg.reg_covariance_weight * cov_reg
    )
    total_loss.backward()

    # Zero out gradients for inactive SH bands (progressive activation)
    if active_sh_degree is not None and active_sh_degree < cfg.sh_degree:
        if active_sh_degree == 0:
            if level_module.delta_sh_rest.grad is not None:
                level_module.delta_sh_rest.grad.zero_()
        else:
            active_rest = (active_sh_degree + 1) ** 2 - 1
            if level_module.delta_sh_rest.grad is not None:
                level_module.delta_sh_rest.grad[:, active_rest:, :] = 0.0

    # Accumulate gradients for adaptive splitting decisions
    level_module.accumulate_grad()

    optimizer.step()

    # Normalize quaternions after optimizer step
    _normalize_quaternions(level_module)

    # Hard clamp aspect ratio on reconstructed log_scales
    if cfg.max_aspect_ratio > 0:
        max_log_ratio = math.log(cfg.max_aspect_ratio)
        with torch.no_grad():
            ls = level_module.init_log_scales + level_module.delta_log_scales
            max_ls = ls.max(dim=-1, keepdim=True).values
            # Clamp by adjusting delta: new_delta = clamped_abs - init
            clamped = ls.clamp(min=max_ls - max_log_ratio)
            level_module.delta_log_scales.data.copy_(clamped - level_module.init_log_scales)

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
        roots = init_roots(cfg.num_roots, sh_degree=sh_degree, device=device,
                           dataset=dataset_root)

        if cfg.root_geometric:
            # Closed-form: derives Gaussian from image moments, no optimization.
            # Position, scale, quaternion, opacity from geometry.
            # Then train SH coefficients to learn color/view-dependence.
            logger.info("Using geometric fit from image moments")
            roots = fit_gaussian_to_views(dataset_root, device, sh_degree=sh_degree)

            # Phase 1b: SH refinement — geometry frozen, only color trains
            logger.info("Phase 1b: Training SH on geometric root")
            sh_optimizer = torch.optim.Adam([
                {"params": [roots.sh_dc], "lr": cfg.lr_sh_dc},
                {"params": [roots.sh_rest], "lr": cfg.lr_sh_rest},
            ], eps=1e-15)

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
                    roots, gt_image, camera, sh_optimizer,
                    ssim_weight=cfg.ssim_weight, background=rand_bg,
                )
                writer.add_scalar("phase1b/loss", loss.item(), step)

                if step % 500 == 0:
                    logger.info(f"Phase 1b step {step}: loss={loss.item():.6f}")

        elif cfg.root_silhouette:
            # Silhouette-based: prioritizes spatial coverage over color accuracy
            logger.info("Using silhouette-based L-BFGS for root fitting")
            roots = fit_roots_silhouette(roots, dataset_root, device)
        elif cfg.root_lbfgs:
            # L-BFGS: fast quasi-Newton for the small root parameter space
            logger.info("Using L-BFGS for root fitting")
            roots = fit_roots_lbfgs(roots, dataset_root, device)
        else:
            # Adam: standard stochastic gradient descent
            optimizer = torch.optim.Adam([
                {"params": [roots.means], "lr": cfg.lr_means},
                {"params": [roots.quats], "lr": cfg.lr_quats},
                {"params": [roots.log_scales], "lr": cfg.lr_log_scales},
                {"params": [roots.opacities], "lr": cfg.lr_opacities},
                {"params": [roots.sh_dc], "lr": cfg.lr_sh_dc},
                {"params": [roots.sh_rest], "lr": cfg.lr_sh_rest},
            ], eps=1e-15)

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
            sh_dc=roots.sh_dc.detach(),
            sh_rest=roots.sh_rest.detach(),
        )

        tree = GaussianTree(quantize_bits=cfg.quantize_bits).to(device)
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

        # --- Adaptive splitting: tiered cuts based on gradient ---
        cuts_per_parent = None
        if tree.depth < current_level:
            logger.info(f"Level {current_level}: previous level was skipped, stopping.")
            break
        prev_level = tree.levels[current_level - 1]
        if current_level > 1 and prev_level.grad_count.sum() > 0:
            max_grad, mean_grad = prev_level.split_scores()
            parent_opacity = torch.sigmoid(
                prev_level.init_opacities + prev_level.delta_opacities
            ).squeeze(-1)

            N_parents = max_grad.shape[0]
            device = max_grad.device

            # Start with all kept (0), then assign tiers based on gradient
            cuts = torch.zeros(N_parents, dtype=torch.long, device=device)
            cuts[max_grad > cfg.split_1cut_threshold] = 1  # 2 children
            cuts[max_grad > cfg.split_2cut_threshold] = 2  # 4 children
            cuts[max_grad > cfg.split_3cut_threshold] = 3  # 8 children

            # Exclude near-transparent parents
            exclude_mask = parent_opacity < cfg.split_min_opacity
            cuts[exclude_mask] = -1

            n_3cut = (cuts == 3).sum().item()
            n_2cut = (cuts == 2).sum().item()
            n_1cut = (cuts == 1).sum().item()
            n_keep = (cuts == 0).sum().item()
            n_exclude = (cuts == -1).sum().item()
            n_children = n_3cut * 8 + n_2cut * 4 + n_1cut * 2 + n_keep

            alive = cuts >= 0
            logger.info(
                f"Tiered split: 8ch={n_3cut}, 4ch={n_2cut}, 2ch={n_1cut}, "
                f"keep={n_keep}, exclude={n_exclude} → {n_children} children"
            )
            if alive.any():
                logger.info(
                    f"  Grad stats (alive): "
                    f"max_grad: mean={max_grad[alive].mean():.5f}, "
                    f"median={max_grad[alive].median():.5f}, "
                    f"p90={max_grad[alive].quantile(0.9):.5f}, "
                    f"max={max_grad[alive].max():.5f}"
                )

            # Skip if all excluded
            if n_children == 0:
                logger.info(f"Level {current_level}: all parents excluded, skipping")
                continue

            cuts_per_parent = cuts

        # Add new level
        if tree.depth <= current_level:
            tree.add_level(
                cuts_per_parent=cuts_per_parent,
                opacity_floor=cfg.child_opacity_floor,
                opacity_scale=cfg.child_opacity_scale,
                opacity_formula=cfg.child_opacity_formula,
            )
        tree = tree.to(device)

        level_module = tree.levels[current_level]
        num_gaussians = level_module.num_gaussians

        # Load dataset at appropriate resolution for this level
        level_res = _get_level_resolution(current_level)
        dataset = _load_dataset_for_level(cfg, current_level)
        num_views = len(dataset)

        # Precompute GT alpha moments for silhouette regularization
        gt_moments_cache = None
        if cfg.reg_centroid_weight > 0 or cfg.reg_covariance_weight > 0:
            from gaussianfractallod.metrics import compute_alpha_moments
            gt_moments_cache = {}
            # Coordinate grids shared across views (same resolution per level)
            sample_alpha = dataset[0][1].numpy().squeeze(-1)
            H, W = sample_alpha.shape
            diagonal = math.sqrt(H**2 + W**2)
            yy, xx = torch.meshgrid(
                torch.arange(H, dtype=torch.float32, device=device),
                torch.arange(W, dtype=torch.float32, device=device),
                indexing="ij",
            )
            for vi in range(num_views):
                _, gt_alpha_vi, _ = dataset[vi]
                alpha_np = gt_alpha_vi.numpy().squeeze(-1)
                moments = compute_alpha_moments(alpha_np)
                if moments["centroid"] is not None:
                    gt_moments_cache[vi] = {
                        "centroid": torch.tensor(moments["centroid"], dtype=torch.float32, device=device),
                        "covariance": torch.tensor(moments["covariance"], dtype=torch.float32, device=device),
                        "yy": yy,
                        "xx": xx,
                        "diagonal": diagonal,
                    }
            logger.info(f"Precomputed GT moments for {len(gt_moments_cache)}/{num_views} views")

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

        # Early stopping state (running avg with window=3, patience=6)
        _ES_WINDOW = 3
        _ES_PATIENCE = 6
        best_running_avg = float("inf")
        best_running_epoch = 0
        stopped_early = False

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

            # Progressive SH activation: unlock one band per sh_band_epochs
            active_sh = None
            if cfg.sh_band_epochs > 0 and cfg.sh_degree > 0:
                epoch = step // num_views
                active_sh = min(epoch // cfg.sh_band_epochs, cfg.sh_degree)

            view_moments = gt_moments_cache.get(idx) if gt_moments_cache else None

            loss = _train_level_step(
                tree, current_level,
                gt_image, gt_hires,
                camera, cam_hires,
                optimizer,
                cfg=cfg, background=rand_bg,
                active_sh_degree=active_sh,
                gt_moments=view_moments,
            )

            writer.add_scalar(f"phase2/level_{current_level}/loss", loss.item(), step)

            epoch_loss_accum += loss.item()
            epoch_loss_count += 1

            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                best_loss_epoch = step / num_views

            if (step + 1) % num_views == 0:
                epoch = int((step + 1) / num_views)
                avg_loss = epoch_loss_accum / epoch_loss_count
                epoch_losses.append(avg_loss)
                logger.info(
                    f"Level {current_level} epoch {epoch}: "
                    f"loss={avg_loss:.6f}"
                )
                epoch_loss_accum = 0.0
                epoch_loss_count = 0

                # Early stopping: running average of epoch losses
                window_start = max(0, len(epoch_losses) - _ES_WINDOW)
                running_avg = sum(epoch_losses[window_start:]) / (len(epoch_losses) - window_start)

                if running_avg < best_running_avg - 1e-6:
                    best_running_avg = running_avg
                    best_running_epoch = epoch
                elif epoch - best_running_epoch >= _ES_PATIENCE:
                    logger.info(
                        f"Level {current_level} early stop at epoch {epoch} "
                        f"(no improvement since epoch {best_running_epoch})"
                    )
                    stopped_early = True
                    break

        # Freeze this level
        for param in tree.level_parameters(current_level):
            param.requires_grad_(False)

        actual_epochs = len(epoch_losses)
        convergence_info = {
            "best_loss": best_loss,
            "best_loss_epoch": best_loss_epoch,
            "total_epochs": cfg.level_epochs,
            "actual_epochs": actual_epochs,
            "stopped_early": stopped_early,
            "epoch_losses": epoch_losses,
            "num_gaussians": num_gaussians,
            "resolution": level_res,
        }
        save_checkpoint(
            Path(cfg.checkpoint_dir) / f"phase2_level_{current_level}.pt",
            roots, tree, phase=2, level=level_idx + 1,
            convergence=convergence_info,
        )
        logger.info(
            f"Level {current_level} complete. {num_gaussians} Gaussians. "
            f"Best loss={best_loss:.6f}. "
            f"{actual_epochs}/{cfg.level_epochs} epochs"
            f"{' (early stop)' if stopped_early else ''}. Saved."
        )

    writer.close()
    return roots, tree

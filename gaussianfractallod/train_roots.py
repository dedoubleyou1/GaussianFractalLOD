"""Phase 1: Train root-level Gaussians.

Supports two initialization modes:
- Data-driven init + Adam gradient descent (default)
- NLLS via L-BFGS for fast convergence on the small parameter space
"""

import torch
import torch.nn.functional as F
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.render import render_gaussians
from gaussianfractallod.loss import rendering_loss


def init_roots(
    num_roots: int, sh_degree: int = 0, device: torch.device = torch.device("cpu"),
    dataset=None,
) -> Gaussian:
    """Initialize root Gaussians from dataset statistics.

    If a dataset is provided, computes:
    - Position: scene center (average camera look-at point, ~origin for NeRF synthetic)
    - Scale: scene extent from camera positions
    - Color: average color across training views
    - Opacity: high (fully opaque)

    Falls back to random initialization if no dataset is given.
    """
    import math
    num_sh = (sh_degree + 1) ** 2

    if dataset is not None and len(dataset) > 0:
        # Compute scene center from camera positions
        cam_positions = []
        avg_color = torch.zeros(3)
        n_pixels = 0
        for i in range(len(dataset)):
            gt_rgb, gt_alpha, camera = dataset[i]
            # cam_pos = -R^T @ t is the camera origin in world space.
            # Valid for any rigid-body w2c regardless of axis convention.
            w2c = camera["viewmat"]
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            cam_pos = -R.T @ t
            cam_positions.append(cam_pos)
            # Average color (alpha-weighted)
            rgb_masked = gt_rgb * gt_alpha
            avg_color += rgb_masked.sum(dim=(0, 1))
            n_pixels += gt_alpha.sum()

        cam_positions = torch.stack(cam_positions)
        # Scene center: where cameras look at (for NeRF synthetic, near origin)
        # Approximate as the point closest to all camera rays
        scene_center = torch.zeros(3)  # NeRF synthetic is centered at origin

        # Scene extent: radius that covers the object
        # Object is roughly within the camera orbit
        cam_radius = cam_positions.norm(dim=-1).mean()
        # Object extent is roughly 1/4 to 1/3 of camera radius
        scene_extent = cam_radius * 0.3
        log_scale = math.log(max(scene_extent.item(), 0.01))

        # Average color → SH DC coefficient
        avg_color = avg_color / n_pixels.clamp(min=1)
        C0 = 0.28209479177387814
        sh_dc_init = ((avg_color - 0.5) / C0).reshape(1, 1, 3)

        means = scene_center.unsqueeze(0).expand(num_roots, 3).clone()
        if num_roots > 1:
            means += torch.randn(num_roots, 3) * scene_extent * 0.1
        means = means.to(device).requires_grad_(True)

        log_scales = torch.full((num_roots, 3), log_scale, device=device)
        log_scales.requires_grad_(True)

        opacities = torch.full((num_roots, 1), 0.0, device=device)  # sigmoid(0) = 0.5
        opacities.requires_grad_(True)

        sh_dc = sh_dc_init.expand(num_roots, 1, 3).clone().to(device)
        sh_dc.requires_grad_(True)
    else:
        # Fallback: random initialization
        means = torch.randn(num_roots, 3, device=device) * 0.5
        means.requires_grad_(True)

        log_scales = torch.full((num_roots, 3), -1.0, device=device)
        log_scales.requires_grad_(True)

        opacities = torch.full((num_roots, 1), 2.0, device=device)
        opacities.requires_grad_(True)

        sh_dc = torch.randn(num_roots, 1, 3, device=device) * 0.1
        sh_dc.requires_grad_(True)

    # Identity quaternions (wxyz): [1, 0, 0, 0]
    quats = torch.zeros(num_roots, 4, device=device)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)

    sh_rest = torch.zeros(num_roots, num_sh - 1, 3, device=device)
    sh_rest.requires_grad_(True)

    return Gaussian(means=means, quats=quats, log_scales=log_scales,
                    opacities=opacities, sh_dc=sh_dc, sh_rest=sh_rest)


def train_roots_step(
    roots: Gaussian,
    gt_image: torch.Tensor,
    camera: dict,
    optimizer: torch.optim.Optimizer,
    ssim_weight: float = 0.2,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single training step for root Gaussians."""
    optimizer.zero_grad()
    rendered = render_gaussians(
        roots, viewmat=camera["viewmat"], K=camera["K"],
        width=camera["width"], height=camera["height"],
        background=background,
    )
    loss = rendering_loss(rendered, gt_image, ssim_weight=ssim_weight)
    loss.backward()
    optimizer.step()
    return loss.detach()


def fit_roots_lbfgs(
    roots: Gaussian,
    dataset,
    device: torch.device,
    max_iter: int = 100,
) -> Gaussian:
    """Fit root Gaussians using L-BFGS (quasi-Newton).

    Much faster convergence than Adam for the small parameter space
    of root Gaussians (~14 params). Uses L2 loss over all training views
    with per-iteration random backgrounds to prevent opacity cheating.
    """
    import logging
    logger = logging.getLogger(__name__)

    params = [roots.means, roots.quats, roots.log_scales,
              roots.opacities, roots.sh_dc, roots.sh_rest]

    optimizer = torch.optim.LBFGS(
        params, lr=0.1, max_iter=20, line_search_fn="strong_wolfe",
    )

    # Preload views (rgb + alpha separate, composite per-iteration with random bg)
    views = []
    for i in range(len(dataset)):
        gt_rgb, gt_alpha, camera = dataset[i]
        gt_rgb = gt_rgb.to(device)
        gt_alpha = gt_alpha.to(device)
        cam = {k: v.to(device) if isinstance(v, torch.Tensor) else v
               for k, v in camera.items()}
        views.append((gt_rgb, gt_alpha, cam))

    best_loss = float("inf")
    bg_gen = torch.Generator(device='cpu')
    for iteration in range(max_iter):
        # Deterministic random background per iteration — same across all
        # views and line search evaluations within this step, different
        # between iterations. Prevents opacity cheating on bright objects.
        bg_gen.manual_seed(iteration)
        iter_bg = torch.rand(3, generator=bg_gen).to(device)

        def closure():
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for gt_rgb, gt_alpha, cam in views:
                gt_image = gt_rgb * gt_alpha + (1.0 - gt_alpha) * iter_bg.view(1, 1, 3)
                rendered = render_gaussians(
                    roots, viewmat=cam["viewmat"], K=cam["K"],
                    width=cam["width"], height=cam["height"],
                    background=iter_bg,
                )
                total_loss = total_loss + F.mse_loss(rendered, gt_image)
            total_loss.backward()
            return total_loss

        loss = optimizer.step(closure)

        # Normalize quaternions
        with torch.no_grad():
            roots.quats.data = F.normalize(roots.quats.data, dim=-1)

        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
        if loss_val < best_loss:
            best_loss = loss_val

        if iteration % 10 == 0:
            logger.info(f"L-BFGS iter {iteration}: loss={loss_val:.6f}")

    logger.info(f"L-BFGS converged: loss={best_loss:.6f} in {max_iter} outer iterations")
    return roots


def fit_roots_silhouette(
    roots: Gaussian,
    dataset,
    device: torch.device,
    max_iter: int = 100,
) -> Gaussian:
    """Fit root Gaussians to match silhouette coverage using L-BFGS.

    Optimizes for alpha/opacity coverage rather than color accuracy.
    A thin cable and a large body contribute equally if both are uncovered.
    This ensures the root Gaussian spans the full spatial extent of the scene.

    Two phases:
    1. Silhouette phase: fit position, scale, opacity to match GT alpha masks
    2. Color phase: freeze geometry, fit SH to match colors
    """
    import logging
    import math
    logger = logging.getLogger(__name__)

    # Start large and opaque — the optimizer shrinks to match the silhouette.
    # Starting small gives vanishing alpha gradients (dead zone).
    with torch.no_grad():
        # Scale to cover entire camera orbit (object is somewhere inside)
        cam_positions = []
        for i in range(len(dataset)):
            _, _, camera = dataset[i]
            w2c = camera["viewmat"]
            R, t = w2c[:3, :3], w2c[:3, 3]
            cam_positions.append(-R.T @ t)
        cam_radius = torch.stack(cam_positions).norm(dim=-1).mean()
        large_scale = math.log(cam_radius.item() * 0.5)
        roots.log_scales.data.fill_(large_scale)
        # High opacity so rendered alpha starts near 1.0
        roots.opacities.data.fill_(5.0)  # sigmoid(5) ≈ 0.993
        logger.info(f"Silhouette init: scale={math.exp(large_scale):.2f}, opacity=0.993")

    # Phase 1: Silhouette — optimize position, scale, opacity only
    geom_params = [roots.means, roots.quats, roots.log_scales, roots.opacities]
    optimizer_geom = torch.optim.LBFGS(
        geom_params, lr=0.1, max_iter=20, line_search_fn="strong_wolfe",
    )

    # Preload views with alpha masks
    views = []
    for i in range(len(dataset)):
        gt_rgb, gt_alpha, camera = dataset[i]
        gt_rgb = gt_rgb.to(device)
        gt_alpha = gt_alpha.to(device)
        cam = {k: v.to(device) if isinstance(v, torch.Tensor) else v
               for k, v in camera.items()}
        views.append((gt_rgb, gt_alpha, cam))

    bg_gen = torch.Generator(device='cpu')

    logger.info("Phase 1: Silhouette fitting (position, scale, opacity)")
    for iteration in range(max_iter // 2):

        def closure_geom():
            optimizer_geom.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for gt_rgb, gt_alpha, cam in views:
                # Compare rendered alpha directly against GT alpha.
                # No color, no background — pure geometric coverage.
                _, render_alpha = render_gaussians(
                    roots, viewmat=cam["viewmat"], K=cam["K"],
                    width=cam["width"], height=cam["height"],
                    return_alpha=True,
                )
                gt_mask = gt_alpha.squeeze(-1)  # (H, W)
                total_loss = total_loss + F.mse_loss(render_alpha.squeeze(-1), gt_mask)
            total_loss.backward()
            return total_loss

        loss = optimizer_geom.step(closure_geom)
        with torch.no_grad():
            roots.quats.data = F.normalize(roots.quats.data, dim=-1)

        if iteration % 10 == 0:
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            logger.info(f"Silhouette iter {iteration}: loss={loss_val:.6f}")

    # Phase 2: Color — optimize SH with geometry frozen
    logger.info("Phase 2: Color fitting (SH coefficients)")
    color_params = [roots.sh_dc, roots.sh_rest]
    optimizer_color = torch.optim.LBFGS(
        color_params, lr=0.1, max_iter=20, line_search_fn="strong_wolfe",
    )

    for iteration in range(max_iter // 2):
        bg_gen.manual_seed(iteration + max_iter)
        iter_bg = torch.rand(3, generator=bg_gen).to(device)

        def closure_color():
            optimizer_color.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for gt_rgb, gt_alpha, cam in views:
                gt_image = gt_rgb * gt_alpha + (1.0 - gt_alpha) * iter_bg.view(1, 1, 3)
                rendered = render_gaussians(
                    roots, viewmat=cam["viewmat"], K=cam["K"],
                    width=cam["width"], height=cam["height"],
                    background=iter_bg,
                )
                total_loss = total_loss + F.mse_loss(rendered, gt_image)
            total_loss.backward()
            return total_loss

        loss = optimizer_color.step(closure_color)

        if iteration % 10 == 0:
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            logger.info(f"Color iter {iteration}: loss={loss_val:.6f}")

    logger.info("Silhouette fitting complete")
    return roots

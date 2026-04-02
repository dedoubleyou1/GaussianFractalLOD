"""Custom quality metrics based on 2D alpha moment comparison.

Compares the silhouette statistics (centroid, covariance, total mass) between
ground truth and rendered views. Works for single-Gaussian scenes (validating
the geometric fit) and multi-Gaussian scenes (validating overall coverage).

These metrics capture geometric quality independently of color:
- Centroid error: is the rendered content in the right place?
- Covariance error: does it have the right spread and shape?
- Mass error: does it cover the right amount of the image?
"""

import numpy as np
import torch


def compute_alpha_moments(alpha: np.ndarray) -> dict:
    """Compute 2D Gaussian moments from an alpha mask.

    Args:
        alpha: (H, W) array with values in [0, 1].

    Returns:
        Dict with 'centroid' (2,), 'covariance' (2, 2), 'mass' (scalar),
        or None values if the mask is empty.
    """
    H, W = alpha.shape
    total_mass = alpha.sum()

    if total_mass < 1e-8:
        return {"centroid": None, "covariance": None, "mass": 0.0}

    yy, xx = np.mgrid[:H, :W].astype(np.float64)

    # Alpha-weighted centroid
    mean_x = (alpha * xx).sum() / total_mass
    mean_y = (alpha * yy).sum() / total_mass
    centroid = np.array([mean_x, mean_y])

    # Alpha-weighted covariance
    dx = xx - mean_x
    dy = yy - mean_y
    cov_xx = (alpha * dx * dx).sum() / total_mass
    cov_xy = (alpha * dx * dy).sum() / total_mass
    cov_yy = (alpha * dy * dy).sum() / total_mass
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

    return {"centroid": centroid, "covariance": covariance, "mass": float(total_mass)}


def expand_covariance_for_mass(moments: dict) -> dict:
    """Expand covariance so a Gaussian at alpha=1.0 matches the total mass.

    If alpha = mass / (2π · sqrt(det(Σ))) > 1.0, the covariance is too
    small. Scale it by alpha² so mass matches at alpha=1.0.

    Returns a new dict with possibly expanded covariance and the computed alpha.
    """
    if moments["covariance"] is None or moments["mass"] < 1e-8:
        return {**moments, "alpha": 0.0}

    det = np.linalg.det(moments["covariance"])
    if det < 1e-10:
        return {**moments, "alpha": 0.5}

    alpha = moments["mass"] / (2 * np.pi * np.sqrt(det))
    cov = moments["covariance"]
    if alpha > 1.0:
        cov = cov * alpha
        alpha = 1.0

    return {**moments, "covariance": cov, "alpha": float(np.clip(alpha, 0.01, 0.99))}


def compare_alpha_moments(gt_alpha: np.ndarray, render_alpha: np.ndarray) -> dict:
    """Compare 2D alpha moments between ground truth and rendered view.

    Args:
        gt_alpha: (H, W) ground truth alpha mask.
        render_alpha: (H, W) rendered alpha mask.

    Returns:
        Dict with per-view metrics:
        - centroid_error: L2 distance between centroids (pixels)
        - covariance_error: Frobenius norm of covariance difference
        - mass_ratio: rendered_mass / gt_mass (1.0 = perfect)
        - mass_error: absolute difference in total mass
    """
    gt_moments = expand_covariance_for_mass(compute_alpha_moments(gt_alpha))
    render_moments = expand_covariance_for_mass(compute_alpha_moments(render_alpha))

    result = {}

    # Centroid comparison
    if gt_moments["centroid"] is not None and render_moments["centroid"] is not None:
        result["centroid_error"] = float(np.linalg.norm(
            gt_moments["centroid"] - render_moments["centroid"]
        ))
    else:
        result["centroid_error"] = float("inf")

    # Covariance comparison
    if gt_moments["covariance"] is not None and render_moments["covariance"] is not None:
        cov_diff = np.linalg.norm(
            gt_moments["covariance"] - render_moments["covariance"], ord="fro"
        )
        gt_cov_mag = np.linalg.norm(gt_moments["covariance"], ord="fro")
        result["covariance_error"] = float(cov_diff)
        # Relative covariance error: 0 = perfect match, 1 = 100% off
        result["covariance_error_rel"] = float(cov_diff / max(gt_cov_mag, 1e-10))
        # Eigenvalue comparison (scale match)
        gt_eigvals = np.linalg.eigvalsh(gt_moments["covariance"])
        render_eigvals = np.linalg.eigvalsh(render_moments["covariance"])
        result["scale_ratio"] = float(np.sqrt(render_eigvals.prod() / max(gt_eigvals.prod(), 1e-10)))
    else:
        result["covariance_error"] = float("inf")
        result["covariance_error_rel"] = float("inf")
        result["scale_ratio"] = 0.0

    # Mass comparison
    gt_mass = gt_moments["mass"]
    render_mass = render_moments["mass"]
    result["mass_ratio"] = float(render_mass / max(gt_mass, 1e-10))
    result["mass_error"] = float(abs(render_mass - gt_mass))
    result["gt_mass"] = gt_mass
    result["render_mass"] = render_mass

    return result


def compute_alpha_moments_torch(alpha: torch.Tensor, yy: torch.Tensor, xx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable 2D moments from a rendered alpha mask.

    Args:
        alpha: (H, W) tensor with values in [0, 1].
        yy: (H, W) precomputed y-coordinate grid.
        xx: (H, W) precomputed x-coordinate grid.

    Returns:
        (centroid, covariance): centroid is (2,), covariance is (2, 2).
    """
    total_mass = alpha.sum() + 1e-8

    mean_x = (alpha * xx).sum() / total_mass
    mean_y = (alpha * yy).sum() / total_mass
    centroid = torch.stack([mean_x, mean_y])

    dx = xx - mean_x
    dy = yy - mean_y
    cov_xx = (alpha * dx * dx).sum() / total_mass
    cov_xy = (alpha * dx * dy).sum() / total_mass
    cov_yy = (alpha * dy * dy).sum() / total_mass
    covariance = torch.stack([torch.stack([cov_xx, cov_xy]),
                              torch.stack([cov_xy, cov_yy])])

    return centroid, covariance


def moment_loss(
    render_alpha: torch.Tensor,
    gt_centroid: torch.Tensor,
    gt_cov: torch.Tensor,
    yy: torch.Tensor,
    xx: torch.Tensor,
    diagonal: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized centroid and covariance losses.

    Args:
        render_alpha: (H, W) rendered alpha, differentiable.
        gt_centroid: (2,) precomputed GT centroid.
        gt_cov: (2, 2) precomputed GT covariance.
        yy, xx: (H, W) coordinate grids (precomputed).
        diagonal: image diagonal in pixels (for centroid normalization).

    Returns:
        (centroid_loss, covariance_loss): both scalar, normalized to ~0-1 range.
    """
    render_centroid, render_cov = compute_alpha_moments_torch(render_alpha, yy, xx)

    # Centroid: L2 distance as fraction of image diagonal
    centroid_loss = (render_centroid - gt_centroid).pow(2).sum().sqrt() / diagonal

    # Covariance: relative Frobenius error
    gt_cov_norm = gt_cov.pow(2).sum().sqrt().clamp(min=1e-6)
    cov_loss = (render_cov - gt_cov).pow(2).sum().sqrt() / gt_cov_norm

    return centroid_loss, cov_loss


def deficit_sdf_loss(
    render_alpha: torch.Tensor,
    gt_alpha: torch.Tensor,
    coverage_radius: int = 2,
) -> torch.Tensor:
    """Pull rendered mass toward uncovered GT regions using a distance field.

    Computes a deficit mask (GT has coverage, no Gaussian nearby), then a
    signed distance field from those deficit regions. The loss gently pulls
    all rendered mass toward the nearest deficit, with 1/distance falloff.

    Args:
        render_alpha: (H, W) rendered alpha, differentiable.
        gt_alpha: (H, W) ground truth alpha.
        coverage_radius: dilation radius for "nearby" coverage check.

    Returns:
        Scalar loss.
    """
    from kornia.contrib import distance_transform
    import torch.nn.functional as F

    # Dilate rendered coverage: a pixel is "covered" if any Gaussian is within radius
    kernel_size = 2 * coverage_radius + 1
    render_dilated = F.max_pool2d(
        render_alpha.detach().unsqueeze(0).unsqueeze(0),
        kernel_size, stride=1, padding=coverage_radius,
    ).squeeze()

    # Deficit: GT has content but no Gaussian even nearby
    deficit_mask = ((gt_alpha > 0.01) & (render_dilated == 0)).float()

    # If no deficit, no loss
    if deficit_mask.sum() < 1:
        return torch.tensor(0.0, device=render_alpha.device)

    # Distance transform: 0 inside deficit, positive = distance to nearest deficit
    # kornia expects (B, C, H, W), 1=foreground
    deficit_sdf = distance_transform(
        deficit_mask.unsqueeze(0).unsqueeze(0)
    ).squeeze()

    # Pull field: 1/distance falloff — nearby Gaussians get nudged harder
    pull_field = 1.0 / (deficit_sdf + 1.0)

    # Loss: pull all rendered mass toward deficit, weighted by proximity
    return (render_alpha * pull_field).mean()


def evaluate_alpha_moments(
    tree,
    dataset,
    target_depth: int,
    device: torch.device,
    background: torch.Tensor | None = None,
) -> dict:
    """Evaluate alpha moment metrics across all test views.

    Args:
        tree: GaussianTree with trained levels.
        dataset: Test dataset.
        target_depth: Which level to render.
        device: Torch device.
        background: Background color.

    Returns:
        Dict with per-view and aggregate moment metrics.
    """
    from gaussianfractallod.render import render_gaussians

    if background is None:
        background = torch.ones(3, device=device)

    per_view = []

    with torch.no_grad():
        gaussians = tree.get_gaussians_at_depth(target_depth)

        for i in range(len(dataset)):
            gt_rgb, gt_alpha, camera = dataset[i]
            gt_alpha_np = gt_alpha.numpy().squeeze(-1)  # (H, W)

            cam = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in camera.items()}

            rendered, render_alpha = render_gaussians(
                gaussians, cam["viewmat"], cam["K"],
                cam["width"], cam["height"], background,
                return_alpha=True,
            )
            render_alpha_np = render_alpha.cpu().numpy().squeeze(-1) if render_alpha.dim() == 3 else render_alpha.cpu().numpy()

            metrics = compare_alpha_moments(gt_alpha_np, render_alpha_np)
            metrics["view"] = i
            per_view.append(metrics)

    # Aggregate
    centroid_errors = [m["centroid_error"] for m in per_view if m["centroid_error"] != float("inf")]
    cov_errors = [m["covariance_error"] for m in per_view if m["covariance_error"] != float("inf")]
    cov_errors_rel = [m["covariance_error_rel"] for m in per_view if m["covariance_error_rel"] != float("inf")]
    mass_ratios = [m["mass_ratio"] for m in per_view]

    summary = {
        "per_view": per_view,
        "num_gaussians": gaussians.num_gaussians,
        "centroid_error_mean": float(np.mean(centroid_errors)) if centroid_errors else float("inf"),
        "centroid_error_max": float(np.max(centroid_errors)) if centroid_errors else float("inf"),
        "covariance_error_mean": float(np.mean(cov_errors)) if cov_errors else float("inf"),
        "covariance_error_rel_mean": float(np.mean(cov_errors_rel)) if cov_errors_rel else float("inf"),
        "mass_ratio_mean": float(np.mean(mass_ratios)),
        "mass_ratio_std": float(np.std(mass_ratios)),
    }

    return summary

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
    gt_moments = compute_alpha_moments(gt_alpha)
    render_moments = compute_alpha_moments(render_alpha)

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

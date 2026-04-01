"""Fit a single 3D Gaussian to a set of training views using closed-form geometry.

No optimization — derives position, covariance, opacity, and color directly
from image moments (alpha-weighted statistics of each training view).

The key insight: a Gaussian in splatting represents a statistical approximation
of both the occluded space (shape/opacity) and the light emanating from that
region (color). Each training view provides a 2D projection of this 3D
Gaussian. By fitting 2D Gaussians to each view's alpha mask and combining
them via multi-view geometry, we reconstruct the 3D Gaussian analytically.
"""

import numpy as np
import torch
from scipy.spatial import SphericalVoronoi
from scipy.spatial.transform import Rotation

from gaussianfractallod.gaussian import Gaussian


def fit_gaussian_to_views(dataset, device: torch.device, sh_degree: int = 0) -> Gaussian:
    """Fit a single 3D Gaussian from training view alpha masks and colors.

    Args:
        dataset: NerfSyntheticDataset with (rgb, alpha, camera) per view.
        device: torch device for the output Gaussian.
        sh_degree: SH degree for the output (rest bands initialized to zero).

    Returns:
        A single Gaussian with all parameters derived from data.
    """
    N = len(dataset)

    # ----------------------------------------------------------------
    # Step 1: Per-view 2D moments
    #
    # For each view, compute the centroid, covariance, total mass, and
    # average color of the silhouette. These are sufficient statistics
    # for a 2D Gaussian — no iteration needed.
    # ----------------------------------------------------------------
    view_stats = []

    for i in range(N):
        gt_rgb, gt_alpha, camera = dataset[i]
        alpha = gt_alpha.numpy().squeeze(-1)  # (H, W)
        rgb = gt_rgb.numpy()  # (H, W, 3)
        H, W = alpha.shape

        total_mass = alpha.sum()
        if total_mass < 1e-8:
            continue

        yy, xx = np.mgrid[:H, :W].astype(np.float64)

        # Alpha-weighted centroid
        mean_x = (alpha * xx).sum() / total_mass
        mean_y = (alpha * yy).sum() / total_mass
        mean_2d = np.array([mean_x, mean_y])

        # Alpha-weighted covariance
        dx = xx - mean_x
        dy = yy - mean_y
        cov_2d = np.array([
            [(alpha * dx * dx).sum() / total_mass, (alpha * dx * dy).sum() / total_mass],
            [(alpha * dx * dy).sum() / total_mass, (alpha * dy * dy).sum() / total_mass],
        ])

        # Per-view average color
        avg_color = np.array([(alpha * rgb[:, :, c]).sum() / total_mass for c in range(3)])

        # Camera parameters
        w2c = camera["viewmat"].numpy()
        K = camera["K"].numpy()
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        cam_pos = -R.T @ t  # camera position in world space

        view_stats.append({
            "mean_2d": mean_2d,
            "cov_2d": cov_2d,
            "total_mass": float(total_mass),
            "avg_color": avg_color,
            "cam_pos": cam_pos,
            "R": R,
            "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
        })

    N_valid = len(view_stats)
    cam_positions = np.array([vs["cam_pos"] for vs in view_stats])

    # ----------------------------------------------------------------
    # Step 2: Per-view opacity + covariance expansion
    #
    # Opacity: α = mass / (2π · sqrt(det(Σ_2d))).
    # If α > 1.0, the Gaussian is too small to contain the GT mass
    # even at full opacity. Expand the covariance analytically:
    # Σ_adjusted = α² · Σ (so mass matches at α = 1.0).
    # ----------------------------------------------------------------
    for vs in view_stats:
        det_2d = np.linalg.det(vs["cov_2d"])
        if det_2d > 1e-10:
            alpha_est = vs["total_mass"] / (2 * np.pi * np.sqrt(det_2d))
            if alpha_est > 1.0:
                vs["cov_2d"] = vs["cov_2d"] * alpha_est ** 2
                vs["alpha"] = 1.0
            else:
                vs["alpha"] = float(np.clip(alpha_est, 0.01, 0.99))
        else:
            vs["alpha"] = 0.5

    # ----------------------------------------------------------------
    # Step 3: Back-project 2D centers to rays
    # ----------------------------------------------------------------
    rays_origin = []
    rays_dir = []
    for vs in view_stats:
        origin = vs["cam_pos"]
        mx, my = vs["mean_2d"]
        dir_cam = np.array([
            (mx - vs["cx"]) / vs["fx"],
            (my - vs["cy"]) / vs["fy"],
            1.0,
        ])
        dir_world = vs["R"].T @ dir_cam
        dir_world = dir_world / np.linalg.norm(dir_world)
        rays_origin.append(origin)
        rays_dir.append(dir_world)

    rays_origin = np.array(rays_origin)
    rays_dir = np.array(rays_dir)

    # ----------------------------------------------------------------
    # Step 4: Least-squares 3D position from rays
    #
    # Find the point closest to all rays.
    # (Σ (I - d·d^T)) @ p = Σ (I - d·d^T) @ origin
    # ----------------------------------------------------------------
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(N_valid):
        d = rays_dir[i]
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ rays_origin[i]

    center_3d = np.linalg.solve(A, b)

    # ----------------------------------------------------------------
    # Step 5: Camera-to-Gaussian directions + Voronoi weights
    #
    # Weight each view by its solid angle coverage on the sphere.
    # Prevents bias from clustered cameras in covariance, opacity, color.
    # ----------------------------------------------------------------
    directions = cam_positions - center_3d
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    try:
        sv = SphericalVoronoi(directions, radius=1.0, center=np.zeros(3))
        sv.sort_vertices_of_regions()
        weights = sv.calculate_areas()
    except Exception:
        weights = np.ones(N_valid)

    weights = weights / weights.sum()

    # ----------------------------------------------------------------
    # Step 6: Refine 3D position with Voronoi-weighted rays
    # ----------------------------------------------------------------
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(N_valid):
        d = rays_dir[i]
        P = np.eye(3) - np.outer(d, d)
        A += weights[i] * P
        b += weights[i] * P @ rays_origin[i]

    center_3d = np.linalg.solve(A, b)

    # ----------------------------------------------------------------
    # Step 7: 3D covariance from Voronoi-weighted projected covariances
    #
    # Each view: Σ_2d = J @ R @ Σ_3d @ R^T @ J^T (linear in Σ_3d).
    # The per-view covariances may have been expanded in Step 2.
    # ----------------------------------------------------------------
    rows_A = []
    rows_b = []
    row_weights = []

    idx_map = {(0, 0): 0, (0, 1): 1, (0, 2): 2,
               (1, 0): 1, (1, 1): 3, (1, 2): 4,
               (2, 0): 2, (2, 1): 4, (2, 2): 5}

    for i, vs in enumerate(view_stats):
        R = vs["R"]
        t = -R @ vs["cam_pos"]
        cam_point = R @ center_3d + t
        depth = cam_point[2]
        if abs(depth) < 1e-6:
            continue

        fx, fy = vs["fx"], vs["fy"]
        x_cam, y_cam = cam_point[0], cam_point[1]
        J = np.array([
            [fx / depth, 0, -fx * x_cam / depth**2],
            [0, fy / depth, -fy * y_cam / depth**2],
        ])
        M = J @ R  # (2, 3)

        for a, b_idx, target in [(0, 0, vs["cov_2d"][0, 0]),
                                  (0, 1, vs["cov_2d"][0, 1]),
                                  (1, 1, vs["cov_2d"][1, 1])]:
            row = np.zeros(6)
            for ii in range(3):
                for jj in range(3):
                    row[idx_map[(ii, jj)]] += M[a, ii] * M[b_idx, jj]
            rows_A.append(row)
            rows_b.append(target)
            row_weights.append(weights[i])

    A_mat = np.array(rows_A)
    b_vec = np.array(rows_b)
    W_diag = np.sqrt(np.array(row_weights))
    sigma_6, _, _, _ = np.linalg.lstsq(A_mat * W_diag[:, None], b_vec * W_diag, rcond=None)

    cov_3d = np.array([
        [sigma_6[0], sigma_6[1], sigma_6[2]],
        [sigma_6[1], sigma_6[3], sigma_6[4]],
        [sigma_6[2], sigma_6[4], sigma_6[5]],
    ])

    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(cov_3d)
    eigvals = np.maximum(eigvals, 1e-7)

    # ----------------------------------------------------------------
    # Step 8: Decompose covariance → quaternion + scales
    # ----------------------------------------------------------------
    scales = np.sqrt(eigvals).astype(np.float32)
    log_scales = np.log(scales)

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1
    quat_scipy = Rotation.from_matrix(eigvecs)
    quat_xyzw = quat_scipy.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0],
                          quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

    # ----------------------------------------------------------------
    # Step 9: Voronoi-weighted opacity from per-view estimates
    # ----------------------------------------------------------------
    alpha_arr = np.array([vs["alpha"] for vs in view_stats])
    opacity = float(np.clip(np.sum(weights * alpha_arr), 0.01, 0.99))
    logit_opacity = np.log(opacity / (1 - opacity))

    # ----------------------------------------------------------------
    # Step 10: Voronoi-weighted color from per-view averages
    #
    # Each view samples the object's color from one direction.
    # Voronoi weights ensure uniform angular coverage so the DC color
    # isn't biased toward over-represented viewing directions.
    # ----------------------------------------------------------------
    colors = np.array([vs["avg_color"] for vs in view_stats])
    avg_color = np.sum(weights[:, None] * colors, axis=0)

    C0 = 0.28209479177387814
    sh_dc = ((avg_color - 0.5) / C0).astype(np.float32).reshape(1, 1, 3)

    # ----------------------------------------------------------------
    # Step 11: Assemble Gaussian
    # ----------------------------------------------------------------
    num_sh_rest = (sh_degree + 1) ** 2 - 1

    return Gaussian(
        means=torch.tensor(center_3d.astype(np.float32)).unsqueeze(0).to(device).requires_grad_(True),
        quats=torch.tensor(quat_wxyz).unsqueeze(0).to(device).requires_grad_(True),
        log_scales=torch.tensor(log_scales).unsqueeze(0).to(device).requires_grad_(True),
        opacities=torch.tensor([[logit_opacity]], dtype=torch.float32).to(device).requires_grad_(True),
        sh_dc=torch.tensor(sh_dc).to(device).requires_grad_(True),
        sh_rest=torch.zeros(1, num_sh_rest, 3, device=device).requires_grad_(True),
    )

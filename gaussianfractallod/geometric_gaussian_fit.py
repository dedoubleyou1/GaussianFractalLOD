"""Fit a single 3D Gaussian to a set of training views using closed-form geometry.

No optimization — derives position, covariance, opacity, and color directly
from image moments (alpha-weighted statistics of each training view).

The key insight: a Gaussian in splatting represents a statistical approximation
of both the occluded space (shape/opacity) and the light emanating from that
region (color). Each training view provides a 2D projection of this 3D
Gaussian. By fitting 2D Gaussians to each view's alpha mask and combining
them via multi-view geometry, we reconstruct the 3D Gaussian analytically.

This avoids the failure modes of optimization-based approaches:
- No opacity death spiral (optimizer can't drive opacity to zero)
- No local minima (closed-form solution)
- No bias toward dense regions (Voronoi weighting corrects for view clustering)
- Guaranteed coverage of the full scene extent (2D moments capture all alpha pixels)
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

    Returns:
        A single Gaussian with position, quaternion, scales, opacity,
        and SH DC color — all derived from data, no optimization.
    """
    N = len(dataset)

    # ----------------------------------------------------------------
    # Step 1: Per-view 2D Gaussian fit from alpha-weighted moments
    #
    # For each view, compute the mean, covariance, total mass, and
    # average color of the silhouette. These are the sufficient
    # statistics for a 2D Gaussian fit — no iteration needed.
    # ----------------------------------------------------------------
    view_stats = []
    cam_positions = []
    cam_rotations = []  # 3x3 rotation matrices (world-to-camera)
    cam_intrinsics = []  # (fx, fy, cx, cy)

    for i in range(N):
        gt_rgb, gt_alpha, camera = dataset[i]
        alpha = gt_alpha.numpy().squeeze(-1)  # (H, W)
        rgb = gt_rgb.numpy()  # (H, W, 3)
        H, W = alpha.shape

        # Pixel coordinate grid
        yy, xx = np.mgrid[:H, :W].astype(np.float64)

        total_mass = alpha.sum()
        if total_mass < 1e-8:
            continue  # skip empty views

        # Alpha-weighted 2D centroid
        mean_x = (alpha * xx).sum() / total_mass
        mean_y = (alpha * yy).sum() / total_mass
        mean_2d = np.array([mean_x, mean_y])

        # Alpha-weighted 2D covariance
        dx = xx - mean_x
        dy = yy - mean_y
        cov_xx = (alpha * dx * dx).sum() / total_mass
        cov_xy = (alpha * dx * dy).sum() / total_mass
        cov_yy = (alpha * dy * dy).sum() / total_mass
        cov_2d = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

        # Per-view average color (alpha-weighted)
        avg_color = np.zeros(3)
        for c in range(3):
            avg_color[c] = (alpha * rgb[:, :, c]).sum() / total_mass

        # Camera parameters
        w2c = camera["viewmat"].numpy()
        K = camera["K"].numpy()
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        # Camera position in world space: -R^T @ t
        # Valid for any rigid w2c regardless of axis convention.
        cam_pos = -R.T @ t

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        view_stats.append({
            "mean_2d": mean_2d,
            "cov_2d": cov_2d,
            "total_mass": total_mass,
            "avg_color": avg_color,
            "cam_pos": cam_pos,
            "R": R,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        })
        cam_positions.append(cam_pos)

    N_valid = len(view_stats)
    cam_positions = np.array(cam_positions)  # (N_valid, 3)

    # ----------------------------------------------------------------
    # Step 2: Initial 3D position from least-squares ray intersection
    #
    # Each 2D centroid back-projects to a ray in 3D. The 3D Gaussian
    # center is the point closest to all rays. This is a standard
    # closest-point-to-multiple-lines problem, solved as a 3x3 system.
    # ----------------------------------------------------------------
    # Build rays: for each view, unproject the 2D centroid
    rays_origin = []
    rays_dir = []
    for vs in view_stats:
        origin = vs["cam_pos"]
        # Unproject 2D centroid to camera-space direction
        mx, my = vs["mean_2d"]
        dir_cam = np.array([
            (mx - vs["cx"]) / vs["fx"],
            (my - vs["cy"]) / vs["fy"],
            1.0,
        ])
        # Transform to world space: R^T @ dir_cam (undo w2c rotation)
        dir_world = vs["R"].T @ dir_cam
        dir_world = dir_world / np.linalg.norm(dir_world)
        rays_origin.append(origin)
        rays_dir.append(dir_world)

    rays_origin = np.array(rays_origin)
    rays_dir = np.array(rays_dir)

    # Closest point to all rays: minimize Σ ||p - (o_i + t_i * d_i)||²
    # Analytically: (Σ (I - d_i d_i^T)) @ p = Σ (I - d_i d_i^T) @ o_i
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(N_valid):
        d = rays_dir[i]
        ddT = np.outer(d, d)
        P = np.eye(3) - ddT  # projection onto plane perpendicular to ray
        A += P
        b += P @ rays_origin[i]

    center_3d = np.linalg.solve(A, b)

    # ----------------------------------------------------------------
    # Step 3: Voronoi weights from camera-to-Gaussian directions
    #
    # Views aren't uniformly distributed on the sphere. To avoid bias
    # in covariance, opacity, and color estimates, we weight each view
    # by the solid angle it uniquely covers. Views in sparse angular
    # regions count more; clustered views count less.
    # ----------------------------------------------------------------
    # Direction from each camera to the Gaussian center
    directions = cam_positions - center_3d
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    try:
        sv = SphericalVoronoi(directions, radius=1.0, center=np.zeros(3))
        sv.sort_vertices_of_regions()
        weights = sv.calculate_areas()
    except Exception:
        # Fallback: uniform weights if Voronoi fails (e.g., degenerate geometry)
        weights = np.ones(N_valid)

    weights = weights / weights.sum()

    # ----------------------------------------------------------------
    # Step 4: Refine 3D position with Voronoi-weighted ray intersection
    #
    # Re-solve using angular weights so the center isn't biased toward
    # directions with many cameras.
    # ----------------------------------------------------------------
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(N_valid):
        d = rays_dir[i]
        ddT = np.outer(d, d)
        P = np.eye(3) - ddT
        A += weights[i] * P
        b += weights[i] * P @ rays_origin[i]

    center_3d = np.linalg.solve(A, b)

    # ----------------------------------------------------------------
    # Step 5: 3D covariance from Voronoi-weighted projected covariances
    #
    # Each view constrains the 3D covariance through the projection
    # equation: Σ_2d = J @ R @ Σ_3d @ R^T @ J^T, where J is the
    # projection Jacobian (known per view once 3D center is known)
    # and R is the view rotation.
    #
    # This is linear in the 6 unique entries of Σ_3d. We stack all
    # views into an overdetermined system and solve weighted least-squares.
    # ----------------------------------------------------------------
    # Build the linear system: for each view, express the 3 unique
    # entries of Σ_2d as a linear function of the 6 unique entries of Σ_3d.
    rows_A = []
    rows_b = []
    row_weights = []

    for i, vs in enumerate(view_stats):
        R = vs["R"]  # 3x3 world-to-camera rotation
        # Depth of center in this camera
        cam_center = R @ center_3d + np.array([
            vs["R"] @ center_3d + np.array([0, 0, 0])
        ]).flatten()
        # Actually: cam_point = R @ world_point + t
        t = -R @ vs["cam_pos"]  # reconstruct t from cam_pos
        cam_point = R @ center_3d + t
        depth = cam_point[2]
        if abs(depth) < 1e-6:
            continue

        # Projection Jacobian: J = [[fx/z, 0, -fx*x/z²], [0, fy/z, -fy*y/z²]]
        fx, fy = vs["fx"], vs["fy"]
        x_cam, y_cam = cam_point[0], cam_point[1]
        J = np.array([
            [fx / depth, 0, -fx * x_cam / depth**2],
            [0, fy / depth, -fy * y_cam / depth**2],
        ])

        # M = J @ R maps 3D world covariance to 2D: Σ_2d = M @ Σ_3d @ M^T
        M = J @ R  # (2, 3)

        # The 3 unique entries of Σ_2d (xx, xy, yy) as linear functions of
        # the 6 unique entries of Σ_3d (xx, xy, xz, yy, yz, zz)
        # Σ_2d[a,b] = Σ_ij M[a,i] * M[b,j] * Σ_3d[i,j]
        # With symmetry: Σ_3d has 6 unique entries, indexed as
        # [0,0]=0, [0,1]=1, [0,2]=2, [1,1]=3, [1,2]=4, [2,2]=5
        for a, b, target in [(0, 0, vs["cov_2d"][0, 0]),
                              (0, 1, vs["cov_2d"][0, 1]),
                              (1, 1, vs["cov_2d"][1, 1])]:
            row = np.zeros(6)
            # Map (i,j) symmetric pair to index in 6-vector
            idx_map = {(0, 0): 0, (0, 1): 1, (0, 2): 2,
                       (1, 0): 1, (1, 1): 3, (1, 2): 4,
                       (2, 0): 2, (2, 1): 4, (2, 2): 5}
            for ii in range(3):
                for jj in range(3):
                    k = idx_map[(ii, jj)]
                    # Account for off-diagonal entries being counted twice
                    coeff = M[a, ii] * M[b, jj]
                    if ii != jj:
                        row[k] += coeff  # both (i,j) and (j,i) map to same k
                    else:
                        row[k] += coeff
            rows_A.append(row)
            rows_b.append(target)
            row_weights.append(weights[i])

    A_mat = np.array(rows_A)  # (3N, 6)
    b_vec = np.array(rows_b)  # (3N,)
    W_diag = np.sqrt(np.array(row_weights))  # sqrt for weighted least-squares

    # Weighted least-squares: minimize Σ w_i * ||A_i @ σ - b_i||²
    A_weighted = A_mat * W_diag[:, None]
    b_weighted = b_vec * W_diag
    sigma_6, _, _, _ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)

    # Reconstruct 3x3 covariance matrix
    cov_3d = np.array([
        [sigma_6[0], sigma_6[1], sigma_6[2]],
        [sigma_6[1], sigma_6[3], sigma_6[4]],
        [sigma_6[2], sigma_6[4], sigma_6[5]],
    ])

    # Ensure positive semi-definite (clamp negative eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(cov_3d)
    eigvals = np.maximum(eigvals, 1e-7)
    cov_3d = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # ----------------------------------------------------------------
    # Step 6: Decompose covariance → quaternion + scales
    # ----------------------------------------------------------------
    scales = np.sqrt(eigvals).astype(np.float32)
    log_scales = np.log(scales)

    # Ensure right-handed rotation (det = +1)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1
    quat_scipy = Rotation.from_matrix(eigvecs)
    quat_xyzw = quat_scipy.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0],
                          quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

    # ----------------------------------------------------------------
    # Step 7: Opacity from alpha mass conservation
    #
    # The total alpha mass in each view equals the integral of the
    # projected Gaussian: m = α · 2π · sqrt(det(Σ_2d)).
    # So per-view opacity: α_i = m_i / (2π · sqrt(det(Σ_2d_i)))
    #
    # Combined with Voronoi weights for angular uniformity.
    # ----------------------------------------------------------------
    opacity_estimates = []
    for vs in view_stats:
        det_2d = np.linalg.det(vs["cov_2d"])
        if det_2d > 1e-10:
            alpha_est = vs["total_mass"] / (2 * np.pi * np.sqrt(det_2d))
            opacity_estimates.append(np.clip(alpha_est, 0.01, 0.99))
        else:
            opacity_estimates.append(0.5)

    opacity_estimates = np.array(opacity_estimates)
    opacity = np.sum(weights * opacity_estimates)
    opacity = np.clip(opacity, 0.01, 0.99)
    logit_opacity = np.log(opacity / (1 - opacity))

    # ----------------------------------------------------------------
    # Step 8: Color from Voronoi-weighted angular average
    #
    # Each view gives one directional sample of the object's color.
    # Voronoi weighting ensures uniform angular coverage — views in
    # sparse directions count more, preventing bias from clustered cameras.
    # This is the best DC color estimate without SH (the angular mean).
    # ----------------------------------------------------------------
    colors = np.array([vs["avg_color"] for vs in view_stats])
    avg_color = np.sum(weights[:, None] * colors, axis=0)

    C0 = 0.28209479177387814  # SH band 0 normalization
    sh_dc = ((avg_color - 0.5) / C0).astype(np.float32).reshape(1, 1, 3)

    # ----------------------------------------------------------------
    # Assemble the Gaussian
    # ----------------------------------------------------------------
    return Gaussian(
        means=torch.tensor(center_3d.astype(np.float32)).unsqueeze(0).to(device).requires_grad_(True),
        quats=torch.tensor(quat_wxyz).unsqueeze(0).to(device).requires_grad_(True),
        log_scales=torch.tensor(log_scales).unsqueeze(0).to(device).requires_grad_(True),
        opacities=torch.tensor([[logit_opacity]], dtype=torch.float32).to(device).requires_grad_(True),
        sh_dc=torch.tensor(sh_dc).to(device).requires_grad_(True),
        sh_rest=torch.zeros(1, (sh_degree + 1) ** 2 - 1, 3, device=device).requires_grad_(True),
    )

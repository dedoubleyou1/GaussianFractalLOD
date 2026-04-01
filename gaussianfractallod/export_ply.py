"""Export Gaussians to standard 3DGS PLY format for viewing in web viewers."""

import struct
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian


def _rotate_sh(sh_rest, rotation):
    """Rotate SH coefficients (bands 1-3) using Wigner D-matrices.

    3DGS uses a non-standard sign convention for real SH (Condon-Shortley
    phase: coefficients for odd |m| are negated vs standard math convention).
    We correct for this before/after applying the Wigner D-matrix.

    Args:
        sh_rest: (N, K-1, 3) SH coefficients for bands 1+.
        rotation: scipy.spatial.transform.Rotation object.

    Returns:
        (N, K-1, 3) rotated SH coefficients.
    """
    from sphecerix import tesseral_wigner_D

    N, K_minus_1, C = sh_rest.shape
    if K_minus_1 == 0:
        return sh_rest.copy()

    result = sh_rest.copy()

    idx = 0
    for l in range(1, 4):
        n_coeffs = 2 * l + 1
        if idx + n_coeffs > K_minus_1:
            break

        # Sign correction for 3DGS convention: (-1)^|m| per coefficient
        # m ranges from -l to +l
        phase = np.array([(-1) ** abs(m) for m in range(-l, l + 1)], dtype=np.float64)

        D = tesseral_wigner_D(l, rotation)  # (2l+1, 2l+1)

        # D' = phase @ D @ phase (phase is diagonal, so element-wise)
        D_corrected = (phase[:, None] * D * phase[None, :])

        for c in range(C):
            band = sh_rest[:, idx:idx + n_coeffs, c]  # (N, 2l+1)
            result[:, idx:idx + n_coeffs, c] = band @ D_corrected.T
        idx += n_coeffs

    return result


def _zup_to_yup(means, quats, log_scales, sh_dc, sh_rest):
    """Convert from Z-up (NeRF synthetic) to Y-up coordinate system.

    Our scene has Z=up, Y=forward. SuperSplat/PlayCanvas wants Y=up, Z=forward.
    Transform: (x, y, z) → (-x, -z, -y) with matching quaternion and SH rotation.
    """
    from scipy.spatial.transform import Rotation

    # Rotation matrix for coordinate transform: (-x, -z, -y)
    M = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64)
    rot = Rotation.from_matrix(M)

    # Positions
    means_yup = means.copy()
    means_yup[:, 0] = -means[:, 0]
    means_yup[:, 1] = -means[:, 2]
    means_yup[:, 2] = -means[:, 1]

    # Log scales: swap Y↔Z
    ls_yup = log_scales.copy()
    ls_yup[:, 1] = log_scales[:, 2]
    ls_yup[:, 2] = log_scales[:, 1]

    # Quaternions: scipy rotation, consistent with M
    quats_xyzw = np.column_stack([quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]])
    rotations = Rotation.from_quat(quats_xyzw)
    rotated = rot * rotations
    result_xyzw = rotated.as_quat()
    quats_yup = np.column_stack([
        result_xyzw[:, 3], result_xyzw[:, 0],
        result_xyzw[:, 1], result_xyzw[:, 2],
    ]).astype(np.float32)

    # SH: DC is view-independent, no change. Rotate rest bands with same M.
    sh_dc_yup = sh_dc.copy()
    sh_rest_yup = _rotate_sh(sh_rest, rot) if sh_rest.shape[1] > 0 else sh_rest.copy()

    return means_yup, quats_yup, ls_yup, sh_dc_yup, sh_rest_yup


def export_ply(gaussians: Gaussian, path: str, sh_degree: int = 0,
               y_up: bool = False) -> None:
    """Export Gaussians to PLY format compatible with 3DGS viewers.

    Args:
        y_up: If True, convert from Z-up (NeRF synthetic) to Y-up coordinates.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    N = gaussians.num_gaussians
    num_sh = (sh_degree + 1) ** 2
    num_rest = max(0, num_sh - 1) * 3

    with torch.no_grad():
        means = gaussians.means.cpu().numpy()
        opacities = gaussians.opacities.cpu().numpy()
        quats = F.normalize(gaussians.quats, dim=-1).cpu().numpy()
        log_scales = gaussians.log_scales.cpu().numpy()
        sh_dc_np = gaussians.sh_dc.cpu().numpy().reshape(N, 1, 3)
        sh_rest_np = gaussians.sh_rest.cpu().numpy()  # (N, K-1, 3)

        if y_up:
            means, quats, log_scales, sh_dc_np, sh_rest_np = _zup_to_yup(
                means, quats, log_scales, sh_dc_np, sh_rest_np
            )

        # PLY f_dc: (N, 3)
        sh_dc = sh_dc_np.reshape(N, 3)
        # PLY f_rest: channel-major (N, 3*(K-1))
        if num_sh > 1:
            sh_rest_channelmajor = np.transpose(sh_rest_np, (0, 2, 1)).reshape(N, -1)
        else:
            sh_rest_channelmajor = None

    header = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property float nx\nproperty float ny\nproperty float nz\n"
    header += "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    for i in range(num_rest):
        header += f"property float f_rest_{i}\n"
    header += "property float opacity\n"
    header += "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
    header += "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
    header += "end_header\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(N):
            f.write(struct.pack("<fff", *means[i]))
            f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
            f.write(struct.pack("<fff", *sh_dc[i]))
            if num_rest > 0:
                f.write(struct.pack(f"<{num_rest}f", *sh_rest_channelmajor[i]))
            f.write(struct.pack("<f", opacities[i, 0]))
            f.write(struct.pack("<fff", *log_scales[i]))
            f.write(struct.pack("<ffff", *quats[i]))

    print(f"Exported {N:,} Gaussians to {path} ({Path(path).stat().st_size / 1024 / 1024:.1f} MB)")

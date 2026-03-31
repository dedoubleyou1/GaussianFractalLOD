"""Export Gaussians to standard 3DGS PLY format for viewing in web viewers."""

import struct
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian


def _zup_to_yup(means, quats, log_scales):
    """Convert from Z-up (NeRF synthetic) to Y-up coordinate system.

    Transform: (-x, z, y) — 90° rotation mapping +Z→+Y, with X negated
    to preserve right-handedness. Quaternion derived via scipy.
    q_rot = (0, 0, √2/2, √2/2) in wxyz.
    """
    import math

    # Positions: (x,y,z) → (-x, z, y)
    means_yup = means.copy()
    means_yup[:, 0] = -means[:, 0]
    means_yup[:, 1] = means[:, 2]
    means_yup[:, 2] = means[:, 1]

    # Log scales: X stays, swap Y↔Z
    ls_yup = log_scales.copy()
    ls_yup[:, 1] = log_scales[:, 2]
    ls_yup[:, 2] = log_scales[:, 1]

    # Quaternion: q_rot = (0, 0, √2/2, √2/2) in wxyz
    # Hamilton product: q_new = q_rot * q_original
    s2 = math.sqrt(2) / 2
    qw, qx, qy, qz = 0.0, 0.0, s2, s2
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    quats_yup = quats.copy()
    quats_yup[:, 0] = qw*w - qx*x - qy*y - qz*z
    quats_yup[:, 1] = qw*x + qx*w + qy*z - qz*y
    quats_yup[:, 2] = qw*y - qx*z + qy*w + qz*x
    quats_yup[:, 3] = qw*z + qx*y - qy*x + qz*w

    return means_yup, quats_yup, ls_yup


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

        if y_up:
            means, quats, log_scales = _zup_to_yup(means, quats, log_scales)

        # SH: dc is (N, 1, 3), rest is (N, K-1, 3)
        # PLY f_dc: (N, 3) — just squeeze the middle dim
        # PLY f_rest: channel-major (N, 3*(K-1)) — transpose then flatten
        sh_dc = gaussians.sh_dc.cpu().numpy().reshape(N, 3)
        if num_sh > 1:
            sh_rest = gaussians.sh_rest.cpu().numpy()  # (N, K-1, 3)
            sh_rest_channelmajor = np.transpose(sh_rest, (0, 2, 1)).reshape(N, -1)
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

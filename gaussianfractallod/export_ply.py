"""Export Gaussians to standard 3DGS PLY format for viewing in web viewers."""

import struct
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian


def export_ply(gaussians: Gaussian, path: str, sh_degree: int = 0) -> None:
    """Export Gaussians to PLY format compatible with 3DGS viewers.

    Quaternions and log-scales are stored directly — no eigendecomposition needed.
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

        # SH coefficients: internal layout is (N, num_sh*3) coefficient-major
        # [SH0_R, SH0_G, SH0_B, SH1_R, SH1_G, SH1_B, ...]
        # PLY format expects channel-major for f_rest:
        # [SH1_R, SH2_R, ..., SH15_R, SH1_G, ..., SH15_G, SH1_B, ..., SH15_B]
        raw_sh = gaussians.sh_coeffs.cpu().numpy()
        sh_dc = raw_sh[:, :3]
        if num_sh > 1:
            sh_rest_interleaved = raw_sh[:, 3:].reshape(N, num_sh - 1, 3)
            sh_rest_channelmajor = np.transpose(sh_rest_interleaved, (0, 2, 1)).reshape(N, -1)
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

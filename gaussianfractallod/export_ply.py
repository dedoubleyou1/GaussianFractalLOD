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

    with torch.no_grad():
        means = gaussians.means.cpu().numpy()
        opacities = gaussians.opacities.cpu().numpy()
        sh_coeffs = gaussians.sh_coeffs.cpu().numpy()
        quats = F.normalize(gaussians.quats, dim=-1).cpu().numpy()
        log_scales = gaussians.log_scales.cpu().numpy()

    N = means.shape[0]
    num_sh = (sh_degree + 1) ** 2
    num_rest = max(0, num_sh - 1) * 3

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
            f.write(struct.pack("<fff", *sh_coeffs[i, :3]))
            if num_rest > 0:
                rest = sh_coeffs[i, 3:3 + num_rest]
                if len(rest) < num_rest:
                    rest = np.concatenate([rest, np.zeros(num_rest - len(rest))])
                f.write(struct.pack(f"<{num_rest}f", *rest))
            f.write(struct.pack("<f", opacities[i, 0]))
            f.write(struct.pack("<fff", *log_scales[i]))
            f.write(struct.pack("<ffff", *quats[i]))

    print(f"Exported {N:,} Gaussians to {path} ({Path(path).stat().st_size / 1024 / 1024:.1f} MB)")

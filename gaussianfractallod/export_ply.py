"""Export Gaussians to standard 3DGS PLY format for viewing in web viewers."""

import struct
import torch
import numpy as np
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.render import _covariance_to_quat_scale


def export_ply(gaussians: Gaussian, path: str, sh_degree: int = 0) -> None:
    """Export Gaussians to PLY format compatible with 3DGS viewers.

    Decomposes Cholesky covariance into quaternion + scale for the
    standard PLY format expected by viewers.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        means = gaussians.means.cpu().numpy()
        opacities = gaussians.opacities.cpu().numpy()
        sh_coeffs = gaussians.sh_coeffs.cpu().numpy()

        # Decompose Cholesky into quat + scale
        quats, scales = _covariance_to_quat_scale(gaussians.L_flat.cpu())
        quats = quats.numpy()
        # Convert scales to log-space (standard 3DGS PLY format)
        log_scales = np.log(scales.numpy().clip(min=1e-8))

    N = means.shape[0]
    num_sh = (sh_degree + 1) ** 2
    num_rest = max(0, num_sh - 1) * 3

    header = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    header += "property float x\n"
    header += "property float y\n"
    header += "property float z\n"
    header += "property float nx\n"
    header += "property float ny\n"
    header += "property float nz\n"
    header += "property float f_dc_0\n"
    header += "property float f_dc_1\n"
    header += "property float f_dc_2\n"
    for i in range(num_rest):
        header += f"property float f_rest_{i}\n"
    header += "property float opacity\n"
    header += "property float scale_0\n"
    header += "property float scale_1\n"
    header += "property float scale_2\n"
    header += "property float rot_0\n"
    header += "property float rot_1\n"
    header += "property float rot_2\n"
    header += "property float rot_3\n"
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

"""Export Gaussians to standard 3DGS PLY format for viewing in web viewers."""

import struct
import torch
import numpy as np
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian


def export_ply(gaussians: Gaussian, path: str, sh_degree: int = 0) -> None:
    """Export Gaussians to PLY format compatible with 3DGS viewers.

    Compatible with:
      - antimatter15/splat (https://antimatter15.com/splat/)
      - gsplat viewer
      - SIBR viewer
      - Any standard 3DGS PLY viewer

    Args:
        gaussians: Batch of Gaussians to export.
        path: Output file path (.ply).
        sh_degree: SH degree used (determines number of SH coefficients written).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        means = gaussians.means.cpu().numpy()
        scales = gaussians.scales.cpu().numpy()  # log-space
        opacities = gaussians.opacities.cpu().numpy()  # pre-sigmoid
        sh_coeffs = gaussians.sh_coeffs.cpu().numpy()

    N = means.shape[0]
    num_sh = (sh_degree + 1) ** 2

    # Identity quaternions (no rotation)
    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    # Build PLY header
    # Standard 3DGS PLY format:
    #   x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..N, opacity, scale_0..2, rot_0..3
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
    # DC SH coefficients (RGB)
    header += "property float f_dc_0\n"
    header += "property float f_dc_1\n"
    header += "property float f_dc_2\n"
    # Rest of SH coefficients
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
            # Position
            f.write(struct.pack("<fff", *means[i]))
            # Normals (unused, set to 0)
            f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
            # DC SH coefficients
            f.write(struct.pack("<fff", *sh_coeffs[i, :3]))
            # Rest SH coefficients
            if num_rest > 0:
                rest = sh_coeffs[i, 3:3 + num_rest]
                # Pad if needed
                if len(rest) < num_rest:
                    rest = np.concatenate([rest, np.zeros(num_rest - len(rest))])
                f.write(struct.pack(f"<{num_rest}f", *rest))
            # Opacity (pre-sigmoid, as stored)
            f.write(struct.pack("<f", opacities[i, 0]))
            # Scale (log-space, as stored)
            f.write(struct.pack("<fff", *scales[i]))
            # Rotation quaternion
            f.write(struct.pack("<ffff", *quats[i]))

    print(f"Exported {N:,} Gaussians to {path} ({Path(path).stat().st_size / 1024 / 1024:.1f} MB)")

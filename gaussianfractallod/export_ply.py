"""Export Gaussians to standard 3DGS PLY format for viewing in web viewers."""

import struct
import torch
import numpy as np
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian


def _covariance_to_quat_scale(cov: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert (N, 3, 3) covariance to quaternion (N, 4) + log-scale (N, 3).

    For PLY export only — not used in training/rendering.
    """
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=1e-8)
    scales = torch.sqrt(eigvals)
    log_scales = torch.log(scales)

    # Ensure proper rotation (det > 0)
    det = torch.linalg.det(eigvecs)
    eigvecs = eigvecs * det.sign().unsqueeze(-1).unsqueeze(-1)

    # Rotation matrix to quaternion
    R = eigvecs
    batch_size = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    quat = torch.zeros(batch_size, 4)

    s = torch.sqrt((trace + 1.0).clamp(min=1e-8)) * 2
    mask = trace > 0
    if mask.any():
        quat[mask, 0] = 0.25 * s[mask]
        quat[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
        quat[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
        quat[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

    mask2 = ~mask & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s2 = torch.sqrt((1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]).clamp(min=1e-8)) * 2
        quat[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
        quat[mask2, 1] = 0.25 * s2
        quat[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
        quat[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    mask3 = ~mask & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s3 = torch.sqrt((1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]).clamp(min=1e-8)) * 2
        quat[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
        quat[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
        quat[mask3, 2] = 0.25 * s3
        quat[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    mask4 = ~mask & ~mask2 & ~mask3
    if mask4.any():
        s4 = torch.sqrt((1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]).clamp(min=1e-8)) * 2
        quat[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
        quat[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
        quat[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
        quat[mask4, 3] = 0.25 * s4

    quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    return quat.numpy(), log_scales.numpy()


def export_ply(gaussians: Gaussian, path: str, sh_degree: int = 0) -> None:
    """Export Gaussians to PLY format compatible with 3DGS viewers."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        means = gaussians.means.cpu().numpy()
        opacities = gaussians.opacities.cpu().numpy()
        sh_coeffs = gaussians.sh_coeffs.cpu().numpy()
        cov = gaussians.covariance().cpu()
        quats, log_scales = _covariance_to_quat_scale(cov)

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

"""Gaussian primitive dataclass — batched tensors for position, covariance, opacity, color."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Gaussian:
    """Batch of N Gaussians stored as tensors.

    All tensors have shape (N, ...) where N is the batch dimension.
    Covariance is represented via quaternion (rotation) + log-scale,
    where Sigma = R @ diag(s^2) @ R.T with s = exp(log_scales).
    """

    means: torch.Tensor       # (N, 3) positions
    quats: torch.Tensor       # (N, 4) rotation quaternions, wxyz convention
    log_scales: torch.Tensor  # (N, 3) log-space scales
    opacities: torch.Tensor   # (N, 1) logit-space opacities (apply sigmoid for alpha)
    sh_dc: torch.Tensor       # (N, 1, 3) SH band 0 (DC color)
    sh_rest: torch.Tensor     # (N, K-1, 3) SH bands 1+ (view-dependent), K=(degree+1)²

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    @property
    def sh_coeffs_packed(self) -> torch.Tensor:
        """(N, K, 3) SH coefficients for gsplat rendering."""
        return torch.cat([self.sh_dc, self.sh_rest], dim=1)

    def to(self, device: torch.device) -> "Gaussian":
        return Gaussian(
            means=self.means.to(device),
            quats=self.quats.to(device),
            log_scales=self.log_scales.to(device),
            opacities=self.opacities.to(device),
            sh_dc=self.sh_dc.to(device),
            sh_rest=self.sh_rest.to(device),
        )

    def scales(self) -> torch.Tensor:
        """Compute (N, 3) scales from log-space representation."""
        return torch.exp(self.log_scales)

    def rotation_matrix(self) -> torch.Tensor:
        """Convert quaternions (wxyz) to (N, 3, 3) rotation matrices."""
        q = F.normalize(self.quats, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.stack([
            1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
            2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
            2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)

        return R

    def covariance(self) -> torch.Tensor:
        """Compute (N, 3, 3) covariance matrices: Sigma = R @ diag(s^2) @ R.T"""
        R = self.rotation_matrix()  # (N, 3, 3)
        s = self.scales()           # (N, 3)
        S_sq = torch.diag_embed(s * s)  # (N, 3, 3) diagonal matrix of s^2
        return R @ S_sq @ R.transpose(-1, -2)

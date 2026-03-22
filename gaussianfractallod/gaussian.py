"""Gaussian primitive dataclass — batched tensors for position, scale, opacity, color."""

import torch
from dataclasses import dataclass


@dataclass
class Gaussian:
    """Batch of N Gaussians stored as tensors.

    All tensors have shape (N, ...) where N is the batch dimension.
    Scales are stored in log-space for numerical stability.
    """

    means: torch.Tensor      # (N, 3) positions
    scales: torch.Tensor      # (N, 3) log-space scales
    opacities: torch.Tensor   # (N, 1) sigmoid-space opacities
    sh_coeffs: torch.Tensor   # (N, D) SH coefficients (D depends on SH degree)

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def to(self, device: torch.device) -> "Gaussian":
        return Gaussian(
            means=self.means.to(device),
            scales=self.scales.to(device),
            opacities=self.opacities.to(device),
            sh_coeffs=self.sh_coeffs.to(device),
        )

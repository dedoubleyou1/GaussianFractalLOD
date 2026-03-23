"""Gaussian primitive dataclass — batched tensors for position, covariance, opacity, color."""

import torch
from dataclasses import dataclass


@dataclass
class Gaussian:
    """Batch of N Gaussians stored as tensors.

    All tensors have shape (N, ...) where N is the batch dimension.
    Covariance is represented via lower-triangular Cholesky factor L,
    where Σ = L @ L.T. L has 6 free values per Gaussian (3x3 lower tri).
    """

    means: torch.Tensor       # (N, 3) positions
    L_flat: torch.Tensor      # (N, 6) lower-triangular Cholesky factor, flattened
    opacities: torch.Tensor   # (N, 1) sigmoid-space opacities
    sh_coeffs: torch.Tensor   # (N, D) SH coefficients (D depends on SH degree)

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def to(self, device: torch.device) -> "Gaussian":
        return Gaussian(
            means=self.means.to(device),
            L_flat=self.L_flat.to(device),
            opacities=self.opacities.to(device),
            sh_coeffs=self.sh_coeffs.to(device),
        )

    def L_matrix(self) -> torch.Tensor:
        """Reconstruct (N, 3, 3) lower-triangular matrix from flat representation.

        L_flat stores [l00, l10, l11, l20, l21, l22] where:
            L = [[l00,   0,   0],
                 [l10, l11,   0],
                 [l20, l21, l22]]

        Diagonal elements are exponentiated to ensure positive definiteness.
        """
        N = self.L_flat.shape[0]
        L = torch.zeros(N, 3, 3, device=self.L_flat.device, dtype=self.L_flat.dtype)
        L[:, 0, 0] = torch.exp(self.L_flat[:, 0])
        L[:, 1, 0] = self.L_flat[:, 1]
        L[:, 1, 1] = torch.exp(self.L_flat[:, 2])
        L[:, 2, 0] = self.L_flat[:, 3]
        L[:, 2, 1] = self.L_flat[:, 4]
        L[:, 2, 2] = torch.exp(self.L_flat[:, 5])
        return L

    def covariance(self) -> torch.Tensor:
        """Compute (N, 3, 3) covariance matrices: Σ = L @ L.T"""
        L = self.L_matrix()
        return L @ L.transpose(-1, -2)

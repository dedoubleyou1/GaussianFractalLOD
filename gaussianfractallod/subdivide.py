"""Subdivide parent Gaussians into 8 children via 3 sequential binary cuts.

Uses least-squares optimal formulas for alpha compositing (not statistical
mixture). Children are sized and positioned to minimize the integrated
squared error between the alpha-composited children and the parent.

Formulas (derived from numerical optimization across opacity/spread space):
  sigma_c = sigma_parent * sqrt(1 - f^2/4) * (1 - 0.124 * alpha_p^2)
  alpha_c = [1 - sqrt(1 - alpha_p)] * (0.932 + 0.114 * f)

where f = spread factor (children at +/-f/2 in parent's local frame).
"""

import torch
import torch.nn.functional as F
import math
from gaussianfractallod.gaussian import Gaussian


# Spread factor: children displaced +/-f/2 in parent's local frame per cut axis
SPREAD_FACTOR = 1.0


def rotate_by_quat(quats: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions (wxyz convention).

    Args:
        quats: (N, 4) quaternions, wxyz convention
        vectors: (N, 3) vectors to rotate

    Returns:
        (N, 3) rotated vectors
    """
    q = F.normalize(quats, dim=-1)
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]

    # Rodrigues via quaternion: v' = v + 2w(u x v) + 2(u x (u x v))
    # where u = (x, y, z)
    u = q[:, 1:4]  # (N, 3)
    uv = torch.cross(u, vectors, dim=-1)  # (N, 3)
    uuv = torch.cross(u, uv, dim=-1)     # (N, 3)
    return vectors + 2.0 * (w * uv + uuv)


def subdivide_to_8(parents: Gaussian) -> Gaussian:
    """Subdivide each parent Gaussian into 8 children.

    Applies 3 sequential binary cuts along the parent's local X, Y, Z axes.
    Each cut uses alpha-compositing-aware formulas for child size and opacity.

    Args:
        parents: N parent Gaussians.

    Returns:
        8N child Gaussians.
    """
    current = parents

    for axis in range(3):
        current = _binary_cut_along_axis(current, axis)

    return current


def _binary_cut_along_axis(gaussians: Gaussian, axis: int) -> Gaussian:
    """Split each Gaussian into 2 along a local axis.

    Uses least-squares optimal formulas for alpha compositing:
    - sigma_c accounts for alpha compositing nonlinearity
    - alpha_c matches center opacity exactly under alpha compositing
    """
    N = gaussians.num_gaussians
    device = gaussians.means.device
    f = SPREAD_FACTOR

    # Displacement: rotate axis_vector * scale * f/2 into world space
    axis_vector = torch.zeros(N, 3, device=device)
    axis_vector[:, axis] = 1.0

    scales = gaussians.scales()  # (N, 3)
    local_offset = axis_vector * scales * (f / 2.0)  # (N, 3)
    displacement_world = rotate_by_quat(gaussians.quats, local_offset)  # (N, 3)

    mu_right = gaussians.means + displacement_world
    mu_left = gaussians.means - displacement_world

    # Child opacity: alpha_c = [1 - sqrt(1 - alpha_p)] * (0.932 + 0.114*f)
    # This matches center opacity under alpha compositing
    alpha_p = torch.sigmoid(gaussians.opacities)  # (N, 1)
    alpha_c = (1.0 - torch.sqrt((1.0 - alpha_p).clamp(min=1e-8))) * (0.932 + 0.114 * f)
    alpha_c = alpha_c.clamp(min=1e-6, max=1.0 - 1e-6)
    child_logit = torch.log(alpha_c / (1.0 - alpha_c))

    # Child scale: sigma_c = sigma_p * sqrt(1 - f^2/4) * (1 - 0.124*alpha_p^2)
    # In log-scale terms: add log(scale_factor) on the cut axis
    scale_base = math.sqrt(max(1.0 - f * f / 4.0, 0.01))
    alpha_correction = (1.0 - 0.124 * alpha_p ** 2)  # (N, 1)
    scale_factor = scale_base * alpha_correction  # (N, 1)

    # Child log_scales: inherit parent, adjust cut axis
    log_scale_adjust = torch.zeros(N, 3, device=device)
    log_scale_adjust[:, axis:axis+1] = torch.log(scale_factor)  # broadcast (N, 1)
    child_log_scales = gaussians.log_scales + log_scale_adjust

    # Child quats: inherit parent quaternion (same orientation)
    child_quats = gaussians.quats  # (N, 4)

    # Color: parent's color + small random perturbation to break symmetry
    # Use a separate CPU generator so subdivision doesn't desync the main RNG
    # (different runs split different Gaussians, consuming different amounts)
    _subdiv_gen = torch.Generator()  # CPU generator
    _subdiv_gen.manual_seed(N * 31 + axis * 7)  # deterministic from input shape
    noise_right = torch.randn(gaussians.sh_coeffs.shape, generator=_subdiv_gen, device='cpu').to(device)
    noise_left = torch.randn(gaussians.sh_coeffs.shape, generator=_subdiv_gen, device='cpu').to(device)
    sh_right = gaussians.sh_coeffs + noise_right * 0.01
    sh_left = gaussians.sh_coeffs + noise_left * 0.01

    # Interleave: [right_0, left_0, right_1, left_1, ...]
    means = torch.stack([mu_right, mu_left], dim=1).reshape(2 * N, 3)
    quats = torch.stack([child_quats, child_quats], dim=1).reshape(2 * N, 4)
    log_scales = torch.stack([child_log_scales, child_log_scales], dim=1).reshape(2 * N, 3)
    opacities = torch.stack([child_logit, child_logit], dim=1).reshape(2 * N, 1)
    sh = torch.stack([sh_right, sh_left], dim=1).reshape(2 * N, -1)

    return Gaussian(means=means, quats=quats, log_scales=log_scales,
                    opacities=opacities, sh_coeffs=sh)

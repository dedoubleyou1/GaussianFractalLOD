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

    Applies 3 sequential binary cuts along axes sorted by scale (longest first).
    For isotropic Gaussians this degrades to the standard octree (X, Y, Z).
    For elongated Gaussians, the longest axis is split first, naturally
    correcting aspect ratio: e.g. 10:1 → 5:1 → 2.5:1 → 1.25:1.

    Args:
        parents: N parent Gaussians.

    Returns:
        8N child Gaussians.
    """
    current = parents
    for cut in range(3):
        # Pick the longest axis from current scales (re-evaluated each cut)
        axes = current.log_scales.argmax(dim=-1)
        current = _binary_cut_along_axis(current, axes)

    return current


def _binary_cut_along_axis(gaussians: Gaussian, axes) -> Gaussian:
    """Split each Gaussian into 2 along a local axis.

    Args:
        gaussians: N Gaussians to split.
        axes: int (same axis for all) or (N,) tensor of per-Gaussian axis indices.

    Uses least-squares optimal formulas for alpha compositing:
    - sigma_c accounts for alpha compositing nonlinearity
    - alpha_c matches center opacity exactly under alpha compositing
    """
    N = gaussians.num_gaussians
    device = gaussians.means.device
    f = SPREAD_FACTOR

    # Handle both scalar and per-Gaussian axes
    if isinstance(axes, int):
        axis_vector = torch.zeros(N, 3, device=device)
        axis_vector[:, axes] = 1.0
        axis_idx = axes
    else:
        # Per-Gaussian axis: build one-hot from index tensor
        axis_vector = torch.zeros(N, 3, device=device)
        axis_vector.scatter_(1, axes.unsqueeze(1), 1.0)
        axis_idx = None  # signal that we use per-Gaussian indexing

    scales = gaussians.scales()  # (N, 3)
    local_offset = axis_vector * scales * (f / 2.0)  # (N, 3)
    displacement_world = rotate_by_quat(gaussians.quats, local_offset)  # (N, 3)

    mu_right = gaussians.means + displacement_world
    mu_left = gaussians.means - displacement_world

    # Child opacity: alpha_c = [1 - sqrt(1 - alpha_p)] * (0.932 + 0.114*f)
    alpha_p = torch.sigmoid(gaussians.opacities)  # (N, 1)
    alpha_c = (1.0 - torch.sqrt((1.0 - alpha_p).clamp(min=1e-8))) * (0.932 + 0.114 * f)
    alpha_c = alpha_c.clamp(min=1e-6, max=1.0 - 1e-6)
    child_logit = torch.log(alpha_c / (1.0 - alpha_c))

    # Child scale: sigma_c = sigma_p * sqrt(1 - f^2/4) * (1 - 0.124*alpha_p^2)
    scale_base = math.sqrt(max(1.0 - f * f / 4.0, 0.01))
    alpha_correction = (1.0 - 0.124 * alpha_p ** 2)  # (N, 1)
    scale_factor = scale_base * alpha_correction  # (N, 1)

    # Child log_scales: inherit parent, adjust cut axis only
    log_scale_adjust = torch.zeros(N, 3, device=device)
    if axis_idx is not None:
        log_scale_adjust[:, axis_idx:axis_idx+1] = torch.log(scale_factor)
    else:
        # Per-Gaussian: scatter the scale adjustment to each Gaussian's cut axis
        log_scale_adjust.scatter_(1, axes.unsqueeze(1), torch.log(scale_factor).expand(N, 1))
    child_log_scales = gaussians.log_scales + log_scale_adjust

    # Child quats: inherit parent quaternion (same orientation)
    child_quats = gaussians.quats  # (N, 4)

    # Interleave: [right_0, left_0, right_1, left_1, ...]
    means = torch.stack([mu_right, mu_left], dim=1).reshape(2 * N, 3)
    quats = torch.stack([child_quats, child_quats], dim=1).reshape(2 * N, 4)
    log_scales = torch.stack([child_log_scales, child_log_scales], dim=1).reshape(2 * N, 3)
    opacities = torch.stack([child_logit, child_logit], dim=1).reshape(2 * N, 1)
    # Color: children inherit parent SH exactly
    sh_dc = gaussians.sh_dc.repeat_interleave(2, dim=0)      # (2N, 1, 3)
    sh_rest = gaussians.sh_rest.repeat_interleave(2, dim=0)   # (2N, K-1, 3)

    return Gaussian(means=means, quats=quats, log_scales=log_scales,
                    opacities=opacities, sh_dc=sh_dc, sh_rest=sh_rest)

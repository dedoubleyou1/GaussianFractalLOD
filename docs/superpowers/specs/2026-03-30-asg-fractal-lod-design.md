# ASG-Based Fractal LOD: Design Specification

## Overview

A new parallel approach (alongside the existing SH-based system) that uses Anisotropic Spherical Gaussians (ASGs) to serve a dual purpose: encoding view-dependent appearance AND defining child gaussians for the next LOD level. Each ASG lobe on a parent gaussian materializes into a spatial child gaussian during level transitions, creating a self-similar fractal hierarchy where appearance decomposition directly mirrors spatial decomposition.

This is implemented as an alternative pipeline, not a replacement for the existing SH-based approach. Both systems coexist in the codebase.

## Motivation

The current system uses SH3 (48 color parameters) with no structural relationship between parent and child color representations. The ASG approach:

- Unifies appearance and hierarchy into a single representation
- Enables structural compression: child init params are derived from parent lobes, not stored
- Reduces storage via delta parameterization on top of lobe-derived initializations
- Creates smooth LOD transitions through mass-conserving splits with residual parents

## References

- [SG-Splatting (arXiv:2501.00342)](https://arxiv.org/abs/2501.00342) — spherical gaussians replacing SH in 3DGS
- [Spec-Gaussian (arXiv:2402.15870)](https://arxiv.org/abs/2402.15870) — anisotropic spherical gaussians for specular appearance
- [GitHub: Specular-Gaussians](https://github.com/ingra14m/Specular-Gaussians) — reference implementation

## Core Representation

### Per-Gaussian Parameters

```
Spatial (11 params, unchanged):
  means:      (N, 3)    — 3D position
  quats:      (N, 4)    — rotation quaternion (wxyz)
  log_scales: (N, 3)    — log-space scales per axis
  opacities:  (N, 1)    — logit-space opacity

Appearance:
  base_color: (N, 3)    — DC RGB color

ASG Lobes (9 params per lobe × K lobes, K ∈ {0, 2, 4, 8}):
  lobe_axes:       (N, K, 3)  — unit vectors, lobe directions (μ)
  lobe_amplitudes: (N, K, 3)  — RGB amplitude per lobe (a)
  lobe_sharpness1: (N, K, 1)  — anisotropic sharpness λ₁
  lobe_sharpness2: (N, K, 1)  — anisotropic sharpness λ₂
  lobe_opacities:  (N, K, 1)  — per-lobe opacity (α_lobe)
```

Total: 14 + 9K parameters per gaussian (K=4 → 50, K=8 → 86).

### Color Evaluation

The color at viewing direction **v** for a gaussian:

```
C(v) = base_color + Σᵢ aᵢ · max(v · μᵢ, 0) · exp(-λ₁ᵢ·(v · tᵢ)² - λ₂ᵢ·(v · bᵢ)²)
```

Where `tᵢ`, `bᵢ` are the tangent and bitangent of lobe i, forming an orthonormal frame `[μᵢ, tᵢ, bᵢ]` derived from the lobe axis and the gaussian's rotation.

### ASG Properties

- Lobes are localized (compact angular support), unlike SH which is global
- The product of two ASGs is another ASG (closed-form)
- Rotation is trivial (rotate the axis vector)
- No Gibbs ringing at truncation boundaries
- Two sharpness parameters (λ₁, λ₂) enable anisotropic angular response (e.g., stretched specular highlights)

## ASG Lobe → Child Gaussian Mapping

Each lobe on a parent gaussian defines a child gaussian for the next LOD level.

### Derived Parameters

```
Child position:    P_child = P_parent + d(μ) · μ
Child base_color:  a (lobe amplitude RGB)
Child opacity:     α_lobe (per-lobe opacity)
Child rotation:    quaternion from ASG frame [μ, t, b]
Child scales:
  tangential₁ = d / √λ₁
  tangential₂ = d / √λ₂
  radial      = √(tangential₁ · tangential₂)   — geometric mean
```

### Distance from Parent Spread

The offset distance is derived from the parent gaussian's spatial extent along the lobe direction — no stored parameter:

```
d(μ) = √(μᵀ Σ μ)
```

Where Σ = R · diag(s²) · Rᵀ is the parent's covariance matrix. This places children at the parent's "surface" in each lobe direction.

## Mass-Conserving Split

When materializing children, the parent persists as a residual with conserved center of mass and total volume.

### Conservation Math

Mass of a gaussian = integral under the curve:

```
m = α · (2π)^(3/2) · s₁ · s₂ · s₃
```

For each child from lobe i:

```
m_childᵢ = α_lobeᵢ · s_tan1ᵢ · s_tan2ᵢ · s_radialᵢ  (proportional)
```

Residual parent:

```
M_residual = M_parent - Σ m_childᵢ
P_residual = P_parent - Σ (m_childᵢ · dᵢ · μᵢ) / M_residual
S_residual = S_parent · (M_residual / M_parent)^(1/3)
α_residual = α_parent  (unchanged — same opacity, smaller size)
base_color_residual = base_color_parent  (unchanged)
```

### Properties

- **Center of mass preserved**: residual shifts opposite to children
- **Total volume preserved**: residual shrinks as children take volume
- **Opacity unchanged**: the residual is the same "stuff," just less of it
- **Graceful decay**: residuals shrink each level, naturally becoming negligible without explicit deletion
- **Smooth LOD**: at any level, residuals from all ancestors + children at that level form a complete representation

## Training Pipeline

### Per-Level Flow

Every level follows the same process:

```
1. Init
   Level 0: random initialization, train position + base_color from scratch
   Level 1+: initialized from parent lobes (position, scale, rotation, opacity, base_color)

2. Gradient accumulation
   Train with L_image at this level's resolution
   Accumulate gradient magnitude and directions per gaussian

3. Allocate lobes
   |∇| magnitude → lobe count K ∈ {0, 2, 4, 8} (thresholds from config)
   ∇ directions → lobe axes μᵢ (initialized toward highest-gradient directions)
   Initialize: λ₁, λ₂ broad; opacity = parent opacity; amplitude from base_color

4. Joint training (multi-objective)
   L = L_image(res_N) + α(t) · L_children(res_{N+1})
   α(t) ramps from 0 → α_max over training

5. Materialize
   Mass-conserving split → residual parent + children
   All get delta parameterization
   Proceed to next level
```

### Multi-Objective Loss

```
L = L_image + α(t) · L_children

L_image    = (1 - w_ssim) · L₁ + w_ssim · (1 - SSIM)    at resolution res_N
L_children = (1 - w_ssim) · L₁ + w_ssim · (1 - SSIM)    at resolution res_{N+1}
```

**L_children** (hypothetical children loss):
1. For each gaussian with lobes, compute hypothetical children via mass-conserving split
2. Render residual parent + children at res_{N+1}
3. Compare to ground truth at res_{N+1}
4. Backprop through child params → lobe params → gaussian params (fully differentiable)

**α(t) schedule**: ramps from 0 to α_max over training. Initially lobes optimize purely for appearance quality; gradually the children quality objective becomes significant. α_max is a tunable hyperparameter (~0.5–1.0).

### Resolution Schedule

Unchanged from current system:

```
res_N = min(32 × (√2)^N, 800)
```

### Gradient-Based Lobe Allocation

Uses the existing tiered splitting thresholds applied to accumulated gradient magnitudes:

```
K = 0: |∇| below split_1cut_threshold (kept as-is)
K = 2: |∇| > split_1cut_threshold
K = 4: |∇| > split_2cut_threshold
K = 8: |∇| > split_3cut_threshold
```

Lobe axes initialized toward the directions with highest accumulated gradient magnitude (directional gradient histogram).

## Compression

### Three-Layer Compression Stack

1. **Delta parameterization**: children store residuals from lobe-derived init values
2. **QAT (existing)**: STE quantization on deltas at 8–16 bits
3. **Structural compression**: child init params are recomputed from parent lobes at decode time, never stored

### Storage Model

```
Level 0 (root):
  Full trained params stored: position, scale, rotation, opacity, base_color
  + lobe params (encode level 1 children)

Level 1+ (derived):
  Init = derived from parent lobes (recomputed, NOT stored)
  + deltas (stored, quantized via QAT)
  + lobe params (encode next level's children)
```

### Storage Per Child

```
Current (SH3 + delta + QAT@16bit):
  55 delta floats × 16 bit = 110 bytes per child

ASG (K=4 + delta + QAT@16bit):
  ~10 delta params × 16 bit = 20 bytes per child
  + parent lobe cost (9 params × 16 bit = 18 bytes, amortized across siblings)
  ≈ 38 bytes per child (2.9× smaller)
```

## Rendering

### ASG Color Evaluation

The color evaluation replaces the current SH evaluation in the rendering pipeline:

```python
def evaluate_asg_color(view_dir, base_color, lobe_axes, lobe_amplitudes,
                       lobe_sharpness1, lobe_sharpness2, lobe_opacities,
                       gaussian_rotation):
    color = base_color
    for each lobe i:
        mu_i = lobe_axes[i]
        t_i, b_i = compute_tangent_bitangent(mu_i, gaussian_rotation)
        smooth = max(dot(view_dir, mu_i), 0)
        aniso = exp(-sharpness1[i] * dot(view_dir, t_i)**2
                    - sharpness2[i] * dot(view_dir, b_i)**2)
        color += lobe_amplitudes[i] * smooth * aniso
    return color
```

### Rendering Paths

- **Primary (gsplat)**: needs custom color evaluation kernel or pre-evaluation of ASG color per gaussian per view, passed as `colors_precomp`
- **Fallback (PyTorch)**: straightforward implementation of ASG evaluation in Python

### LOD Rendering

At any target LOD level, render:
- All residual parents from ancestor levels (accumulated through splits)
- All gaussians at the target level
- This gives a complete scene representation at any level of detail

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SG type | Anisotropic (2 sharpnesses) | Captures elliptical highlights; 2 tangential axes map to child ellipsoid |
| Diffuse term | DC base_color + SG lobes | Base color is gaussian's identity; lobes encode variation + children |
| Radial child scale | Geometric mean of tangential | Zero extra storage; reasonable approximation to start |
| Distance | Derived from parent spread | No stored parameter; children placed at parent's "surface" |
| Branching | Gradient-driven 0/2/4/8 | Proven adaptive strategy from current system |
| Training | Multi-objective (approach C) | Single phase; α ramp balances appearance vs child quality |
| Volume siphoning | Mass-conserving (scale reduction + CoM shift) | Physically motivated; parent shrinks rather than fading |
| Per-lobe opacity | Yes (1 extra param) | Trainable via joint loss; decouples brightness from transparency |

## Open Questions / Future Work

- **Tangent/bitangent construction**: exact method for deriving t, b from μ and gaussian rotation. Gram-Schmidt from μ + one gaussian axis? Or store a rotation angle per lobe?
- **α_max tuning**: optimal balance between appearance and children quality objectives
- **Lobe axis initialization**: exact algorithm for extracting dominant gradient directions (PCA on directional gradient histogram? Top-K directions?)
- **gsplat integration**: custom CUDA kernel for ASG evaluation vs pre-evaluation tradeoff
- **Negative lobes**: should lobe amplitudes be allowed to go negative (subtractive color)? Could improve expressiveness.
- **Lobe count for roots**: fixed or also gradient-driven from Phase 1?
- **Regularization on lobes**: should there be regularization to prevent degenerate configurations (e.g., all lobes pointing the same direction)?

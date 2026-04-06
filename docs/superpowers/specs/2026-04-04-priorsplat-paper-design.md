# PriorSplat Paper Design

**Title:** PriorSplat: Recursive Gaussian Priors for Object Reconstruction with Built-in LOD

**Venue:** arxiv preprint

**Tone:** Accessible and explanatory — prioritize clarity of core ideas, intuition, and figures over exhaustive comparisons.

**Scope:** Bounded objects with clearly defined boundaries via alpha masks. Not a general scene-level method.

---

## Core Insight

A 3D Gaussian can be viewed as a statistical approximation of the colors and occlusion coming from an approximate area of space. Viewed this way, a Gaussian acts as a natural prior for the next level of Gaussians within that region. Every component of the system follows from this observation:

- **Initialization**: the coarse approximation seeds initial placement, scale, and appearance for the finer level
- **Compression**: children are parameterized as deltas from the prior — only corrections are stored
- **LOD**: render any level of the hierarchy for bandwidth-adaptive fidelity
- **Adaptive allocation**: split where the prior is insufficient (high gradient = poor summary)

This is distinct from mechanical hierarchy approaches (Octree-GS, Scaffold-GS) that impose structure without exploiting the statistical nature of Gaussians themselves.

---

## Paper Structure

### Abstract

A 3D Gaussian can be viewed as a statistical approximation of the appearance and geometry of a spatial region. We show that this perspective yields a principled prior for finer-grained Gaussians within that region. We build PriorSplat, an object reconstruction method that exploits this insight: given a bounded object with known alpha masks, we compute the coarsest prior — a single Gaussian — in closed form from silhouette statistics, then recursively refine through gradient-driven subdivision. Each child Gaussian is parameterized as a delta from its parent prior, yielding a self-similar hierarchy with built-in level-of-detail, structural compression, and principled initialization at every scale.

### Section 1: Introduction (~1 page)

- 3DGS recap: flat unstructured cloud, no hierarchy, no LOD, initialization depends on SfM
- The problem with flat representations: no compression structure, no principled way to add detail where needed, no LOD
- Key insight: a Gaussian already IS a statistical summary of a region — this gives you a prior for refinement for free
- Scope: bounded objects with alpha masks — the silhouette provides the starting point for the coarsest prior
- What the prior gives you: initialization, compression, LOD, adaptive allocation
- Brief positioning: prior-based hierarchy vs. mechanical hierarchy (Octree-GS, Scaffold-GS)
- Contributions:
  1. Framing of Gaussians as recursive statistical priors for hierarchical reconstruction
  2. Closed-form geometric root fitting from silhouette moments (no optimization)
  3. Gradient-driven adaptive subdivision with optimal child placement
  4. Delta parameterization enabling structural compression
  5. Multi-resolution training with built-in LOD

### Section 2: Related Work (~1 page)

- **3D Gaussian Splatting**: original method, properties, limitations for our setting
- **Hierarchical / LOD approaches**: Octree-GS (TPAMI 2025), Scaffold-GS (CVPR 2024) — impose structure mechanically without prior-based reasoning
- **Compression methods**: CompGS, ContextGS (NeurIPS 2024), PCGS — post-hoc compression vs. built-in structural compression
- **Object reconstruction**: methods that assume bounded objects with masks, relationship to NeRF-based object methods
- **Initialization**: SfM-dependent vs. our closed-form approach

### Section 3: Method (~3-4 pages)

**3.1 Gaussians as Statistical Priors**

The conceptual foundation. A Gaussian is not a bounded chunk of an object — it is a soft, distributional claim about space: *there is approximately this much occlusion and this color, approximately in this area, distributed approximately like this.* We define **coverage** as the accumulated occlusion contributed by a Gaussian under alpha compositing. A Gaussian then encodes:

- Position (mean): approximately where in space
- Spatial extent (covariance): roughly how the contribution is distributed
- Coverage (accumulated occlusion): how much it occludes
- Appearance (SH coefficients): what it looks like from different directions

Both coarse and fine levels are soft approximations of the same underlying scene, just at different fidelities. When the coarse approximation is insufficient (training gradients remain high), it decomposes into a finer set of soft claims that together remain consistent with the coarse one. Delta parameterization (`child = coarse_init + learned_correction`) makes that consistency the default: refinement adds local variation without abandoning the coarse distributional claim.

**3.2 Geometric Root Fitting (Phase 1)**

For bounded objects with alpha masks, the coarsest prior (a single Gaussian for the whole object) can be computed in closed form:

1. Per-view 2D Gaussian fitting from silhouette statistics (centroid, covariance, coverage)
2. 3D position via least-squares ray intersection across views
3. 3D covariance via Voronoi-weighted averaging (corrects for camera clustering bias)
4. Phase 1b: SH coefficient refinement on frozen geometry

Properties: no optimization, no local minima, guaranteed silhouette coverage. The alpha mask boundary makes this possible — without a clear foreground/background separation, moment computation would be ill-defined.

**3.3 Adaptive Subdivision (Phase 2)**

Gradient magnitude measures "prior insufficiency" — where the current level's Gaussian summaries fail to explain the observations:

- Tiered splitting based on accumulated gradient norm:
  - High gradient (>0.004): 8 children (3 binary cuts)
  - Medium gradient (>0.001): 4 children (2 cuts)
  - Low gradient (>0.00025): 2 children (1 cut)
  - Below threshold or low opacity: keep as-is
- Sequential binary cuts along longest axis
- Optimal child placement via least-squares formulas:
  - Position: offset +/- f/2 along longest axis
  - Scale: corrected for spread factor and opacity
  - Opacity: linear area-preserving formula with floor

**3.4 Multi-Resolution Training**

Resolution schedule: `res_N = min(32 * sqrt(2)^N, max_res)`

- Pixel count doubles per level — each level resolves detail at the scale where it matters
- Per-level training with 3DGS-style learning rate schedule (100x position LR decay)
- Regularization:
  - Position: penalize drift from expected offset (normalized by parent scale)
  - Scale: penalize extreme volume changes
  - Aspect ratio: prevent degenerate elongation

**3.5 Compression via Hierarchy**

- Child initialization is fully determined by parent parameters and split decisions — never stored
- Only deltas (corrections to the prior) need storage
- Optional quantization-aware training via straight-through estimator (8-16 bit)
- Storage analysis: bytes per Gaussian compared to flat approaches

### Section 4: Results (~2 pages)

- **Benchmark**: NeRF Synthetic (Lego and potentially others), PSNR / SSIM / LPIPS
- **Per-level quality progression**: visualize the hierarchy building up from coarse to fine, showing how each level's prior gets refined
- **LOD demonstration**: quality vs. Gaussian count at each level — single model, multiple fidelity levels
- **Compression**: storage vs. quality against flat 3DGS and compression baselines
- **Ablations**:
  - Geometric root fitting vs. random/Adam initialization
  - Delta parameterization vs. absolute parameters
  - Adaptive splitting vs. uniform splitting
  - Effect of resolution schedule
- **Figures**:
  - Hierarchy visualization (level 0 through N, showing progressive refinement)
  - Gaussian ellipsoid rendering at each level
  - Split pattern visualization (where the model allocates more Gaussians)

### Section 5: Discussion & Future Work (~0.5 page)

- ASG direction: unifying appearance lobes with spatial hierarchy (appearance lobes as seeds for child Gaussians)
- Extension to multi-object scenes
- Streaming applications: bandwidth-adaptive rendering from the hierarchy
- Limitations: requires alpha masks, designed for bounded objects, current SH-based appearance

### Section 6: Conclusion (~0.25 page)

Restate the core insight, summarize results, point to future directions.

---

## Key Figures to Prepare

1. **Teaser figure**: Object at multiple LOD levels from the same model, showing coarse-to-fine
2. **Method overview**: Pipeline diagram showing root fitting -> adaptive splitting -> multi-resolution training
3. **Prior concept figure**: Parent Gaussian as summary, children as refinements with deltas
4. **Geometric root fitting**: 2D silhouette moments -> 3D Gaussian, multi-view diagram
5. **Subdivision visualization**: Gradient heatmap -> split decisions -> child placement
6. **Per-level results**: Grid showing rendering at each hierarchy level
7. **Quantitative plots**: PSNR/storage curves, ablation bar charts

## Key Equations to Present

1. Gaussian as region summary: mean, covariance, coverage, appearance
2. Delta parameterization: `theta_child = theta_init(parent) + delta`
3. Geometric root fitting: moment computation, ray intersection, Voronoi weighting
4. Child placement: position offset, scale correction, opacity formula
5. Resolution schedule: `res_N = 32 * sqrt(2)^N`
6. Loss function: `(1 - w_ssim) * L1 + w_ssim * (1 - SSIM)` plus regularization terms

---

## Positioning Against Related Work

| Method | Hierarchy | Prior-based | Built-in LOD | Compression | Init |
|--------|-----------|-------------|--------------|-------------|------|
| 3DGS | None | No | No | No | SfM |
| Octree-GS | Octree | No (mechanical) | Yes | No | SfM |
| Scaffold-GS | Anchor+MLP | No | No | Partial | SfM |
| CompGS | None | No | No | Yes (post-hoc) | SfM |
| **PriorSplat** | **Recursive Gaussian** | **Yes** | **Yes** | **Yes (structural)** | **Closed-form** |

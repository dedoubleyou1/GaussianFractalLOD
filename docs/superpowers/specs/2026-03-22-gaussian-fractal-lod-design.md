# GaussianFractalLOD: Hierarchical Split-Variable Gaussian Splatting for Unified LoD and Compression

**Date:** 2026-03-22
**Status:** Draft
**Type:** Research prototype targeting paper submission

## 1. Problem Statement

Current 3D Gaussian Splatting methods treat level-of-detail (LoD) and compression as separate problems. LoD systems (Octree-GS) organize Gaussians into spatial hierarchies but don't exploit the hierarchy for compression. Compression systems (ContextGS, CompGS, PCGS) use anchor-based prediction but don't produce renderable intermediate representations. No existing method unifies these two capabilities in a single recursive structure.

## 2. Core Thesis

A single recursive Gaussian hierarchy can simultaneously provide level-of-detail rendering and compression by making renderable parent Gaussians serve as consistency constraints for their children. The parent Gaussian is a statistical approximation of its subtree's aggregate appearance — this makes it both a renderable coarse representation and a natural compression prior for its descendants.

## 3. Key Insight: Children as a Coupled System

The parent Gaussian defines conservation laws that its children must jointly satisfy:

- **Opacity conservation:** Σ αᵢ = α_parent
- **Center of mass:** Σ πᵢμᵢ = μ_parent
- **Appearance conservation:** Σ πᵢcᵢ(d) = c_parent(d) for all viewing directions d
- **Spatial extent:** Σ πᵢ[Σᵢ + (μᵢ - μ_p)(μᵢ - μ_p)ᵀ] = Σ_parent

These are not regularization losses — they structurally reduce the degrees of freedom. Changing one child forces compensating changes in the others. The children live in a lower-dimensional subspace than unconstrained Gaussians.

This insight draws directly from the **Reversible Jump MCMC** literature on Gaussian mixture split-merge moves (Richardson & Green, 1997; Jain & Neal, 2004), which parameterize the decomposition of one Gaussian into children such that conservation is guaranteed by construction.

## 4. Representation: Tree of Split Variables

### 4.1 Binary Split Cascade

Rather than splitting one parent into 8 children directly, each octant level is decomposed into **3 consecutive binary splits** (1→2→4→8). This means every split in the entire hierarchy uses the well-understood Richardson & Green 1→2 parameterization exactly. No generalization is needed.

Benefits:
- R&G conservation is exact at every split — no approximation
- LoD granularity is 3× finer than octree: 1→2→4→8→16→32→... instead of 1→8→64→512
- Each intermediate binary level is a renderable LoD
- Split variables don't need to be axis-aligned — optimizer finds natural split planes
- Initialize axis-aligned (X→Y→Z cycling) as equivalent to octree starting point

### 4.2 Split Variables per Binary Split

Each binary split stores:

| Variable | Values | Description |
|----------|--------|-------------|
| Occupancy | 2 bits | Which children are active |
| Mass partition u₁ | 1 | Logit of opacity ratio |
| Position split u₂ | 3 | 3D displacement in parent's local frame |
| Variance split u₃ | 3 | Per-axis variance partition |
| Color split Δc | D | SH coefficient deviations |
| **Total** | **D + 7** | |

Where D depends on SH degree: D=3 (SH0), D=12 (SH1), D=27 (SH2), D=48 (SH3).

### 4.3 Child Derivation (Bijective Map)

Given parent parameters (μ_p, Σ_p, α_p, c_p) and split variables (u₁, u₂, u₃, Δc):

```
π_A = σ(u₁),  π_B = 1 - π_A          # mass partition
α_A = π_A · α_p,  α_B = π_B · α_p     # opacity conservation

L_p = cholesky(Σ_p)                    # parent's local frame
μ_A = μ_p + L_p · u₂ · √(π_B/π_A)    # position (R&G formula)
μ_B = μ_p - L_p · u₂ · √(π_A/π_B)    # center-of-mass conserved

σ²_A = (1 - u₂²) · σ²_p · u₃ / π_A   # per-axis variance split
σ²_B = (1 - u₂²) · σ²_p · (1-u₃) / π_B

c_A = c_p + Δc                         # color split
c_B = c_p - (π_A/π_B) · Δc            # weighted zero-sum conserved
```

All operations are differentiable. No Gaussian except the root is stored directly — every descendant is derived by recursively applying splits.

### 4.4 Reconstruction and LoD

```python
def reconstruct(parent, split_tree, target_depth):
    if target_depth == 0 or parent not in split_tree:
        return [parent]  # render this Gaussian directly

    child_a, child_b = derive_children(parent, split_tree[parent])
    result = []
    if child_a is not None:
        result += reconstruct(child_a, split_tree, target_depth - 1)
    if child_b is not None:
        result += reconstruct(child_b, split_tree, target_depth - 1)
    return result
```

LoD selection: choose how deep to recurse. Each level is self-consistent because conservation is built into the parameterization.

## 5. Training Procedure

### 5.1 Phase 1: Root Gaussian Fitting

Train a small set of root-level Gaussians (~8–64) using standard 3DGS training with a controlled Gaussian budget. These form the coarsest renderable LoD — a "forest" of trees. Standard densification and pruning, capped at the target root count. Freeze root parameters after convergence.

### 5.2 Phase 2: Level-by-Level Split Fitting

For each binary depth level L = 1, 2, ..., max_depth:

**Initialize split variables:**
- Occupancy: both children active
- Mass partition: u₁ = 0 (uniform, π = 0.5 each)
- Position split: u₂ = canonical axis displacement (cycling X→Y→Z)
- Variance split: u₃ = 0.5 (uniform)
- Color split: Δc = 0 (children start with parent's color)

**Optimize:**
- Reconstruct all Gaussians down to level L via the split-variable chain
- Rasterize with gsplat's differentiable rasterizer
- Loss: L₁ + λ·SSIM against training views (standard 3DGS loss)
- Gradients flow through: rendering → Gaussian params → split derivation → split variables
- Adam optimizer on split variables only (all ancestor levels frozen)

**Prune:**
- After convergence, set occupancy bit to 0 for children whose mass partition π_i < ε
- Renormalize remaining child's mass to maintain conservation
- Pruned subtrees don't exist — parent Gaussian renders in their place

**Freeze and descend.**

### 5.3 Phase 3: Joint Fine-Tuning (Optional, Experiment B)

Unfreeze all split variables across all levels. Train with multi-scale loss:

```
L_total = Σ_d  w_d · L_render(reconstruct(roots, splits, depth=d), GT)
```

Each iteration renders at multiple LoD depths. This allows levels to co-adapt — parents may adjust to better serve their children. Risk: coarse-LoD quality may degrade. The depth weights w_d control this tradeoff.

## 6. Parameter Budget

### 6.1 Raw Storage

A binary tree with N leaves has N-1 internal nodes (splits). Per split: D+7 values.

| Configuration | Values/Split | 200K Leaves | Float32 Size | vs. 3DGS (45 MB) |
|--------------|-------------|-------------|-------------|-------------------|
| SH3 (D=48) | 55 | 11.0M | 42 MB | 0.93× |
| SH2 (D=27) | 34 | 6.8M | 26 MB | 0.58× |
| SH1 (D=12) | 19 | 3.8M | 14.5 MB | 0.32× |
| SH0 (D=3) | 10 | 2.0M | 7.6 MB | 0.17× |

### 6.2 Compression Advantages

Raw parameter count at the same SH degree is similar to vanilla 3DGS. The compression wins come from:

- **Sparsity:** Pruned subtrees reduce split count. 40% pruning → 0.56× storage.
- **Entropy:** Split variables cluster near canonical values (uniform partition, small displacements, near-zero color deviations) — highly compressible with entropy coding.
- **Variable SH depth:** Coarse levels can use lower SH degrees than leaf levels.
- **Quantization:** Split variables live in bounded, well-behaved ranges.

Conservative estimate with entropy coding (~4 bits/value): a 200K-leaf tree at SH3 could compress to ~5 MB.

## 7. Evaluation Plan

### 7.1 Dataset

NeRF Synthetic / Blender: 8 scenes, 100 training views, 200 test views. Standard bounded-object benchmark.

### 7.2 Experiments

**Experiment 1 — Proof of Concept:** Train a single scene (Lego) end-to-end. Render at every LoD level (binary depths 0, 3, 6, 9, 12, 15, 18). Show smooth visual degradation and PSNR vs. depth curve.

**Experiment 2 — Quality Benchmark:** Full-depth rendering on all 8 scenes. Compare PSNR, SSIM, LPIPS against 3DGS, Scaffold-GS, and Octree-GS. Target: within 1–2 dB of Scaffold-GS.

**Experiment 3 — Compression:** Measure total storage of split variables (quantized, float32) vs. equivalent flat Gaussian set. Measure entropy of split-variable distributions. Plot rate-distortion curves varying quantization bitwidth.

**Experiment 4 — Core Ablation:** Compare three variants:
- **Full model:** Renderable parents + conservation-constrained split variables
- **Ablation A:** Same hierarchy, but parents are not rendered / not constrained to be good approximations
- **Ablation B:** Parents rendered but conservation constraints removed — children are independent residuals

Show that the full model produces smaller residuals (better compression) AND better coarse-LoD rendering than both ablations.

**Experiment 5 — Analysis:** Occupancy statistics per level, mass partition distributions, residual magnitudes by depth, split plane orientation visualization, hierarchy depth maps on 3D objects.

### 7.3 Stretch Goals

- Phase 3 joint fine-tuning comparison
- Progressive streaming demonstration
- Adaptive distance-based LoD rendering with consistent frame rates

## 8. Technical Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Rasterizer | gsplat | Decoupled function-call API — accepts arbitrary Gaussian tensors |
| Autograd | PyTorch | Split-variable math is standard differentiable ops |
| Data loading | Custom | NeRF Synthetic JSON + images (~50 lines) |
| Training | Custom loop | Need custom phases (root fit → level-by-level splits) |
| Metrics | torchmetrics | PSNR, SSIM, LPIPS |
| Logging | tensorboard or wandb | Training curves, rendered images |
| Language | Python | No custom CUDA needed — hierarchy is pure PyTorch |

## 9. Relationship to Prior Work

| Method | Relationship | Key Difference |
|--------|-------------|----------------|
| Scaffold-GS | Anchors predict child Gaussians | Our anchors (parents) are renderable; children are coupled through conservation |
| Octree-GS | Octree LoD structure | Our levels have parent-child prediction, not independent; binary splits give 3× finer LoD |
| ContextGS | Coarse-to-fine anchor context for entropy coding | Our context is structural (conservation), not learned; hierarchy is the representation, not just a coding tool |
| CompGS | Anchor + residual prediction | Our residuals are split variables with built-in conservation, not independent per-child embeddings |
| PCGS | Progressive bitstream | Our progressivity comes from tree depth, not masking/quantization refinement |
| R&G 1997 | Split-merge MCMC for Gaussian mixtures | We adopt their split parameterization as a differentiable, trainable representation |

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Conservation constraints are too rigid, limit quality | Full-depth rendering can't match baselines | Phase 3 joint fine-tuning relaxes constraints; can soften conservation to a loss term |
| Coarse-to-fine training gets stuck in bad local minima | Poor quality at fine levels | Canonical octant initialization provides reasonable starting point; can warm-start from a standard 3DGS fit |
| Cholesky decomposition is numerically unstable | NaN gradients during training | Use log-space parameterization for scales; add small epsilon to diagonal before Cholesky |
| 18 binary levels is too many sequential training phases | Training is slow | Parallelize: train all splits at the same depth simultaneously; batch across scenes |
| Split variables don't compress well in practice | Compression claim doesn't hold | Entropy analysis in Exp 3 will reveal this early; variable SH depth and sparsity may suffice |

## References

- Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH 2023.
- Lu et al. "Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering." CVPR 2024.
- Ren et al. "Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians." TPAMI 2025.
- Wang et al. "ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model." NeurIPS 2024.
- Liu et al. "CompGS: Efficient 3D Scene Representation via Compressed Gaussian Splatting." ACM MM 2024.
- Chen et al. "PCGS: Progressive Compression of 3D Gaussian Splatting." AAAI 2026.
- Richardson & Green. "On Bayesian Analysis of Mixtures with an Unknown Number of Components." JRSS-B 1997.
- Jain & Neal. "A Split-Merge Markov Chain Monte Carlo Procedure for the Dirichlet Process Mixture Model." Journal of Computational and Graphical Statistics 2004.

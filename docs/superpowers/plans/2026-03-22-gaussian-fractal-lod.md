# GaussianFractalLOD Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research prototype that implements hierarchical split-variable Gaussian splatting, unifying LoD rendering and compression on NeRF Synthetic scenes.

**Architecture:** A binary tree of R&G split variables, where each parent Gaussian is renderable and children are derived via conservation-preserving bijective maps. Training is coarse-to-fine: fit root Gaussians, then fit split variables level-by-level. gsplat provides the differentiable rasterizer; all hierarchy math is pure PyTorch.

**Tech Stack:** Python, PyTorch, gsplat, torchmetrics, LPIPS, Google Colab, tensorboard

**Spec:** `docs/superpowers/specs/2026-03-22-gaussian-fractal-lod-design.md`

---

## File Structure

```
GaussianFractalLOD/
├── gaussianfractallod/
│   ├── __init__.py              # Package init, version
│   ├── config.py                # Dataclass config with defaults
│   ├── data.py                  # NeRF Synthetic dataset loading
│   ├── gaussian.py              # Gaussian dataclass (μ, Σ, α, c)
│   ├── derive.py                # R&G child derivation (bijective map)
│   ├── split_tree.py            # Binary split tree data structure
│   ├── reconstruct.py           # Tree → flat Gaussian list at target depth
│   ├── render.py                # gsplat rendering wrapper
│   ├── loss.py                  # L1 + SSIM loss
│   ├── train_roots.py           # Phase 1: root Gaussian fitting
│   ├── train_splits.py          # Phase 2: level-by-level split fitting
│   ├── prune.py                 # Mass-based child pruning
│   ├── checkpoint.py            # Save/load state dicts
│   └── eval.py                  # PSNR, SSIM, LPIPS evaluation
├── tests/
│   ├── test_derive.py           # Split derivation + conservation tests
│   ├── test_split_tree.py       # Tree structure tests
│   ├── test_reconstruct.py      # Reconstruction tests
│   ├── test_render.py           # Rendering integration tests
│   ├── test_data.py             # Data loading tests
│   ├── test_loss.py             # Loss function tests
│   ├── test_train_roots.py      # Root training tests
│   ├── test_train_splits.py     # Split training tests
│   ├── test_prune.py            # Pruning tests
│   └── test_checkpoint.py       # Checkpoint round-trip tests
├── notebooks/
│   └── train_colab.ipynb        # Thin Colab entry point
├── configs/
│   └── default.yaml             # Default hyperparameters
├── setup.py                     # Package install
└── requirements.txt             # Dependencies
```

---

## Chunk 1: Foundation — Data, Math, Rendering

### Task 1: Project Scaffolding

**Files:**
- Create: `setup.py`
- Create: `requirements.txt`
- Create: `gaussianfractallod/__init__.py`
- Create: `gaussianfractallod/config.py`
- Create: `configs/default.yaml`

- [ ] **Step 1: Create requirements.txt**

```
torch>=2.0.0
gsplat>=1.0.0
torchmetrics>=1.0.0
lpips>=0.1.4
Pillow>=9.0.0
numpy>=1.24.0
pyyaml>=6.0
tensorboard>=2.12.0
pytest>=7.0.0
```

- [ ] **Step 2: Create setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="gaussianfractallod",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
)
```

- [ ] **Step 3: Create config dataclass**

File: `gaussianfractallod/config.py`

```python
from dataclasses import dataclass, field


@dataclass
class Config:
    # Data
    data_dir: str = ""
    image_scale: float = 1.0

    # Root fitting (Phase 1)
    num_roots: int = 32
    root_lr: float = 1e-3
    root_iterations: int = 10000
    root_convergence_window: int = 1000

    # Split fitting (Phase 2)
    max_binary_depth: int = 18
    split_lr: float = 5e-3
    split_iterations_per_level: int = 5000
    split_convergence_window: int = 500

    # SH degree
    sh_degree: int = 3

    # Pruning
    prune_mass_threshold: float = 0.01

    # Loss
    ssim_weight: float = 0.2

    # Checkpointing
    checkpoint_dir: str = "checkpoints"

    # Rendering
    background_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
```

- [ ] **Step 4: Create default.yaml**

File: `configs/default.yaml`

```yaml
data_dir: ""
image_scale: 1.0
num_roots: 32
root_lr: 0.001
root_iterations: 10000
root_convergence_window: 1000
max_binary_depth: 18
split_lr: 0.005
split_iterations_per_level: 5000
split_convergence_window: 500
sh_degree: 3
prune_mass_threshold: 0.01
ssim_weight: 0.2
checkpoint_dir: "checkpoints"
background_color: [1.0, 1.0, 1.0]
```

- [ ] **Step 5: Create package init**

File: `gaussianfractallod/__init__.py`

```python
"""GaussianFractalLOD: Hierarchical split-variable Gaussian splatting."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Commit**

```bash
git add setup.py requirements.txt configs/default.yaml gaussianfractallod/__init__.py gaussianfractallod/config.py
git commit -m "feat: project scaffolding with config and dependencies"
```

---

### Task 2: Gaussian Data Structure

**Files:**
- Create: `gaussianfractallod/gaussian.py`
- Create: `tests/test_derive.py` (first part)

- [ ] **Step 1: Write test for Gaussian construction**

File: `tests/test_derive.py`

```python
import torch
from gaussianfractallod.gaussian import Gaussian


def test_gaussian_construction():
    g = Gaussian(
        means=torch.zeros(1, 3),
        scales=torch.ones(1, 3),
        opacities=torch.ones(1, 1),
        sh_coeffs=torch.zeros(1, 3),  # SH0: just DC
    )
    assert g.means.shape == (1, 3)
    assert g.scales.shape == (1, 3)
    assert g.opacities.shape == (1, 1)
    assert g.sh_coeffs.shape == (1, 3)


def test_gaussian_num_gaussians():
    g = Gaussian(
        means=torch.zeros(5, 3),
        scales=torch.ones(5, 3),
        opacities=torch.ones(5, 1),
        sh_coeffs=torch.zeros(5, 3),
    )
    assert g.num_gaussians == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/davidwallin/Developer/GaussianFractalLOD && python -m pytest tests/test_derive.py::test_gaussian_construction -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Gaussian dataclass**

File: `gaussianfractallod/gaussian.py`

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_derive.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gaussianfractallod/gaussian.py tests/test_derive.py
git commit -m "feat: Gaussian dataclass with batched tensor storage"
```

---

### Task 3: R&G Child Derivation (Core Math)

**Files:**
- Create: `gaussianfractallod/derive.py`
- Modify: `tests/test_derive.py` (add derivation tests)

This is the most critical module — the bijective map from (parent + split variables) → (child_a, child_b).

- [ ] **Step 1: Write conservation tests**

Append to `tests/test_derive.py`:

```python
from gaussianfractallod.derive import derive_children, SplitVariables


def _make_parent(sh_dim=3):
    """Helper: single parent Gaussian with known parameters."""
    return Gaussian(
        means=torch.tensor([[1.0, 2.0, 3.0]]),
        scales=torch.tensor([[0.0, 0.0, 0.0]]),  # log-space, so scale=1
        opacities=torch.tensor([[0.8]]),
        sh_coeffs=torch.randn(1, sh_dim),
    )


def _make_split_vars(sh_dim=3):
    """Helper: split variables with known values."""
    return SplitVariables(
        mass_logit=torch.tensor([0.0]),  # uniform split
        position_split=torch.tensor([[0.1, 0.0, 0.0]]),
        variance_split=torch.tensor([[0.5, 0.5, 0.5]]),  # uniform
        color_split=torch.zeros(1, sh_dim),
    )


def test_opacity_conservation():
    parent = _make_parent()
    sv = _make_split_vars()
    child_a, child_b = derive_children(parent, sv)
    total_opacity = child_a.opacities + child_b.opacities
    torch.testing.assert_close(total_opacity, parent.opacities, atol=1e-6, rtol=1e-6)


def test_center_of_mass_conservation():
    parent = _make_parent()
    sv = _make_split_vars()
    child_a, child_b = derive_children(parent, sv)
    pi_a = torch.sigmoid(sv.mass_logit)
    pi_b = 1.0 - pi_a
    weighted_mean = pi_a * child_a.means + pi_b * child_b.means
    torch.testing.assert_close(weighted_mean, parent.means, atol=1e-5, rtol=1e-5)


def test_color_conservation():
    parent = _make_parent()
    sv = SplitVariables(
        mass_logit=torch.tensor([0.5]),  # asymmetric
        position_split=torch.tensor([[0.1, 0.2, -0.1]]),
        variance_split=torch.tensor([[0.3, 0.6, 0.5]]),
        color_split=torch.randn(1, 3) * 0.1,
    )
    child_a, child_b = derive_children(parent, sv)
    pi_a = torch.sigmoid(sv.mass_logit)
    pi_b = 1.0 - pi_a
    weighted_color = pi_a * child_a.sh_coeffs + pi_b * child_b.sh_coeffs
    torch.testing.assert_close(weighted_color, parent.sh_coeffs, atol=1e-5, rtol=1e-5)


def test_derivation_is_differentiable():
    parent = _make_parent()
    sv = SplitVariables(
        mass_logit=torch.tensor([0.0], requires_grad=True),
        position_split=torch.tensor([[0.1, 0.0, 0.0]], requires_grad=True),
        variance_split=torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True),
        color_split=torch.zeros(1, 3, requires_grad=True),
    )
    child_a, child_b = derive_children(parent, sv)
    loss = child_a.means.sum() + child_b.means.sum()
    loss.backward()
    assert sv.mass_logit.grad is not None
    assert sv.position_split.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_derive.py::test_opacity_conservation -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement derive.py**

File: `gaussianfractallod/derive.py`

```python
"""R&G-style binary split: derive two children from a parent Gaussian + split variables.

Conservation guarantees (by construction):
  - Opacity: alpha_A + alpha_B = alpha_parent
  - Center of mass: pi_A * mu_A + pi_B * mu_B = mu_parent
  - Color: pi_A * c_A + pi_B * c_B = c_parent
"""

import torch
from dataclasses import dataclass
from gaussianfractallod.gaussian import Gaussian


EPS = 1e-6


@dataclass
class SplitVariables:
    """Split variables for a single binary split.

    All tensors have shape (N, ...) for N parallel splits.
    """

    mass_logit: torch.Tensor       # (N,) logit of mass partition ratio
    position_split: torch.Tensor   # (N, 3) displacement in parent's local frame
    variance_split: torch.Tensor   # (N, 3) per-axis variance partition in (0, 1)
    color_split: torch.Tensor      # (N, D) SH coefficient deviation


def derive_children(
    parent: Gaussian, split_vars: SplitVariables
) -> tuple[Gaussian, Gaussian]:
    """Derive two children from parent + split variables.

    Uses Richardson & Green (1997) split parameterization generalized to 3D.
    All conservation laws are satisfied by construction.

    Args:
        parent: Batch of N parent Gaussians.
        split_vars: Batch of N split variable sets.

    Returns:
        (child_a, child_b): Two batches of N child Gaussians.
    """
    # Mass partition
    pi_a = torch.sigmoid(split_vars.mass_logit).unsqueeze(-1)  # (N, 1)
    pi_b = 1.0 - pi_a

    # Opacity conservation: alpha_i = pi_i * alpha_parent
    alpha_a = pi_a * parent.opacities
    alpha_b = pi_b * parent.opacities

    # Position split in parent's local frame
    # Parent scale defines the local frame (diagonal approximation)
    parent_scale = torch.exp(parent.scales)  # (N, 3), from log-space
    u2 = split_vars.position_split  # (N, 3)

    # R&G position formulas (center-of-mass conserved by construction):
    #   mu_A = mu_p + scale_p * u2 * sqrt(pi_B / pi_A)
    #   mu_B = mu_p - scale_p * u2 * sqrt(pi_A / pi_B)
    sqrt_ratio_ab = torch.sqrt(pi_b / (pi_a + EPS) + EPS)  # (N, 1)
    sqrt_ratio_ba = torch.sqrt(pi_a / (pi_b + EPS) + EPS)  # (N, 1)

    mu_a = parent.means + parent_scale * u2 * sqrt_ratio_ab
    mu_b = parent.means - parent_scale * u2 * sqrt_ratio_ba

    # Variance split (per-axis, in log-space for stability)
    # u3 in (0, 1) via sigmoid; split parent variance
    u3 = torch.sigmoid(split_vars.variance_split)  # (N, 3)
    u2_sq = u2 ** 2

    # R&G variance formulas (per-axis approximation):
    #   var_A = (1 - u2^2) * var_p * u3 / pi_A
    #   var_B = (1 - u2^2) * var_p * (1 - u3) / pi_B
    parent_var = torch.exp(2.0 * parent.scales)  # (N, 3), variance = scale^2
    shared_factor = (1.0 - u2_sq).clamp(min=EPS) * parent_var

    var_a = shared_factor * u3 / (pi_a + EPS)
    var_b = shared_factor * (1.0 - u3) / (pi_b + EPS)

    # Convert back to log-scale
    scale_a = 0.5 * torch.log(var_a.clamp(min=EPS))
    scale_b = 0.5 * torch.log(var_b.clamp(min=EPS))

    # Color conservation: pi_A * c_A + pi_B * c_B = c_parent
    #   c_A = c_parent + delta_c
    #   c_B = c_parent - (pi_A / pi_B) * delta_c
    delta_c = split_vars.color_split
    c_a = parent.sh_coeffs + delta_c
    c_b = parent.sh_coeffs - (pi_a / (pi_b + EPS)) * delta_c

    child_a = Gaussian(means=mu_a, scales=scale_a, opacities=alpha_a, sh_coeffs=c_a)
    child_b = Gaussian(means=mu_b, scales=scale_b, opacities=alpha_b, sh_coeffs=c_b)

    return child_a, child_b
```

- [ ] **Step 4: Run all derivation tests**

Run: `python -m pytest tests/test_derive.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add gaussianfractallod/derive.py tests/test_derive.py
git commit -m "feat: R&G child derivation with conservation guarantees"
```

---

### Task 4: Split Tree Data Structure

**Files:**
- Create: `gaussianfractallod/split_tree.py`
- Create: `tests/test_split_tree.py`

- [ ] **Step 1: Write split tree tests**

File: `tests/test_split_tree.py`

```python
import torch
from gaussianfractallod.split_tree import SplitTree


def test_create_empty_tree():
    tree = SplitTree(num_roots=4, sh_dim=3)
    assert tree.num_roots == 4
    assert tree.depth == 0
    assert tree.num_splits == 0


def test_add_level():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    assert tree.depth == 1
    # 2 roots → 2 splits at level 0
    assert tree.num_splits == 2


def test_add_two_levels():
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()  # 1 root → 2 children
    tree.add_level()  # 2 children → 4 grandchildren (2 new splits)
    assert tree.depth == 2
    assert tree.num_splits == 3  # 1 + 2


def test_split_vars_are_parameters():
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    params = list(tree.level_parameters(0))
    assert len(params) == 4  # mass_logit, position_split, variance_split, color_split
    for p in params:
        assert p.requires_grad


def test_split_vars_initialized_correctly():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    sv = tree.get_level_split_vars(0)
    # Mass logit should be 0 (uniform split)
    torch.testing.assert_close(sv.mass_logit, torch.zeros(2))
    # Color split should be 0
    torch.testing.assert_close(sv.color_split, torch.zeros(2, 3))


def test_occupancy_mask():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    # Both children active by default
    assert tree.get_occupancy(0).all()
    # Prune child B of root 0
    tree.set_occupancy(level=0, node_idx=0, child_b=False)
    occ = tree.get_occupancy(0)
    assert occ[0, 0] == True   # root 0, child A
    assert occ[0, 1] == False  # root 0, child B
    assert occ[1, 0] == True   # root 1, child A
    assert occ[1, 1] == True   # root 1, child B
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_split_tree.py::test_create_empty_tree -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement SplitTree**

File: `gaussianfractallod/split_tree.py`

```python
"""Binary split tree: stores split variables for each level of the hierarchy.

The tree is organized by levels. Level 0 splits the root Gaussians.
Level 1 splits the children from level 0. Each level stores split
variables for all nodes at that depth.

Node count at level L: num_roots * 2^L (before pruning).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from gaussianfractallod.derive import SplitVariables


class SplitLevel(nn.Module):
    """Split variables for all nodes at one level of the tree."""

    def __init__(self, num_nodes: int, sh_dim: int, axis_idx: int = 0):
        super().__init__()
        self.num_nodes = num_nodes
        self.sh_dim = sh_dim

        # Learnable split variables
        self.mass_logit = nn.Parameter(torch.zeros(num_nodes))
        self.position_split = nn.Parameter(torch.zeros(num_nodes, 3))
        self.variance_split = nn.Parameter(torch.zeros(num_nodes, 3))
        self.color_split = nn.Parameter(torch.zeros(num_nodes, sh_dim))

        # Initialize position split along cycling axis (X→Y→Z)
        with torch.no_grad():
            self.position_split[:, axis_idx] = 0.25

        # Occupancy: (num_nodes, 2) — [child_a_active, child_b_active]
        self.register_buffer(
            "occupancy", torch.ones(num_nodes, 2, dtype=torch.bool)
        )

    def get_split_vars(self) -> SplitVariables:
        return SplitVariables(
            mass_logit=self.mass_logit,
            position_split=self.position_split,
            variance_split=self.variance_split,
            color_split=self.color_split,
        )


class SplitTree(nn.Module):
    """Binary split tree organized by levels."""

    def __init__(self, num_roots: int, sh_dim: int):
        super().__init__()
        self.num_roots = num_roots
        self.sh_dim = sh_dim
        self.levels = nn.ModuleList()

    @property
    def depth(self) -> int:
        return len(self.levels)

    @property
    def num_splits(self) -> int:
        return sum(level.num_nodes for level in self.levels)

    def add_level(self) -> None:
        """Add a new split level at the bottom of the tree.

        Node count: each active child at the previous level becomes a
        node to split at this level. occupancy is (num_nodes, 2) bools,
        so sum() counts total active children across both slots.
        """
        if self.depth == 0:
            num_nodes = self.num_roots
        else:
            prev_level = self.levels[-1]
            # Count active children: each True in occupancy (num_nodes, 2)
            # is one child that becomes a node at the next level
            num_nodes = int(prev_level.occupancy.sum().item())

        axis_idx = self.depth % 3  # Cycle X→Y→Z
        level = SplitLevel(num_nodes, self.sh_dim, axis_idx)
        self.levels.append(level)

    def get_level_split_vars(self, level_idx: int) -> SplitVariables:
        return self.levels[level_idx].get_split_vars()

    def level_parameters(self, level_idx: int):
        """Yield learnable parameters for a specific level."""
        level = self.levels[level_idx]
        yield level.mass_logit
        yield level.position_split
        yield level.variance_split
        yield level.color_split

    def get_occupancy(self, level_idx: int) -> torch.Tensor:
        return self.levels[level_idx].occupancy

    def set_occupancy(
        self, level: int, node_idx: int, child_a: bool = True, child_b: bool = True
    ) -> None:
        self.levels[level].occupancy[node_idx, 0] = child_a
        self.levels[level].occupancy[node_idx, 1] = child_b
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_split_tree.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add gaussianfractallod/split_tree.py tests/test_split_tree.py
git commit -m "feat: SplitTree data structure with level-based organization"
```

---

### Task 5: Reconstruction — Tree to Flat Gaussian List

**Files:**
- Create: `gaussianfractallod/reconstruct.py`
- Create: `tests/test_reconstruct.py`

- [ ] **Step 1: Write reconstruction tests**

File: `tests/test_reconstruct.py`

```python
import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.reconstruct import reconstruct


def _make_roots(n=2, sh_dim=3):
    return Gaussian(
        means=torch.randn(n, 3),
        scales=torch.zeros(n, 3),
        opacities=torch.ones(n, 1) * 0.8,
        sh_coeffs=torch.randn(n, sh_dim),
    )


def test_reconstruct_depth_zero_returns_roots():
    roots = _make_roots(2)
    tree = SplitTree(num_roots=2, sh_dim=3)
    result = reconstruct(roots, tree, target_depth=0)
    assert result.num_gaussians == 2
    torch.testing.assert_close(result.means, roots.means)


def test_reconstruct_depth_one_doubles_count():
    roots = _make_roots(2, sh_dim=3)
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    result = reconstruct(roots, tree, target_depth=1)
    assert result.num_gaussians == 4  # 2 roots × 2 children


def test_reconstruct_depth_two():
    roots = _make_roots(1, sh_dim=3)
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    tree.add_level()
    result = reconstruct(roots, tree, target_depth=2)
    assert result.num_gaussians == 4  # 1 root → 2 → 4


def test_reconstruct_respects_occupancy():
    roots = _make_roots(1, sh_dim=3)
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    # Prune child B of root 0
    tree.set_occupancy(level=0, node_idx=0, child_b=False)
    result = reconstruct(roots, tree, target_depth=1)
    assert result.num_gaussians == 1  # Only child A survives


def test_reconstruct_deeper_than_tree_returns_leaves():
    roots = _make_roots(1, sh_dim=3)
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()  # Only 1 level
    result = reconstruct(roots, tree, target_depth=5)
    # Can only go 1 level deep, so 2 Gaussians
    assert result.num_gaussians == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reconstruct.py::test_reconstruct_depth_zero_returns_roots -v`
Expected: FAIL

- [ ] **Step 3: Implement reconstruct.py**

File: `gaussianfractallod/reconstruct.py`

```python
"""Reconstruct flat Gaussian list from roots + split tree at a target depth.

Iteratively applies split derivation level-by-level, respecting occupancy masks.
Uses iterative (not recursive) approach for GPU-friendly batched operations.
"""

import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.derive import derive_children


def reconstruct(
    roots: Gaussian, tree: SplitTree, target_depth: int
) -> Gaussian:
    """Reconstruct Gaussians at a target binary depth.

    Args:
        roots: The root-level Gaussians (N_roots,).
        tree: The split tree containing split variables.
        target_depth: How many binary split levels to apply.
            0 = return roots, 1 = first split, etc.

    Returns:
        Gaussian batch containing all leaf Gaussians at the target depth.
    """
    if target_depth == 0 or tree.depth == 0:
        return roots

    current = roots
    actual_depth = min(target_depth, tree.depth)

    for level_idx in range(actual_depth):
        split_vars = tree.get_level_split_vars(level_idx)
        occupancy = tree.get_occupancy(level_idx)  # (num_nodes, 2)

        child_a, child_b = derive_children(current, split_vars)

        # Collect active children
        # occupancy[:, 0] = child_a active, occupancy[:, 1] = child_b active
        parts_means = []
        parts_scales = []
        parts_opacities = []
        parts_sh = []

        mask_a = occupancy[:, 0]  # (num_nodes,)
        mask_b = occupancy[:, 1]

        if mask_a.any():
            parts_means.append(child_a.means[mask_a])
            parts_scales.append(child_a.scales[mask_a])
            parts_opacities.append(child_a.opacities[mask_a])
            parts_sh.append(child_a.sh_coeffs[mask_a])

        if mask_b.any():
            parts_means.append(child_b.means[mask_b])
            parts_scales.append(child_b.scales[mask_b])
            parts_opacities.append(child_b.opacities[mask_b])
            parts_sh.append(child_b.sh_coeffs[mask_b])

        if not parts_means:
            # All children pruned — return parent (shouldn't happen in practice)
            return current

        current = Gaussian(
            means=torch.cat(parts_means, dim=0),
            scales=torch.cat(parts_scales, dim=0),
            opacities=torch.cat(parts_opacities, dim=0),
            sh_coeffs=torch.cat(parts_sh, dim=0),
        )

    return current
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_reconstruct.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add gaussianfractallod/reconstruct.py tests/test_reconstruct.py
git commit -m "feat: batched reconstruction from split tree to flat Gaussians"
```

---

### Task 6: NeRF Synthetic Data Loading

**Files:**
- Create: `gaussianfractallod/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write data loading tests**

File: `tests/test_data.py`

```python
import pytest
import torch
import json
import os
from pathlib import Path
from PIL import Image
from gaussianfractallod.data import NerfSyntheticDataset


@pytest.fixture
def mock_nerf_scene(tmp_path):
    """Create a minimal mock NeRF Synthetic scene."""
    # Create transforms_train.json
    frames = []
    for i in range(3):
        img_path = f"train/r_{i:03d}"
        frames.append({
            "file_path": f"./{img_path}",
            "transform_matrix": [
                [1, 0, 0, float(i)],
                [0, 1, 0, 0],
                [0, 0, 1, 4],
                [0, 0, 0, 1],
            ],
        })

    transforms = {"camera_angle_x": 0.6911, "frames": frames}
    with open(tmp_path / "transforms_train.json", "w") as f:
        json.dump(transforms, f)

    # Create dummy images (RGBA)
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    for i in range(3):
        img = Image.new("RGBA", (32, 32), (128, 64, 200, 255))
        img.save(train_dir / f"r_{i:03d}.png")

    return tmp_path


def test_load_dataset(mock_nerf_scene):
    dataset = NerfSyntheticDataset(str(mock_nerf_scene), split="train")
    assert len(dataset) == 3


def test_dataset_returns_image_and_camera(mock_nerf_scene):
    dataset = NerfSyntheticDataset(str(mock_nerf_scene), split="train")
    image, camera = dataset[0]
    assert image.shape == (32, 32, 3)  # H, W, C (RGB, no alpha)
    assert image.dtype == torch.float32
    assert "viewmat" in camera
    assert "K" in camera
    assert camera["viewmat"].shape == (4, 4)
    assert camera["K"].shape == (3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data.py::test_load_dataset -v`
Expected: FAIL

- [ ] **Step 3: Implement data.py**

File: `gaussianfractallod/data.py`

```python
"""NeRF Synthetic / Blender dataset loader."""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class NerfSyntheticDataset(Dataset):
    """Loads a NeRF Synthetic scene (Blender format).

    Each item returns (image, camera) where:
      - image: (H, W, 3) float32 tensor in [0, 1], alpha-composited on white bg
      - camera: dict with 'viewmat' (4x4), 'K' (3x3), 'width', 'height'
    """

    def __init__(self, root: str, split: str = "train", scale: float = 1.0):
        self.root = Path(root)
        self.scale = scale

        with open(self.root / f"transforms_{split}.json") as f:
            meta = json.load(f)

        self.camera_angle_x = meta["camera_angle_x"]
        self.frames = meta["frames"]

        # Pre-load all images
        self.images = []
        self.cameras = []
        for frame in self.frames:
            img_path = self.root / f"{frame['file_path']}.png"
            if not img_path.exists():
                # Try without ./ prefix
                img_path = self.root / f"{frame['file_path'].lstrip('./')}.png"

            img = Image.open(img_path)
            if scale != 1.0:
                w, h = img.size
                img = img.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )

            img_np = np.array(img, dtype=np.float32) / 255.0

            # Alpha composite onto white background
            if img_np.shape[2] == 4:
                alpha = img_np[:, :, 3:4]
                rgb = img_np[:, :, :3] * alpha + (1.0 - alpha)
            else:
                rgb = img_np[:, :, :3]

            self.images.append(torch.from_numpy(rgb))

            # Build camera
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            w2c = torch.linalg.inv(c2w)

            h, w_px = rgb.shape[:2]
            focal = 0.5 * w_px / np.tan(0.5 * self.camera_angle_x)

            K = torch.tensor(
                [[focal, 0, w_px / 2.0],
                 [0, focal, h / 2.0],
                 [0, 0, 1]],
                dtype=torch.float32,
            )

            self.cameras.append({
                "viewmat": w2c,
                "K": K,
                "width": w_px,
                "height": h,
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.cameras[idx]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_data.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add gaussianfractallod/data.py tests/test_data.py
git commit -m "feat: NeRF Synthetic dataset loader"
```

---

### Task 7: gsplat Rendering Wrapper

**Files:**
- Create: `gaussianfractallod/render.py`
- Create: `gaussianfractallod/loss.py`
- Create: `tests/test_render.py`
- Create: `tests/test_loss.py`

- [ ] **Step 1: Write render tests**

File: `tests/test_render.py`

```python
import pytest
import torch

# Skip all tests if gsplat is not installed (allows running other tests locally)
gsplat = pytest.importorskip("gsplat")

from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.render import render_gaussians


def test_render_produces_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("gsplat requires CUDA")

    gaussians = Gaussian(
        means=torch.tensor([[0.0, 0.0, 3.0]], device=device),
        scales=torch.tensor([[-1.0, -1.0, -1.0]], device=device),
        opacities=torch.tensor([[5.0]], device=device),  # pre-sigmoid
        sh_coeffs=torch.tensor([[1.0, 0.0, 0.0]], device=device),
    )

    viewmat = torch.eye(4, device=device)
    K = torch.tensor(
        [[200.0, 0, 64.0], [0, 200.0, 64.0], [0, 0, 1]], device=device
    )

    image = render_gaussians(
        gaussians, viewmat=viewmat, K=K,
        width=128, height=128,
        background=torch.ones(3, device=device),
    )
    assert image.shape == (128, 128, 3)
    assert image.dtype == torch.float32
    # Should not be all white (background) since we have a Gaussian
    assert not torch.allclose(image, torch.ones_like(image), atol=0.1)
```

- [ ] **Step 2: Write loss tests**

File: `tests/test_loss.py`

```python
import torch
from gaussianfractallod.loss import rendering_loss


def test_loss_zero_for_identical():
    img = torch.rand(64, 64, 3)
    loss = rendering_loss(img, img, ssim_weight=0.2)
    assert loss.item() < 1e-5


def test_loss_positive_for_different():
    pred = torch.rand(64, 64, 3)
    gt = torch.rand(64, 64, 3)
    loss = rendering_loss(pred, gt, ssim_weight=0.2)
    assert loss.item() > 0


def test_loss_is_differentiable():
    pred = torch.rand(64, 64, 3, requires_grad=True)
    gt = torch.rand(64, 64, 3)
    loss = rendering_loss(pred, gt, ssim_weight=0.2)
    loss.backward()
    assert pred.grad is not None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_render.py tests/test_loss.py -v`
Expected: FAIL

- [ ] **Step 4: Implement render.py**

File: `gaussianfractallod/render.py`

```python
"""gsplat rendering wrapper: Gaussian batch → rendered image."""

import torch
from gsplat import rasterization
from gaussianfractallod.gaussian import Gaussian


def render_gaussians(
    gaussians: Gaussian,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    background: torch.Tensor | None = None,
    sh_degree: int | None = None,
) -> torch.Tensor:
    """Render Gaussians to an image using gsplat.

    Args:
        gaussians: Batch of N Gaussians.
        viewmat: (4, 4) world-to-camera matrix.
        K: (3, 3) camera intrinsics.
        width, height: Image dimensions.
        background: (3,) background color. Defaults to white.
        sh_degree: SH degree for color. If None, inferred from sh_coeffs.

    Returns:
        (H, W, 3) rendered image.
    """
    device = gaussians.means.device

    if background is None:
        background = torch.ones(3, device=device)

    N = gaussians.num_gaussians

    # gsplat expects quaternions (N, 4) for rotation. We use identity
    # (axis-aligned) since the split tree operates in parent local frames
    # via diagonal scale approximation. Full rotation support can be added
    # later if the per-axis variance approximation proves insufficient.
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0  # identity quaternion

    # Infer SH degree from coefficient count
    D = gaussians.sh_coeffs.shape[-1]
    if sh_degree is None:
        if D <= 3:
            sh_degree = 0
        elif D <= 12:
            sh_degree = 1
        elif D <= 27:
            sh_degree = 2
        else:
            sh_degree = 3

    # Reshape SH coeffs for gsplat: (N, num_sh, 3)
    num_sh = (sh_degree + 1) ** 2
    expected_dim = num_sh * 3
    assert gaussians.sh_coeffs.shape[-1] >= expected_dim, (
        f"SH coeffs dim {gaussians.sh_coeffs.shape[-1]} < expected {expected_dim} "
        f"for sh_degree={sh_degree}"
    )
    sh_coeffs_3 = gaussians.sh_coeffs[:, :expected_dim].reshape(N, num_sh, 3)

    renders, alphas, meta = rasterization(
        means=gaussians.means,
        quats=quats,
        scales=torch.exp(gaussians.scales),
        opacities=torch.sigmoid(gaussians.opacities.squeeze(-1)),
        colors=sh_coeffs_3,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=sh_degree,
        backgrounds=background.unsqueeze(0),
    )

    return renders[0]  # (H, W, 3)
```

- [ ] **Step 5: Implement loss.py**

File: `gaussianfractallod/loss.py`

```python
"""Rendering loss: L1 + weighted SSIM."""

import torch
import torch.nn.functional as F


def _ssim_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create Gaussian window for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    return window.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute SSIM between two images.

    Args:
        pred, gt: (H, W, 3) images in [0, 1].

    Returns:
        Scalar SSIM value.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # (H, W, 3) → (1, 3, H, W)
    pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
    gt_4d = gt.permute(2, 0, 1).unsqueeze(0)

    window = _ssim_window().to(pred.device)
    window = window.expand(3, 1, -1, -1)  # (3, 1, 11, 11)

    mu1 = F.conv2d(pred_4d, window, padding=5, groups=3)
    mu2 = F.conv2d(gt_4d, window, padding=5, groups=3)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_4d ** 2, window, padding=5, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(gt_4d ** 2, window, padding=5, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred_4d * gt_4d, window, padding=5, groups=3) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


def rendering_loss(
    pred: torch.Tensor, gt: torch.Tensor, ssim_weight: float = 0.2
) -> torch.Tensor:
    """Combined L1 + SSIM loss.

    Args:
        pred, gt: (H, W, 3) images.
        ssim_weight: Weight for SSIM term.

    Returns:
        Scalar loss value.
    """
    l1 = F.l1_loss(pred, gt)
    ssim_val = ssim(pred, gt)
    return (1.0 - ssim_weight) * l1 + ssim_weight * (1.0 - ssim_val)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_loss.py -v`
Expected: PASS (render tests may skip without CUDA)

- [ ] **Step 7: Commit**

```bash
git add gaussianfractallod/render.py gaussianfractallod/loss.py tests/test_render.py tests/test_loss.py
git commit -m "feat: gsplat rendering wrapper and L1+SSIM loss"
```

---

## Chunk 2: Training Pipeline

### Task 8: Checkpoint Save/Load

**Files:**
- Create: `gaussianfractallod/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write checkpoint tests**

File: `tests/test_checkpoint.py`

```python
import torch
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_round_trip(tmp_path):
    roots = Gaussian(
        means=torch.randn(4, 3),
        scales=torch.randn(4, 3),
        opacities=torch.randn(4, 1),
        sh_coeffs=torch.randn(4, 3),
    )
    tree = SplitTree(num_roots=4, sh_dim=3)
    tree.add_level()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, roots, tree, phase=1, level=0)

    loaded_roots, loaded_tree, meta = load_checkpoint(path)
    torch.testing.assert_close(loaded_roots.means, roots.means)
    torch.testing.assert_close(loaded_roots.scales, roots.scales)
    assert loaded_tree.depth == tree.depth
    assert loaded_tree.num_roots == tree.num_roots
    assert meta["phase"] == 1
    assert meta["level"] == 0


def test_checkpoint_preserves_split_vars(tmp_path):
    roots = Gaussian(
        means=torch.randn(2, 3),
        scales=torch.randn(2, 3),
        opacities=torch.randn(2, 1),
        sh_coeffs=torch.randn(2, 3),
    )
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    # Modify split vars
    with torch.no_grad():
        tree.levels[0].mass_logit.fill_(0.7)
        tree.levels[0].color_split.fill_(0.1)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, roots, tree, phase=2, level=0)

    _, loaded_tree, _ = load_checkpoint(path)
    torch.testing.assert_close(
        loaded_tree.levels[0].mass_logit,
        torch.tensor([0.7, 0.7]),
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL

- [ ] **Step 3: Implement checkpoint.py**

File: `gaussianfractallod/checkpoint.py`

```python
"""Checkpoint save/load for root Gaussians and split tree."""

import torch
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree


def save_checkpoint(
    path: str | Path,
    roots: Gaussian,
    tree: SplitTree,
    phase: int,
    level: int,
    **extra_meta,
) -> None:
    """Save training state to disk."""
    state = {
        "roots": {
            "means": roots.means.detach().cpu(),
            "scales": roots.scales.detach().cpu(),
            "opacities": roots.opacities.detach().cpu(),
            "sh_coeffs": roots.sh_coeffs.detach().cpu(),
        },
        "tree": tree.state_dict(),
        "tree_meta": {
            "num_roots": tree.num_roots,
            "sh_dim": tree.sh_dim,
            "depth": tree.depth,
        },
        "meta": {"phase": phase, "level": level, **extra_meta},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path, device: torch.device | None = None
) -> tuple[Gaussian, SplitTree, dict]:
    """Load training state from disk.

    Returns:
        (roots, tree, meta)
    """
    state = torch.load(path, map_location="cpu", weights_only=False)

    roots = Gaussian(
        means=state["roots"]["means"],
        scales=state["roots"]["scales"],
        opacities=state["roots"]["opacities"],
        sh_coeffs=state["roots"]["sh_coeffs"],
    )

    tm = state["tree_meta"]
    tree = SplitTree(num_roots=tm["num_roots"], sh_dim=tm["sh_dim"])
    for _ in range(tm["depth"]):
        tree.add_level()
    tree.load_state_dict(state["tree"])

    if device is not None:
        roots = roots.to(device)
        tree = tree.to(device)

    return roots, tree, state["meta"]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_checkpoint.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add gaussianfractallod/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: checkpoint save/load for roots and split tree"
```

---

### Task 9: Phase 1 — Root Gaussian Fitting

**Files:**
- Create: `gaussianfractallod/train_roots.py`
- Create: `tests/test_train_roots.py`

- [ ] **Step 1: Write root training test**

File: `tests/test_train_roots.py`

```python
import pytest
import torch

gsplat = pytest.importorskip("gsplat")

from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.train_roots import init_roots, train_roots_step


def test_init_roots():
    roots = init_roots(num_roots=8, sh_degree=0, device=torch.device("cpu"))
    assert roots.num_gaussians == 8
    assert roots.means.shape == (8, 3)
    assert roots.sh_coeffs.shape == (8, 3)  # SH0 = 3 (RGB DC)
    assert roots.means.requires_grad


def test_train_roots_step_reduces_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("gsplat requires CUDA")

    roots = init_roots(num_roots=4, sh_degree=0, device=device)
    optimizer = torch.optim.Adam(
        [roots.means, roots.scales, roots.opacities, roots.sh_coeffs],
        lr=1e-2,
    )

    # Dummy target: red image
    gt_image = torch.zeros(64, 64, 3, device=device)
    gt_image[:, :, 0] = 1.0

    viewmat = torch.eye(4, device=device)
    K = torch.tensor(
        [[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1]], device=device
    )
    camera = {"viewmat": viewmat, "K": K, "width": 64, "height": 64}

    loss1 = train_roots_step(roots, gt_image, camera, optimizer, ssim_weight=0.2)
    loss2 = train_roots_step(roots, gt_image, camera, optimizer, ssim_weight=0.2)
    # Loss should generally decrease (not guaranteed in 2 steps, but likely)
    assert loss1.item() > 0
    assert loss2.item() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_roots.py::test_init_roots -v`
Expected: FAIL

- [ ] **Step 3: Implement train_roots.py**

File: `gaussianfractallod/train_roots.py`

```python
"""Phase 1: Train root-level Gaussians with standard splatting loss.

Simplification: This prototype uses a fixed root count without adaptive
densification/pruning (unlike standard 3DGS). Roots are initialized
randomly and optimized via gradient descent. Densification can be added
later if root quality is insufficient.
"""

import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.render import render_gaussians
from gaussianfractallod.loss import rendering_loss


def init_roots(
    num_roots: int, sh_degree: int = 0, device: torch.device = torch.device("cpu")
) -> Gaussian:
    """Initialize root Gaussians with random positions on a unit sphere.

    All parameters require grad for optimization.
    """
    sh_dim = 3 * ((sh_degree + 1) ** 2)

    # Random points on unit sphere, scaled to scene bounds
    means = torch.randn(num_roots, 3, device=device) * 0.5
    means.requires_grad_(True)

    # Start with moderate scale (log-space)
    scales = torch.full((num_roots, 3), -1.0, device=device)
    scales.requires_grad_(True)

    # Start opaque (sigmoid(2.0) ≈ 0.88)
    opacities = torch.full((num_roots, 1), 2.0, device=device)
    opacities.requires_grad_(True)

    # Random SH coefficients
    sh_coeffs = torch.randn(num_roots, sh_dim, device=device) * 0.1
    sh_coeffs.requires_grad_(True)

    return Gaussian(
        means=means, scales=scales, opacities=opacities, sh_coeffs=sh_coeffs
    )


def train_roots_step(
    roots: Gaussian,
    gt_image: torch.Tensor,
    camera: dict,
    optimizer: torch.optim.Optimizer,
    ssim_weight: float = 0.2,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single training step for root Gaussians.

    Returns:
        Scalar loss value.
    """
    optimizer.zero_grad()

    rendered = render_gaussians(
        roots,
        viewmat=camera["viewmat"],
        K=camera["K"],
        width=camera["width"],
        height=camera["height"],
        background=background,
    )

    loss = rendering_loss(rendered, gt_image, ssim_weight=ssim_weight)
    loss.backward()
    optimizer.step()

    return loss.detach()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_train_roots.py -v`
Expected: PASS (CUDA tests skip without GPU)

- [ ] **Step 5: Commit**

```bash
git add gaussianfractallod/train_roots.py tests/test_train_roots.py
git commit -m "feat: Phase 1 root Gaussian initialization and training step"
```

---

### Task 10: Phase 2 — Split Variable Training

**Files:**
- Create: `gaussianfractallod/train_splits.py`
- Create: `tests/test_train_splits.py`

- [ ] **Step 1: Write split training tests**

File: `tests/test_train_splits.py`

```python
import pytest
import torch

gsplat = pytest.importorskip("gsplat")

from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.train_splits import train_split_level_step


def _make_frozen_roots(device):
    """Roots with no grad (frozen from Phase 1)."""
    return Gaussian(
        means=torch.tensor([[0.0, 0.0, 3.0]], device=device),
        scales=torch.tensor([[-1.0, -1.0, -1.0]], device=device),
        opacities=torch.tensor([[2.0]], device=device),
        sh_coeffs=torch.tensor([[0.5, 0.2, 0.1]], device=device),
    )


def test_split_training_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("gsplat requires CUDA")

    roots = _make_frozen_roots(device)
    tree = SplitTree(num_roots=1, sh_dim=3).to(device)
    tree.add_level()

    optimizer = torch.optim.Adam(tree.level_parameters(0), lr=1e-2)

    gt_image = torch.rand(64, 64, 3, device=device)
    camera = {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor(
            [[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1]], device=device
        ),
        "width": 64,
        "height": 64,
    }

    loss = train_split_level_step(
        roots, tree, target_depth=1,
        gt_image=gt_image, camera=camera,
        optimizer=optimizer, ssim_weight=0.2,
    )
    assert loss.item() > 0
    # Verify gradients flowed to split variables
    assert tree.levels[0].mass_logit.grad is not None
    assert tree.levels[0].position_split.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_splits.py -v`
Expected: FAIL

- [ ] **Step 3: Implement train_splits.py**

File: `gaussianfractallod/train_splits.py`

```python
"""Phase 2: Train split variables level-by-level with frozen parents."""

import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.reconstruct import reconstruct
from gaussianfractallod.render import render_gaussians
from gaussianfractallod.loss import rendering_loss


def train_split_level_step(
    roots: Gaussian,
    tree: SplitTree,
    target_depth: int,
    gt_image: torch.Tensor,
    camera: dict,
    optimizer: torch.optim.Optimizer,
    ssim_weight: float = 0.2,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single training step for split variables at target depth.

    Reconstructs Gaussians down to target_depth, renders, and
    backpropagates through the split-variable chain.

    Args:
        roots: Frozen root Gaussians (no grad).
        tree: Split tree with current level's variables requiring grad.
        target_depth: Binary depth to reconstruct to.
        gt_image: (H, W, 3) ground truth image.
        camera: Camera parameters dict.
        optimizer: Optimizer for current level's split variables.
        ssim_weight: SSIM loss weight.
        background: Background color tensor.

    Returns:
        Scalar loss value.
    """
    optimizer.zero_grad()

    # Reconstruct Gaussians from roots through split tree
    gaussians = reconstruct(roots, tree, target_depth)

    # Render
    rendered = render_gaussians(
        gaussians,
        viewmat=camera["viewmat"],
        K=camera["K"],
        width=camera["width"],
        height=camera["height"],
        background=background,
    )

    # Loss
    loss = rendering_loss(rendered, gt_image, ssim_weight=ssim_weight)
    loss.backward()
    optimizer.step()

    return loss.detach()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_train_splits.py -v`
Expected: PASS (skip without CUDA)

- [ ] **Step 5: Commit**

```bash
git add gaussianfractallod/train_splits.py tests/test_train_splits.py
git commit -m "feat: Phase 2 split-variable training step"
```

---

### Task 11: Pruning

**Files:**
- Create: `gaussianfractallod/prune.py`
- Create: `tests/test_prune.py`

- [ ] **Step 1: Write pruning tests**

File: `tests/test_prune.py`

```python
import torch
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.prune import prune_level


def test_prune_low_mass_children():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()

    # Set mass logits: root 0 has very asymmetric split, root 1 is uniform
    with torch.no_grad():
        # sigmoid(-5) ≈ 0.007 < 0.01 threshold → child A of root 0 pruned
        tree.levels[0].mass_logit[0] = -5.0
        tree.levels[0].mass_logit[1] = 0.0  # uniform

    pruned_count = prune_level(tree, level_idx=0, threshold=0.01)
    occ = tree.get_occupancy(0)
    assert occ[0, 0] == False  # child A of root 0 pruned
    assert occ[0, 1] == True   # child B survives
    assert occ[1, 0] == True   # root 1 both children survive
    assert occ[1, 1] == True
    assert pruned_count == 1


def test_prune_renormalizes_mass_to_survivor():
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()

    with torch.no_grad():
        # sigmoid(-5) ≈ 0.007 → child A pruned, child B survives
        tree.levels[0].mass_logit[0] = -5.0
        tree.levels[0].color_split[0] = torch.tensor([0.5, 0.3, 0.1])

    prune_level(tree, level_idx=0, threshold=0.01)

    occ = tree.get_occupancy(0)
    assert occ[0, 0] == False  # child A pruned
    assert occ[0, 1] == True   # child B survives
    # Dead split vars should be zeroed
    torch.testing.assert_close(
        tree.levels[0].color_split[0], torch.zeros(3)
    )
    # Mass should be renormalized to survivor B: sigmoid(-10) ≈ 0
    assert torch.sigmoid(tree.levels[0].mass_logit[0]).item() < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_prune.py -v`
Expected: FAIL

- [ ] **Step 3: Implement prune.py**

File: `gaussianfractallod/prune.py`

```python
"""Mass-based child pruning: remove children with negligible opacity share.

When one child is pruned, the surviving child effectively becomes the
parent — split variables are zeroed so the derivation produces a child
identical to the parent (mass_logit is set to a large positive/negative
value so sigmoid gives ~1.0 or ~0.0, assigning all mass to the survivor).
"""

import torch
from gaussianfractallod.split_tree import SplitTree


def prune_level(
    tree: SplitTree, level_idx: int, threshold: float = 0.01
) -> int:
    """Prune children whose mass partition falls below threshold.

    Args:
        tree: The split tree.
        level_idx: Which level to prune.
        threshold: Mass partition threshold. Children with pi < threshold
            are pruned.

    Returns:
        Number of children pruned.
    """
    level = tree.levels[level_idx]
    mass_logit = level.mass_logit.detach()
    pi_a = torch.sigmoid(mass_logit)  # (N,)
    pi_b = 1.0 - pi_a

    pruned_count = 0

    with torch.no_grad():
        # Prune child A where pi_a < threshold
        prune_a = pi_a < threshold
        if prune_a.any():
            level.occupancy[prune_a, 0] = False
            pruned_count += prune_a.sum().item()

        # Prune child B where pi_b < threshold
        prune_b = pi_b < threshold
        if prune_b.any():
            level.occupancy[prune_b, 1] = False
            pruned_count += prune_b.sum().item()

        # For nodes with only one surviving child, zero the split vars
        # and set mass_logit so all mass goes to the survivor.
        only_a = level.occupancy[:, 0] & ~level.occupancy[:, 1]
        only_b = ~level.occupancy[:, 0] & level.occupancy[:, 1]
        dead = only_a | only_b | (~level.occupancy[:, 0] & ~level.occupancy[:, 1])

        if dead.any():
            level.position_split[dead] = 0.0
            level.variance_split[dead] = 0.0
            level.color_split[dead] = 0.0

        # Renormalize: push all mass to the surviving child
        if only_a.any():
            level.mass_logit[only_a] = 10.0   # sigmoid(10) ≈ 1.0 → all mass to A
        if only_b.any():
            level.mass_logit[only_b] = -10.0  # sigmoid(-10) ≈ 0.0 → all mass to B

    return pruned_count
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_prune.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add gaussianfractallod/prune.py tests/test_prune.py
git commit -m "feat: mass-based child pruning with dead variable cleanup"
```

---

### Task 12: Full Training Orchestrator

**Files:**
- Create: `gaussianfractallod/train.py`

This ties together Phase 1, Phase 2, checkpointing, and pruning into a single entry point.

- [ ] **Step 1: Implement train.py**

File: `gaussianfractallod/train.py`

```python
"""Full training orchestrator: Phase 1 (roots) + Phase 2 (splits)."""

import torch
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from gaussianfractallod.config import Config
from gaussianfractallod.data import NerfSyntheticDataset
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.train_roots import init_roots, train_roots_step
from gaussianfractallod.train_splits import train_split_level_step
from gaussianfractallod.prune import prune_level
from gaussianfractallod.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


def train(cfg: Config, resume_from: str | None = None) -> tuple[Gaussian, SplitTree]:
    """Run full training pipeline.

    Args:
        cfg: Training configuration.
        resume_from: Path to checkpoint to resume from.

    Returns:
        (roots, tree): Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = NerfSyntheticDataset(cfg.data_dir, split="train", scale=cfg.image_scale)
    logger.info(f"Loaded {len(dataset)} training images")

    sh_degree = cfg.sh_degree
    sh_dim = 3 * ((sh_degree + 1) ** 2)
    background = torch.tensor(cfg.background_color, device=device)

    writer = SummaryWriter(log_dir=str(Path(cfg.checkpoint_dir) / "logs"))

    start_phase = 1
    start_level = 0

    if resume_from:
        roots, tree, meta = load_checkpoint(resume_from, device=device)
        start_phase = meta["phase"]
        start_level = meta.get("level", 0)
        logger.info(f"Resumed from phase {start_phase}, level {start_level}")
    else:
        roots = None
        tree = None

    # ========================
    # Phase 1: Root fitting
    # ========================
    if start_phase <= 1:
        logger.info(f"Phase 1: Fitting {cfg.num_roots} root Gaussians")
        roots = init_roots(cfg.num_roots, sh_degree=sh_degree, device=device)

        optimizer = torch.optim.Adam(
            [roots.means, roots.scales, roots.opacities, roots.sh_coeffs],
            lr=cfg.root_lr,
        )

        best_loss = float("inf")
        plateau_count = 0

        for step in range(cfg.root_iterations):
            idx = step % len(dataset)
            gt_image, camera = dataset[idx]
            gt_image = gt_image.to(device)
            camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera.items()}

            loss = train_roots_step(
                roots, gt_image, camera, optimizer,
                ssim_weight=cfg.ssim_weight, background=background,
            )

            writer.add_scalar("phase1/loss", loss.item(), step)

            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= cfg.root_convergence_window:
                logger.info(f"Phase 1 converged at step {step}, loss={best_loss:.6f}")
                break

            if step % 500 == 0:
                logger.info(f"Phase 1 step {step}: loss={loss.item():.6f}")

        # Freeze roots
        roots = Gaussian(
            means=roots.means.detach(),
            scales=roots.scales.detach(),
            opacities=roots.opacities.detach(),
            sh_coeffs=roots.sh_coeffs.detach(),
        )

        tree = SplitTree(num_roots=cfg.num_roots, sh_dim=sh_dim).to(device)

        save_checkpoint(
            Path(cfg.checkpoint_dir) / "phase1_roots.pt",
            roots, tree, phase=2, level=0,
        )
        logger.info("Phase 1 complete. Roots saved.")

    # ========================
    # Phase 2: Level-by-level split fitting
    # ========================
    for level in range(start_level, cfg.max_binary_depth):
        logger.info(f"Phase 2: Training level {level}")
        # Only add a new level if the tree doesn't already have it
        # (it might if we resumed from a checkpoint)
        if tree.depth <= level:
            tree.add_level()
        tree = tree.to(device)

        optimizer = torch.optim.Adam(
            tree.level_parameters(level), lr=cfg.split_lr,
        )

        target_depth = level + 1
        best_loss = float("inf")
        plateau_count = 0

        for step in range(cfg.split_iterations_per_level):
            idx = step % len(dataset)
            gt_image, camera = dataset[idx]
            gt_image = gt_image.to(device)
            camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera.items()}

            loss = train_split_level_step(
                roots, tree, target_depth,
                gt_image, camera, optimizer,
                ssim_weight=cfg.ssim_weight, background=background,
            )

            writer.add_scalar(f"phase2/level_{level}/loss", loss.item(), step)

            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= cfg.split_convergence_window:
                logger.info(
                    f"Level {level} converged at step {step}, loss={best_loss:.6f}"
                )
                break

            if step % 500 == 0:
                logger.info(f"Level {level} step {step}: loss={loss.item():.6f}")

        # Prune
        pruned = prune_level(tree, level, threshold=cfg.prune_mass_threshold)
        logger.info(f"Level {level}: pruned {pruned} children")

        # Freeze this level's parameters
        for param in tree.level_parameters(level):
            param.requires_grad_(False)

        save_checkpoint(
            Path(cfg.checkpoint_dir) / f"phase2_level_{level}.pt",
            roots, tree, phase=2, level=level + 1,
        )
        logger.info(f"Level {level} complete. Checkpoint saved.")

    writer.close()
    return roots, tree
```

- [ ] **Step 2: Commit**

```bash
git add gaussianfractallod/train.py
git commit -m "feat: full training orchestrator with Phase 1 + Phase 2"
```

---

### Task 13: Evaluation Module

**Files:**
- Create: `gaussianfractallod/eval.py`

- [ ] **Step 1: Implement eval.py**

File: `gaussianfractallod/eval.py`

```python
"""Evaluation: render test views and compute PSNR, SSIM, LPIPS."""

import torch
import logging
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.reconstruct import reconstruct
from gaussianfractallod.render import render_gaussians
from gaussianfractallod.data import NerfSyntheticDataset

logger = logging.getLogger(__name__)


def evaluate(
    roots: Gaussian,
    tree: SplitTree,
    dataset: NerfSyntheticDataset,
    target_depth: int,
    device: torch.device,
    background: torch.Tensor | None = None,
) -> dict:
    """Evaluate model on a dataset split.

    Args:
        roots: Root Gaussians.
        tree: Split tree.
        dataset: Evaluation dataset.
        target_depth: Binary depth to reconstruct.
        device: Torch device.
        background: Background color.

    Returns:
        Dict with 'psnr', 'ssim', 'lpips' (mean values).
    """
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    if background is None:
        background = torch.ones(3, device=device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    with torch.no_grad():
        gaussians = reconstruct(roots, tree, target_depth)
        logger.info(f"Reconstructed {gaussians.num_gaussians} Gaussians at depth {target_depth}")
        for i in range(len(dataset)):
            gt_image, camera = dataset[i]
            gt_image = gt_image.to(device)
            camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera.items()}

            rendered = render_gaussians(
                gaussians,
                viewmat=camera["viewmat"],
                K=camera["K"],
                width=camera["width"],
                height=camera["height"],
                background=background,
            )

            # torchmetrics expects (B, C, H, W)
            pred_4d = rendered.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            gt_4d = gt_image.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

            psnr_values.append(psnr_metric(pred_4d, gt_4d).item())
            ssim_values.append(ssim_metric(pred_4d, gt_4d).item())
            lpips_values.append(lpips_metric(pred_4d, gt_4d).item())

            if i % 50 == 0:
                logger.info(f"Evaluated {i+1}/{len(dataset)} views")

    results = {
        "psnr": sum(psnr_values) / len(psnr_values),
        "ssim": sum(ssim_values) / len(ssim_values),
        "lpips": sum(lpips_values) / len(lpips_values),
        "num_gaussians": gaussians.num_gaussians,
        "target_depth": target_depth,
    }

    logger.info(
        f"Depth {target_depth}: PSNR={results['psnr']:.2f} "
        f"SSIM={results['ssim']:.4f} LPIPS={results['lpips']:.4f} "
        f"({gaussians.num_gaussians} Gaussians)"
    )

    return results
```

- [ ] **Step 2: Commit**

```bash
git add gaussianfractallod/eval.py
git commit -m "feat: evaluation module with PSNR, SSIM, LPIPS"
```

---

## Chunk 3: Colab Integration

### Task 14: Colab Notebook

**Files:**
- Create: `notebooks/train_colab.ipynb`

- [ ] **Step 1: Create the Colab notebook**

File: `notebooks/train_colab.ipynb`

The notebook has 6 cells:

**Cell 1 — Setup:**
```python
# Install dependencies and clone repo
!pip install -q gsplat torchmetrics lpips tensorboard pyyaml

# Clone repo (replace with your repo URL)
import os
if not os.path.exists("GaussianFractalLOD"):
    !git clone https://github.com/dedoubleyou1/GaussianFractalLOD.git
%cd GaussianFractalLOD
!pip install -e . -q

# Mount Google Drive for checkpoints
from google.colab import drive
drive.mount("/content/drive")
```

**Cell 2 — Download data:**
```python
# Download NeRF Synthetic dataset
import os
DATA_DIR = "/content/nerf_synthetic"
if not os.path.exists(DATA_DIR):
    !wget -q https://huggingface.co/datasets/nerfstudio/nerf-synthetic-dataset/resolve/main/nerf_synthetic.zip
    !unzip -q nerf_synthetic.zip -d /content/
    print("Dataset downloaded")
else:
    print("Dataset already exists")
```

**Cell 3 — Configure:**
```python
from gaussianfractallod.config import Config

SCENE = "lego"  # Change for different scenes

cfg = Config(
    data_dir=f"{DATA_DIR}/{SCENE}",
    num_roots=32,
    sh_degree=0,  # Start with SH0 for fast iteration
    max_binary_depth=6,  # Start shallow, increase later
    root_iterations=5000,
    split_iterations_per_level=3000,
    checkpoint_dir=f"/content/drive/MyDrive/GaussianFractalLOD/checkpoints/{SCENE}",
)
print(f"Training {SCENE} with {cfg.num_roots} roots, depth {cfg.max_binary_depth}")
```

**Cell 4 — Train (or resume):**
```python
import logging
logging.basicConfig(level=logging.INFO)

from gaussianfractallod.train import train
from pathlib import Path

# To resume from a checkpoint, set this path:
RESUME_FROM = None  # e.g., f"{cfg.checkpoint_dir}/phase2_level_3.pt"

roots, tree = train(cfg, resume_from=RESUME_FROM)
print(f"Training complete! {tree.num_splits} splits across {tree.depth} levels")
```

**Cell 5 — Evaluate at multiple LoDs:**
```python
import torch
from gaussianfractallod.data import NerfSyntheticDataset
from gaussianfractallod.eval import evaluate

device = torch.device("cuda")
test_dataset = NerfSyntheticDataset(cfg.data_dir, split="test")
background = torch.tensor(cfg.background_color, device=device)

results = {}
for depth in range(tree.depth + 1):
    r = evaluate(roots.to(device), tree.to(device), test_dataset, depth, device, background)
    results[depth] = r
    print(f"Depth {depth}: PSNR={r['psnr']:.2f}, {r['num_gaussians']} Gaussians")
```

**Cell 6 — Visualize LoD progression:**
```python
import matplotlib.pyplot as plt
from gaussianfractallod.reconstruct import reconstruct
from gaussianfractallod.render import render_gaussians

# Pick a test view
gt_image, camera = test_dataset[0]
camera = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in camera.items()}

fig, axes = plt.subplots(1, min(tree.depth + 1, 7), figsize=(24, 4))
depths = list(range(min(tree.depth + 1, 7)))

for ax, depth in zip(axes, depths):
    with torch.no_grad():
        gaussians = reconstruct(roots.to(device), tree.to(device), depth)
        rendered = render_gaussians(
            gaussians, camera["viewmat"], camera["K"],
            camera["width"], camera["height"], background,
        )
    ax.imshow(rendered.cpu().numpy().clip(0, 1))
    ax.set_title(f"Depth {depth}\n{gaussians.num_gaussians} G")
    ax.axis("off")

plt.suptitle(f"LoD Progression — {SCENE}")
plt.tight_layout()
plt.savefig(f"{cfg.checkpoint_dir}/lod_progression.png", dpi=150)
plt.show()
```

- [ ] **Step 2: Verify notebook is valid JSON**

Run: `python -c "import json; json.load(open('notebooks/train_colab.ipynb'))"`
Expected: No error (valid notebook JSON)

- [ ] **Step 3: Commit**

```bash
git add notebooks/train_colab.ipynb
git commit -m "feat: Colab notebook for training and visualization"
```

---

## Summary

| Chunk | Tasks | What it delivers |
|-------|-------|-----------------|
| **1: Foundation** | Tasks 1–7 | Project setup, Gaussian dataclass, R&G derivation with conservation tests, split tree structure, reconstruction, data loading, rendering + loss |
| **2: Training** | Tasks 8–13 | Checkpoints, Phase 1 root fitting, Phase 2 split training, pruning, full orchestrator, evaluation |
| **3: Colab** | Task 14 | Runnable Colab notebook with training, multi-LoD evaluation, and visualization |

After implementing all tasks, run **Experiment 1** (Proof of Concept) from the spec: train Lego scene end-to-end, render at multiple LoD depths, verify smooth degradation.

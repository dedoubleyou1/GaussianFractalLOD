from dataclasses import dataclass, field


@dataclass
class Config:
    # Data
    data_dir: str = ""
    image_scale: float = 1.0

    # Root fitting (Phase 1)
    num_roots: int = 1
    root_lr: float = 1e-3
    root_iterations: int = 10000
    root_convergence_window: int = 1000

    # Level fitting (Phase 2)
    max_levels: int = 6
    level_epochs: int = 60            # passes through all training views per level

    # Per-parameter learning rates (matching 3DGS conventions)
    lr_means: float = 1.6e-4       # position — low, with decay
    lr_means_final: float = 1.6e-6  # position decays 100×
    lr_quats: float = 1e-3          # rotation quaternions
    lr_log_scales: float = 5e-3     # log-space scales
    lr_opacities: float = 2.5e-2    # opacity — high
    lr_sh_dc: float = 2.5e-3         # SH band 0 (DC color)
    lr_sh_rest: float = 1.25e-4       # SH bands 1-3 (view-dependent), 20× lower per 3DGS

    # Adaptive splitting — tiered: gradient determines number of cuts (1→2, 1→4, 1→8)
    split_3cut_threshold: float = 0.004   # 8 children (full octree)
    split_2cut_threshold: float = 0.001   # 4 children
    split_1cut_threshold: float = 0.00025 # 2 children
    split_min_opacity: float = 0.01       # don't subdivide near-transparent Gaussians

    # Child opacity: multiply subdivision-derived opacity by this factor.
    # Forces children to re-earn opacity from a low starting point (like 3DGS reset).
    child_opacity_scale: float = 0.1

    # Regularization
    reg_scale_weight: float = 0.01
    reg_position_weight: float = 0.01
    reg_aspect_weight: float = 0.001  # exp(spread²) wall beyond dead zone
    aspect_dead_zone: float = 2.0  # no aspect penalty up to this ratio beyond init
    max_aspect_ratio: float = 100.0  # hard clamp ceiling
    sh_band_epochs: int = 0           # epochs per SH band activation (0=all bands from start)
    # SH degree
    sh_degree: int = 3

    # Loss
    ssim_weight: float = 0.2

    # Checkpointing
    checkpoint_dir: str = "checkpoints"

    # Rendering
    background_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Reproducibility
    seed: int = 42

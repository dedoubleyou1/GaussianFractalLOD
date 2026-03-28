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
    level_epochs: int = 30            # passes through all training views per level

    # Per-parameter learning rates (matching 3DGS conventions)
    lr_means: float = 1.6e-4       # position — low, with decay
    lr_means_final: float = 1.6e-6  # position decays 100×
    lr_quats: float = 1e-3          # rotation quaternions
    lr_log_scales: float = 5e-3     # log-space scales
    lr_opacities: float = 2.5e-2    # opacity — high
    lr_sh_coeffs: float = 2.5e-3    # SH DC color

    # Adaptive splitting (OR logic: split if EITHER gradient threshold exceeded)
    split_max_threshold: float = 0.003   # any single view needs detail
    split_mean_threshold: float = 0.0005  # consistent gradient across views (coverage)
    split_min_opacity: float = 0.01      # don't subdivide near-transparent Gaussians

    # Child opacity: multiply subdivision-derived opacity by this factor.
    # Forces children to re-earn opacity from a low starting point (like 3DGS reset).
    child_opacity_scale: float = 0.1

    # Regularization
    reg_scale_weight: float = 0.01
    reg_position_weight: float = 0.01
    reg_aspect_weight: float = 0.001  # exp(spread²) wall beyond dead zone
    aspect_dead_zone: float = 2.0  # no aspect penalty up to this ratio beyond init
    max_aspect_ratio: float = 100.0  # hard clamp ceiling

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

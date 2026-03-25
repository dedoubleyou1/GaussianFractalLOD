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

    # Level fitting (Phase 2) — each level has up to 8× more Gaussians
    max_levels: int = 6
    level_base_iterations: int = 500  # doubled each level: 500, 1000, 2000, 4000, 8000, 16000
    level_convergence_window: int = 500

    # Per-parameter learning rates (matching 3DGS conventions)
    lr_means: float = 1.6e-4       # position — low, with decay
    lr_means_final: float = 1.6e-6  # position decays 100×
    lr_quats: float = 1e-3          # rotation quaternions
    lr_log_scales: float = 5e-3     # log-space scales
    lr_opacities: float = 2.5e-2    # opacity — high
    lr_sh_coeffs: float = 2.5e-3    # SH DC color

    # Opacity reset
    opacity_reset_interval: int = 3000  # reset opacity every N steps
    opacity_reset_value: float = -2.2   # inverse_sigmoid(0.1)

    # Adaptive splitting
    split_grad_threshold: float = 0.0002  # only split Gaussians with grad above this

    # Regularization
    reg_scale_weight: float = 0.01
    reg_position_weight: float = 0.01
    reg_aspect_weight: float = 0.1
    max_aspect_ratio: float = 10.0  # hard clamp after each step

    # SH degree
    sh_degree: int = 3

    # Loss
    ssim_weight: float = 0.2

    # Checkpointing
    checkpoint_dir: str = "checkpoints"

    # Rendering
    background_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])

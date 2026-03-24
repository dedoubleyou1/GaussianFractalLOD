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

    # Level fitting (Phase 2) — each level has 8× more Gaussians
    max_levels: int = 6  # 8^6 = 262K max Gaussians from 1 root
    level_lr: float = 5e-3
    level_iterations: int = 5000
    level_convergence_window: int = 500

    # SH degree
    sh_degree: int = 3

    # Pruning
    prune_mass_threshold: float = 0.05

    # Loss
    ssim_weight: float = 0.2

    # Checkpointing
    checkpoint_dir: str = "checkpoints"

    # Rendering
    background_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])

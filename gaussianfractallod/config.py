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

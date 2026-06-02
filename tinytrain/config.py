"""Training configuration with sensible defaults."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainConfig:
    """All training hyperparameters in one place.

    Example::

        cfg = TrainConfig(epochs=10, lr=3e-4, amp=True, grad_accum=4)
    """

    # Optimization
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Memory & Performance
    amp: bool = True
    grad_accum: int = 1
    grad_checkpointing: bool = False

    # Device
    device: Optional[str] = None

    # Logging
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    output_dir: str = "./checkpoints"

    # Scheduler
    scheduler: str = "cosine"
    min_lr_ratio: float = 0.1

    # Data
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Mixed precision scaler
    use_scaler: bool = True

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

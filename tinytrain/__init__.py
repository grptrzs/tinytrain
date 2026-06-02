"""
tinytrain - Dead simple GPU training for PyTorch.
One-line model training on consumer GPUs with automatic mixed precision,
gradient accumulation, and memory-efficient training.
"""

__version__ = "0.2.2"

from .trainer import Trainer
from .config import TrainConfig
from .data import DataLoader, Dataset

__all__ = ["Trainer", "TrainConfig", "DataLoader", "Dataset"]

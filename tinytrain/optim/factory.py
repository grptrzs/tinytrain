"""Optimizer and scheduler factories with sensible defaults."""

import math
from typing import Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from ..config import TrainConfig


def get_optimizer(model, config: TrainConfig) -> optim.Optimizer:
    """Create AdamW with proper weight decay handling."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "layernorm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.999), eps=1e-8)


def get_scheduler(optimizer: optim.Optimizer, config: TrainConfig) -> Optional[object]:
    """Create learning rate scheduler with warmup."""
    if config.scheduler == "constant" and config.warmup_steps == 0:
        return None

    total_steps = config.epochs * 1000
    warmup = config.warmup_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup:
            return current_step / max(1, warmup)

        progress = (current_step - warmup) / max(1, total_steps - warmup)
        progress = min(progress, 1.0)

        if config.scheduler == "cosine":
            return config.min_lr_ratio + 0.5 * (1.0 - config.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        elif config.scheduler == "linear":
            return max(config.min_lr_ratio, 1.0 - progress)
        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda)

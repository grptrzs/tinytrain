"""Minimal example - 3 lines to train on GPU.

Usage:
    python examples/minimal.py
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from tinytrain import Trainer, TrainConfig, DataLoader, Dataset


# Fake data
X = torch.randn(5000, 20)
y = torch.randint(0, 3, (5000,))
train_loader = DataLoader(Dataset(X[:4000], y[:4000]), batch_size=64)
val_loader = DataLoader(Dataset(X[4000:], y[4000:]), batch_size=128, shuffle=False)

# Simple model
model = nn.Sequential(
    nn.Linear(20, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
)

# Train in 3 lines
cfg = TrainConfig(epochs=5, lr=1e-3, amp=True, output_dir="./checkpoints/minimal")
trainer = Trainer(model, cfg)
trainer.train(train_loader, val_loader)

#!/usr/bin/env python3
"""
Minimal MNIST training example using tinytrain.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tinytrain import Trainer, Config

# Simple model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Dummy data (replace with real MNIST)
X_train = torch.randn(1000, 1, 28, 28)
y_train = torch.randint(0, 10, (1000,))
X_val = torch.randn(200, 1, 28, 28)
y_val = torch.randint(0, 10, (200,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

# Train
config = Config({
    'training': {'epochs': 10, 'lr': 0.001, 'mixed_precision': True},
    'checkpointing': {'save_dir': './checkpoints', 'save_every': 5}
})

model = SimpleCNN()
trainer = Trainer(model, config, train_loader, val_loader)
trainer.train()

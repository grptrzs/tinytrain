"""Train a small CNN on CIFAR-10 with tinytrain.

Usage:
    python examples/cifar10.py

Requires: torch, torchvision
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from tinytrain import Trainer, TrainConfig, DataLoader, Dataset


class SmallCNN(nn.Module):
    """Lightweight CNN for CIFAR-10 (~2M params)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean()


def main():
    # Data
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=128, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    # Model & Config
    model = SmallCNN(num_classes=10)
    cfg = TrainConfig(
        epochs=30,
        lr=1e-3,
        scheduler="cosine",
        warmup_steps=100,
        amp=True,
        grad_accum=1,
        output_dir="./checkpoints/cifar10",
        log_every=50,
    )

    # Train
    trainer = Trainer(
        model=model,
        config=cfg,
        loss_fn=nn.CrossEntropyLoss(),
        metrics={"accuracy": accuracy},
    )

    trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    main()

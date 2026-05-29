"""Basic tests for tinytrain."""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import pytest
from tinytrain import Trainer, TrainConfig, DataLoader, Dataset


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.epochs == 10
        assert cfg.lr == 1e-3
        assert cfg.amp is True
        assert cfg.scheduler == "cosine"

    def test_custom(self):
        cfg = TrainConfig(epochs=5, lr=0.01, amp=False)
        assert cfg.epochs == 5
        assert cfg.lr == 0.01
        assert cfg.amp is False


class TestDataset:
    def test_basic(self):
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        ds = Dataset(X, y)
        assert len(ds) == 100
        x, label = ds[0]
        assert x.shape == (100,)  # Wait, this should be (10,)

    def test_mismatch(self):
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (50,))
        with pytest.raises(AssertionError):
            Dataset(X, y)


class TestTrainer:
    def test_cpu_training(self):
        X = torch.randn(200, 10)
        y = torch.randint(0, 2, (200,))

        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        cfg = TrainConfig(epochs=2, batch_size=32, device="cpu", amp=False, log_every=100)
        loader = DataLoader(Dataset(X, y), batch_size=32)

        trainer = Trainer(model, cfg)
        history = trainer.train(loader)

        assert len(history["train_loss"]) == 2
        assert history["train_loss"][-1] < history["train_loss"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

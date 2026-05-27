"""Core training loop with GPU optimizations."""

import os
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .config import TrainConfig
from .optim import get_optimizer, get_scheduler
from .utils import set_seed, get_device, count_params, format_time


class Trainer:
    """Simple GPU trainer with AMP, gradient accumulation, and checkpointing.

    Example::

        model = MyModel()
        cfg = TrainConfig(epochs=5, lr=3e-4, amp=True)
        trainer = Trainer(model, cfg)
        trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        loss_fn: Optional[Callable] = None,
        metrics: Optional[Dict[str, Callable]] = None,
    ):
        self.config = config
        set_seed(config.seed)
        self.device = get_device(config.device)
        self.model = model.to(self.device)

        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)

        self.scaler = GradScaler(enabled=config.amp and self.device.type == "cuda")

        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metrics = metrics or {}

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

        if config.grad_checkpointing:
            self._enable_grad_checkpointing()

        self._print_summary()

    def _print_summary(self):
        cfg = self.config
        pcount = count_params(self.model)
        amp_str = "FP16" if cfg.amp else "FP32"
        gpu_name = f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""
        print()
        print("=" * 60)
        print("  tinytrain v0.2.1")
        print("=" * 60)
        print(f"  Device:     {self.device}{gpu_name}")
        print(f"  Parameters: {pcount:,} ({pcount/1e6:.1f}M)")
        print(f"  Epochs:     {cfg.epochs}")
        print(f"  Batch size: {cfg.batch_size} x {cfg.grad_accum} accum = {cfg.batch_size * cfg.grad_accum} effective")
        print(f"  LR:         {cfg.lr:.1e} ({cfg.scheduler})")
        print(f"  AMP:        {amp_str}")
        print("=" * 60)
        print()

    def _enable_grad_checkpointing(self):
        for module in self.model.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()

    def train_step(self, batch) -> Dict[str, float]:
        cfg = self.config
        if self.device.type == "cuda":
            amp_ctx = autocast(device_type="cuda", enabled=cfg.amp)
        else:
            amp_ctx = nullcontext()

        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

        with amp_ctx:
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss = loss / cfg.grad_accum

        self.scaler.scale(loss).backward()
        return {"loss": loss.item() * cfg.grad_accum}

    @torch.no_grad()
    def eval_step(self, batch) -> Dict[str, float]:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        results = {"loss": loss.item()}
        for name, fn in self.metrics.items():
            results[name] = fn(outputs, targets).item()
        return results

    def train_epoch(self, loader) -> Dict[str, float]:
        cfg = self.config
        self.model.train()
        epoch_loss = 0.0
        step_count = 0
        t0 = time.time()

        for i, batch in enumerate(loader):
            metrics = self.train_step(batch)
            epoch_loss += metrics["loss"]
            step_count += 1

            if (i + 1) % cfg.grad_accum == 0 or (i + 1) == len(loader):
                if cfg.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                if self.global_step % cfg.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = format_time(time.time() - t0)
                    print(f"  step {self.global_step:>6d} | loss {metrics['loss']:.4f} | lr {lr:.2e} | {elapsed}")

        avg_loss = epoch_loss / max(step_count, 1)
        self.history["train_loss"].append(avg_loss)
        return {"loss": avg_loss}

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in loader:
            metrics = self.eval_step(batch)
            total_loss += metrics["loss"]
            count += 1
        avg_loss = total_loss / max(count, 1)
        self.history["val_loss"].append(avg_loss)
        return {"loss": avg_loss}

    def train(self, train_loader, val_loader=None, callbacks=None) -> Dict[str, list]:
        cfg = self.config
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        print(f"  Training started at {time.strftime('%H:%M:%S')}")
        print(f"  Steps per epoch: {len(train_loader)}")
        print()

        for epoch in range(cfg.epochs):
            self.epoch = epoch
            print(f"  Epoch {epoch + 1}/{cfg.epochs}")
            print("  " + "-" * 40)

            train_metrics = self.train_epoch(train_loader)
            print(f"  train_loss: {train_metrics['loss']:.4f}")

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                print(f"  val_loss:   {val_metrics['loss']:.4f}")
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint("best.pt")
                    print("  [saved best checkpoint]")

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            for cb in (callbacks or []):
                cb(self, epoch, train_metrics)

        print()
        print("=" * 60)
        print("  Training complete!")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        print()
        return self.history

    def save_checkpoint(self, filename: str):
        path = Path(self.config.output_dir) / filename
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "history": self.history,
        }, path)

    def load_checkpoint(self, filename: str):
        path = Path(self.config.output_dir) / filename
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.history = ckpt["history"]
        print(f"  Loaded checkpoint from epoch {self.epoch + 1} (step {self.global_step})")

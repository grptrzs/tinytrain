# tinytrain

Train PyTorch models without writing training loops.

```bash
pip install tinytrain
```

```python
from tinytrain import Trainer

trainer = Trainer(model, epochs=10, lr=3e-4, amp=True)
trainer.train(train_loader, val_loader)
```

That's the whole API. It handles:

- Mixed precision (FP16)
- Gradient accumulation
- Cosine LR schedule with warmup
- Checkpoint save/load
- Training logs

## Why

I was copy-pasting the same training loop iinto every project. Same AMP setup, same gradient accumulation logic, same checkpoint code. Got tired of it.

## Consumer GPU stuff

Built this on an RTX 3060 12GB. Everything is designed for limited VRAM:

- Gradient checkpointing to fit bigger models
- Automatic batch size scaling
- Memory-efficient optimizers

## Config

No YAML. Just kwargs:

```python
trainer = Trainer(
    model,
    epochs=50,
    lr=1e-4,
    amp=True,
    grad_accum=4,
    grad_ckpt=True,
    save_best=True,
    log_dir="./logs",
)
```

Or use `TrainConfig` if you prefer:

```python
from tinytrain import TrainConfig
cfg = TrainConfig(epochs=50, lr=1e-4, amp=True)
trainer = Trainer(model, cfg)
```

## Examples

See `examples/` for:
- Image classification (CIFAR-10)
- Text classification (AG News)
- Fine-tuning a small LLM

## Status

Works. I use it for my own projects. API might change if I find something better.

MIT License.


## Troubleshooting
**Q: Getting OOM errors?**
A: Reduce batch size or enable gradient checkpointing.

## Hardware Tested
- AMD RX 7800 XT (RDNA3)
- AMD RX 7900 XTX (RDNA3)
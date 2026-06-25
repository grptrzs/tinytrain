# tinytrain

tinytrain is a small PyTorch training wrapper I use when I want to test an idea quickly without rebuilding the same loop, logging, checkpointing, and config handling every time.

## Why this exists

Every time I wanted to test a new model or dataset, I ended up copy-pasting the same training boilerplate: data loading, optimizer setup, training loop, validation, checkpointing, logging. It's not hard, but it's tedious and error-prone.

tinytrain is my attempt to have a minimal, opinionated training wrapper that handles the boring parts so I can focus on the model.

## What it does

- **Config-driven**: YAML config for hyperparameters, data paths, model settings
- **Training loop**: Standard PyTorch loop with mixed precision support
- **Checkpointing**: Save/load model state, optimizer state, epoch number
- **Logging**: Console output + optional TensorBoard
- **Early stopping**: Stop when validation loss plateaus

## What it doesn't do

- No distributed training (single GPU only)
- No hyperparameter search
- No model architecture definitions (bring your own model)
- No data preprocessing (bring your own Dataset)

## Quick start

```bash
pip install -r requirements.txt
python train.py --config configs/mnist.yaml
```

## Example config

```yaml
model:
  name: SimpleCNN
  params:
    num_classes: 10

training:
  epochs: 20
  batch_size: 64
  lr: 0.001
  mixed_precision: true

data:
  train_path: ./data/mnist/train
  val_path: ./data/mnist/val

checkpointing:
  save_dir: ./checkpoints
  save_every: 5
```

## Design philosophy

- **Minimal**: Do less, but do it well
- **Explicit**: No magic — you can see exactly what the training loop does
- **Composable**: Easy to extend with custom callbacks or hooks

See `docs/why_tinytrain.md` for more context.


## Hardware Tested
- AMD RX 7800 XT (RDNA3)
- AMD RX 7900 XTX (RDNA3)

## Troubleshooting
**Q: Getting OOM errors?**
A: Reduce batch size or enable gradient checkpointing.

## Recent Updates
- Performance improvements for batch processing
- Better error messages for common issues
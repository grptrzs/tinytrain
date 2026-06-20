# Mixed Precision Training Notes

## Why mixed precision?

- ~2x faster on modern GPUs (tensor cores)
- ~50% less memory usage
- Minimal accuracy loss in practice

## How it works in tinytrain

```python
# In the training loop:
scaler = torch.cuda.amp.GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Gotchas

### Loss scaling
The scaler automatically adjusts the loss scale. If you see `inf` or `nan` in loss, the scaler will reduce the scale factor. This is normal — it self-corrects.

### Batch norm
Batch norm should run in FP32 even with mixed precision. tinytrain handles this automatically.

### Model outputs
Some model outputs (like softmax logits) should be FP32 for numerical stability. Use `torch.cuda.amp.autocast()` around the loss calculation but keep the final output in FP32.

### ROCm notes
Mixed precision works on ROCm but:
- Some ops don't have FP16 kernels yet
- Performance gains may be smaller than CUDA
- Test with `torch.cuda.amp.autocast(enabled=True)` first

## Performance comparison

| Config | Time/epoch | Memory | Accuracy |
|--------|-----------|--------|----------|
| FP32 | 45s | 2.1GB | 98.2% |
| Mixed | 24s | 1.2GB | 98.1% |

Tested on MNIST with SimpleCNN, batch_size=64.

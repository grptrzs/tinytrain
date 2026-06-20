# Why tinytrain?

## The problem

Every ML project starts the same way:

1. Write model
2. Write data loader
3. Write training loop
4. Realize you forgot checkpointing
5. Add checkpointing
6. Realize you need validation
7. Add validation loop
8. Realize you need logging
9. Add TensorBoard
10. Realize you need early stopping
11. Add early stopping
12. Finally start training

Steps 3-11 are boilerplate. They're the same in every project, but you always end up rewriting them slightly differently.

## Existing solutions

- **PyTorch Lightning**: Full-featured but opinionated and heavy. Overkill for quick experiments.
- **Hugging Face Trainer**: Great for transformers, but tied to the HF ecosystem.
- **timm**: Excellent for vision models, but not general-purpose.

## What tinytrain is

A middle ground: more than raw PyTorch, less than Lightning. Just the training loop, checkpointing, and config handling. No abstractions for models, data, or metrics.

## Design principles

1. **No magic**: The training loop is a simple for-loop you can read
2. **No inheritance**: You don't subclass anything. Just pass your model, loaders, and config.
3. **Config-driven**: Hyperparameters in YAML, not hardcoded
4. **Single GPU**: No distributed complexity
5. **Easy to debug**: When something breaks, you can step through the code

## When to use tinytrain

- Quick model experiments (not production training)
- Testing ideas before committing to a full framework
- Teaching/learning PyTorch training loops

## When NOT to use tinytrain

- Multi-GPU / distributed training
- Large-scale training (> 1M steps)
- Complex pipelines with custom callbacks

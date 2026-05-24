# Contributing

Sure, open a PR.

## Rules

- Keep it simple. This project exists because training loops are annoying boilerplate.
- Don't add config file support. The API is kwargs-only by design.
- Test on consumer GPUs (RTX 3060/4060 class). If it only works on A100s, it doesn't belong here.

## Setup

```bash
pip install -e ".[dev]"
pytest test/
```

That's it.

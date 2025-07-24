# MuonClip Examples

Transformer training examples with MuonClip.

## Files

- `example_pytorch.py` - PyTorch small GPT training
- `example_jax.py` - JAX/Flax small GPT training

## Features

- Trains two identical models (one with MuonClip, one with AdamW) on the tiny_shakespeare dataset
- Tracks attention logits for QK-Clip
- Plots training metrics

## Run

```bash
python examples/example_pytorch.py
python examples/example_jax.py
```
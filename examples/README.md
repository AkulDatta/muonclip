# MuonClip Examples

Transformer training examples with MuonClip.

## Files

- `example_pytorch.py` - PyTorch small GPT training
- `example_jax.py` - JAX/Flax small GPT training

## Features

- Trains on tiny_shakespeare dataset
- Compares MuonClip vs AdamW
- Tracks attention logits for QK-Clip
- Plots training metrics

## Run

```bash
python examples/example_pytorch.py
python examples/example_jax.py
```
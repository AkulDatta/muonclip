# MuonClip

PyTorch and JAX implementation of MuonClip optimizer from the [Kimi K2 Technical Report](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf).

MuonClip is an optimizer that combines:
- Muon momentum-based updates with Newton-Schulz orthogonalization
- Consistent RMS scaling for stability
- Per-head QK-Clip mechanism to prevent attention logit explosion
- Weight decay for regularization

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### PyTorch
```python
from src.muon_clip_pytorch import MuonClip

# Create optimizer
optimizer = MuonClip(
    model.parameters(), 
    lr=2e-4, 
    momentum=0.95, 
    weight_decay=0.1, 
    tau=100.0
)
optimizer.set_model(model)  # Required for QK-Clip

# Training step
loss.backward()
optimizer.step()
```

### JAX
```python
from src.muon_clip_jax import muonclip

# Create optimizer
optimizer = muonclip(learning_rate=2e-4, momentum=0.95, weight_decay=0.1)

# In training loop, QK-Clip is applied separately after gradient update
state = state.apply_gradients(grads=grads)
# Then apply QK-Clip to attention weights based on max_logits
```

## Examples

See `examples/` for complete training examples comparing MuonClip with AdamW on a GPT model:

```bash
# PyTorch example
python examples/example_pytorch.py

# JAX example  
python examples/example_jax.py
```

Both examples train two identical transformer models (one with MuonClip, one with AdamW) on the tiny_shakespeare dataset.

## Tests

```bash
pytest
```

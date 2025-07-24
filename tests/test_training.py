import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
import pytest

# Imports from inside functions
from muon_clip_pytorch import MuonClip, create_optimizer_separate_groups

# Optional JAX imports
jax = pytest.importorskip("jax", reason="JAX not available")
jnp = pytest.importorskip("jax.numpy", reason="JAX not available")
from jax import random
optax = pytest.importorskip("optax", reason="Optax not available")
flax = pytest.importorskip("flax", reason="Flax not available")
from flax.training import train_state
from muon_clip_jax import muonclip

def test_pytorch_simple_training():
    """Test PyTorch implementation can perform basic training steps"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    
    # Create simple model
    model = nn.Linear(32, 16, bias=False)
    initial_weight = model.weight.data.clone()
    
    # Create optimizer
    optimizer = MuonClip([model.weight], lr=1e-3, momentum=0.95, weight_decay=0.01)
    
    # Create data
    x = torch.randn(8, 32)
    y = torch.randn(8, 16)
    
    # Perform several training steps
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    weight_change = (model.weight.data - initial_weight).norm().item()
    
    # Assertions
    assert all(l > 0 for l in losses), "All losses should be positive"
    assert weight_change > 0, "Weights should change during training"
    assert len(losses) == 5, "Should complete 5 training steps"

def test_jax_basic_functionality():
    """Test JAX implementation basic functionality"""
    
    # Create simple parameters
    key = random.PRNGKey(42)
    params = {'weight': random.normal(key, (32, 16)) * 0.02}
    
    # Create optimizer
    tx = muonclip(learning_rate=1e-3, momentum=0.95, weight_decay=0.01)
    state = train_state.TrainState.create(
        apply_fn=lambda params, x: x @ params['weight'],
        params=params,
        tx=tx
    )
    
    # Create data
    x = jnp.array(np.random.randn(8, 32).astype(np.float32))
    y = jnp.array(np.random.randn(8, 16).astype(np.float32))
    
    # Define loss function
    def loss_fn(params):
        pred = x @ params['weight']
        return jnp.mean((pred - y) ** 2)
    
    initial_params = state.params['weight'].copy()
    
    # Perform training steps
    losses = []
    for _ in range(5):
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        losses.append(float(loss))
    
    param_change = jnp.linalg.norm(state.params['weight'] - initial_params)
    
    # Assertions
    assert all(l > 0 for l in losses), "All losses should be positive"
    assert float(param_change) > 0, "Parameters should change during training"
    assert len(losses) == 5, "Should complete 5 training steps"

def test_optimizer_parameter_consistency():
    """Test that optimizers handle the same parameter types consistently"""
    torch.manual_seed(42)
    
    # Test different parameter shapes
    test_shapes = [(64,), (32, 64), (16, 32, 64)]
    
    
    for shape in test_shapes:
        # Create parameter
        param = torch.randn(shape, requires_grad=True)
        initial_param = param.data.clone()
        
        # Create optimizer
        optimizer = MuonClip([param], lr=1e-3, momentum=0.95, weight_decay=0.01)
        
        # Create gradient
        grad = torch.randn_like(param)
        param.grad = grad
        
        # Optimizer step
        optimizer.step()
        
        # Check parameter changed
        param_change = (param.data - initial_param).norm().item()
        
        # Assertion
        assert param_change > 0, f"Parameter of shape {shape} should change after optimizer step"

def test_attention_tracking_integration():
    """Test that attention tracking works with optimizer integration"""
    torch.manual_seed(42)
    
    # Add examples to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
    
    # Create simple attention-like model
    class SimpleAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(32, 32, bias=False)
            self.key = nn.Linear(32, 32, bias=False)
            self.value = nn.Linear(32, 32, bias=False)
            self.max_logits = []
        
        def forward(self, x):
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)
            
            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / 8.0
            
            # Track max logit
            if self.training:
                self.max_logits.append(scores.max().item())
            
            # Apply attention
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, V)
        
        def get_attention_layers(self):
            return [("attention", self)]
    
    model = SimpleAttentionModel()
    
    # Create optimizers with attention tracking
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    muon_opt = MuonClip(muon_params, lr=1e-3, momentum=0.95, weight_decay=0.01)
    muon_opt.set_model(model)
    
    # Create data and perform training step
    x = torch.randn(4, 8, 32)
    
    model.train()
    muon_opt.zero_grad()
    
    output = model(x)
    loss = output.mean()
    loss.backward()
    
    # Check that max logits were tracked
    assert len(model.max_logits) > 0, "Should track attention logits"
    
    # Perform optimizer step (should handle QK-Clip)
    muon_opt.step()
    
    # Should complete without error
    assert True, "Training step with attention tracking should complete successfully"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
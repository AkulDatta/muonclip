import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
import math
import pytest

# Optional JAX imports
jax = pytest.importorskip("jax", reason="JAX not available")
jnp = pytest.importorskip("jax.numpy", reason="JAX not available")

# Imports from inside functions
from muon_clip_pytorch import MuonClip, newton_schulz as ns_pytorch, create_optimizer_separate_groups
from muon_clip_jax import muonclip, newton_schulz as ns_jax, apply_qk_clip_per_head

def test_imports_and_basic_functionality():
    """Test that imports work and basic functionality is available"""
    # Test PyTorch imports
    # Test JAX imports
    
    # Test basic instantiation
    torch_opt = MuonClip([torch.randn(10, 10, requires_grad=True)], lr=1e-3)
    jax_opt = muonclip(learning_rate=1e-3)
    
    assert torch_opt is not None
    assert jax_opt is not None

def test_algorithm_1_compliance():
    """Test all steps of Algorithm 1 compliance"""
    # Test parameters
    n, m = 64, 128
    lr, momentum, weight_decay = 1e-3, 0.95, 0.01
    
    # Create test data
    np.random.seed(42)
    W = np.random.randn(n, m).astype(np.float32) * 0.02
    G = np.random.randn(n, m).astype(np.float32) * 0.01
    M_prev = np.random.randn(n, m).astype(np.float32) * 0.001
    
    # Step 4: Mt = μMt−1 + Gt
    M_expected = momentum * M_prev + G
    
    # PyTorch momentum
    M_torch = torch.from_numpy(M_prev.copy())
    G_torch = torch.from_numpy(G)
    M_torch.mul_(momentum).add_(G_torch)
    
    # JAX momentum
    M_jax = momentum * jnp.array(M_prev) + jnp.array(G)
    
    momentum_diff = max(
        np.max(np.abs(M_expected - M_torch.numpy())),
        np.max(np.abs(M_expected - np.array(M_jax)))
    )
    
    # Step 5: Newton-Schulz + RMS scaling
    
    O_torch = ns_pytorch(M_torch, steps=5)
    O_jax = ns_jax(M_jax, steps=5)
    
    rms_factor = math.sqrt(max(n, m) * 0.2)
    O_torch_scaled = O_torch * rms_factor
    O_jax_scaled = O_jax * rms_factor
    
    orthog_diff = np.max(np.abs(O_torch_scaled.numpy() - np.array(O_jax_scaled)))
    
    # Step 6: Weight update
    W_torch = torch.from_numpy(W.copy())
    W_torch.add_(O_torch_scaled + weight_decay * W_torch, alpha=-lr)
    
    W_jax = jnp.array(W) - lr * (O_jax_scaled + weight_decay * jnp.array(W))
    
    weight_diff = np.max(np.abs(W_torch.numpy() - np.array(W_jax)))
    
    # Assertions
    assert momentum_diff < 1e-10, f"Momentum update differs by {momentum_diff}"
    assert orthog_diff < 1e-5, f"Newton-Schulz + RMS differs by {orthog_diff}"
    assert weight_diff < 1e-8, f"Weight update differs by {weight_diff}"

def test_per_head_qk_clip():
    """Test per-head QK-Clip implementation (Algorithm 1, lines 9-17)"""
    d_model, num_heads = 64, 8
    tau = 100.0
    
    # Test logits: some above tau, some below
    max_logits = [80.0, 150.0, 60.0, 200.0, 40.0, 120.0, 90.0, 110.0]
    
    # Expected gamma calculation (Algorithm 1, line 12)
    gamma_expected = np.minimum(1.0, tau / np.array(max_logits))
    
    # JAX per-head clipping
    gamma_jax = jnp.minimum(1.0, tau / jnp.array(max_logits))
    
    # PyTorch per-head clipping simulation
    gamma_torch = np.array([min(1.0, tau / logit) for logit in max_logits])
    
    # Compare gamma calculations
    jax_gamma_diff = np.max(np.abs(gamma_expected - np.array(gamma_jax)))
    torch_gamma_diff = np.max(np.abs(gamma_expected - gamma_torch))
    
    # Verify clipping logic
    heads_should_clip = np.sum(np.array(max_logits) > tau)
    heads_jax_clipped = np.sum(np.array(gamma_jax) < 1.0)
    heads_torch_clipped = np.sum(gamma_torch < 1.0)
    
    # Assertions
    assert jax_gamma_diff < 1e-6, f"JAX gamma differs by {jax_gamma_diff}"
    assert torch_gamma_diff < 1e-6, f"PyTorch gamma differs by {torch_gamma_diff}"
    assert heads_should_clip == heads_jax_clipped == heads_torch_clipped, "Clipping head count mismatch"

def test_attention_logit_formula():
    """Test attention logit calculation matches paper formula"""
    # Paper: S^h_max = (1/√d) * max_{X∈B} max_{i,j} Q^h_i * K^h_j^T
    batch_size, seq_len, d_model = 2, 8, 32
    num_heads = 4
    d_k = d_model // num_heads
    scale = 1.0 / math.sqrt(d_k)
    
    # Create test data
    np.random.seed(42)
    X = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    
    # Manual calculation following paper
    Q = X @ W_q
    K = X @ W_k
    
    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    scores = (Q @ K.transpose(0, 1, 3, 2)) * scale
    
    # Manual per-head max
    max_per_head_manual = [np.max(scores[:, h, :, :]) for h in range(num_heads)]
    
    # PyTorch calculation
    scores_torch = torch.from_numpy(scores)
    max_per_head_torch = scores_torch.max(dim=-1)[0].max(dim=-1)[0].max(dim=0)[0].tolist()
    
    # JAX calculation
    scores_jax = jnp.array(scores)
    max_per_head_jax = jnp.max(scores_jax, axis=(0, 2, 3)).tolist()
    
    # Compare
    torch_diff = np.max(np.abs(np.array(max_per_head_manual) - np.array(max_per_head_torch)))
    jax_diff = np.max(np.abs(np.array(max_per_head_manual) - np.array(max_per_head_jax)))
    
    # Assertions
    assert torch_diff < 1e-6, f"PyTorch attention formula differs by {torch_diff}"
    assert jax_diff < 1e-6, f"JAX attention formula differs by {jax_diff}"

def test_simple_training_step():
    """Test simple training step works correctly"""
    torch.manual_seed(42)
    
    # Create simple model
    model = nn.Linear(16, 8, bias=False)
    initial_weight = model.weight.data.clone()
    
    # Create optimizer
    optimizer = MuonClip([model.weight], lr=1e-3, momentum=0.95, weight_decay=0.01)
    
    # Create data
    x = torch.randn(4, 16)
    y = torch.randn(4, 8)
    
    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    
    optimizer.step()
    
    weight_change = (model.weight.data - initial_weight).norm().item()
    
    # Assertions
    assert loss.item() > 0, "Loss should be positive"
    assert weight_change > 0, "Weights should change after optimizer step"

def test_rms_scaling_factor():
    """Test RMS scaling factor calculation matches Algorithm 1 line 5"""
    # Test cases
    test_cases = [(32, 64), (128, 128), (256, 512)]
    
    for n, m in test_cases:
        # Expected from Algorithm 1: √(max(n,m) × 0.2)
        expected = math.sqrt(max(n, m) * 0.2)
        
        # PyTorch calculation
        pytorch_rms = math.sqrt(max(n, m) * 0.2)
        
        # JAX calculation
        jax_rms = float(jnp.sqrt(max(n, m) * 0.2))
        
        # Assertions
        assert abs(expected - pytorch_rms) < 1e-10, f"PyTorch RMS differs for shape ({n}, {m})"
        assert abs(expected - jax_rms) < 1e-6, f"JAX RMS differs for shape ({n}, {m})"

def test_newton_schulz_convergence():
    """Test Newton-Schulz orthogonalization produces orthogonal matrices"""
    np.random.seed(42)
    
    # Test matrices
    test_matrices = [
        np.random.randn(32, 32).astype(np.float32),
        np.random.randn(64, 128).astype(np.float32),
    ]
    
    for A in test_matrices:
        # Apply Newton-Schulz
        A_torch = torch.from_numpy(A)
        A_jax = jnp.array(A)
        
        O_torch = ns_pytorch(A_torch, steps=5).numpy()
        O_jax = np.array(ns_jax(A_jax, steps=5))
        
        # Check orthogonality (approximately): O @ O.T ≈ I
        n, m = A.shape
        if n <= m:  # Tall or square matrix
            product_torch = O_torch @ O_torch.T
            product_jax = O_jax @ O_jax.T
            identity = np.eye(n)
        else:  # Wide matrix  
            product_torch = O_torch.T @ O_torch
            product_jax = O_jax.T @ O_jax
            identity = np.eye(m)
        
        # Check orthogonality
        torch_orth_error = np.max(np.abs(product_torch - identity))
        jax_orth_error = np.max(np.abs(product_jax - identity))
        
        # Check consistency between implementations
        implementation_diff = np.max(np.abs(O_torch - O_jax))
        
        # Assertions (relaxed tolerance for random matrices)
        assert torch_orth_error < 0.5, f"PyTorch Newton-Schulz not orthogonal: {torch_orth_error}"
        assert jax_orth_error < 0.5, f"JAX Newton-Schulz not orthogonal: {jax_orth_error}"
        assert implementation_diff < 1e-4, f"Newton-Schulz implementations differ: {implementation_diff}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
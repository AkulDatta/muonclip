"""
MuonClip Optimizer - JAX/Optax Implementation
=============================================

A token-efficient optimizer that combines Muon with QK-Clip for stable LLM training.
Based on Algorithm 1 from the Kimi K2 Technical Report.

Features:
- Newton-Schulz orthogonalization for 2D+ parameters
- QK-Clip mechanism to prevent attention logit explosion
- Standard momentum for 1D parameters
- RMS matching for proper scaling
- Full Optax integration

Author: Based on Kimi K2 Technical Report
License: Educational/Research Use
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, tree_util, lax
import flax.linen as nn
from flax.training import train_state
import optax
from typing import NamedTuple, Tuple, Optional, Dict, Any
import numpy as np


def newton_schulz(G: jnp.ndarray, steps: int = 5, eps: float = 1e-7) -> jnp.ndarray:
    """
    Newton-Schulz iteration for matrix orthogonalization (general version).
    
    This version handles rectangular matrices by transposing when needed.
    Not JIT-compiled due to conditional logic.
    
    Args:
        G: Input matrix
        steps: Number of iteration steps
        eps: Small epsilon for numerical stability
        
    Returns:
        Orthogonalized matrix
    """
    # Coefficients from Muon paper
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # Normalize by Frobenius norm
    X = G / (jnp.linalg.norm(G, 'fro') + eps)
    
    # Handle rectangular matrices by transposing
    if G.shape[0] > G.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    
    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    # Transpose back if needed
    if transposed:
        X = X.T
    
    return X


@jit
def newton_schulz_square(G: jnp.ndarray, steps: int = 5, eps: float = 1e-7) -> jnp.ndarray:
    """
    JIT-compiled Newton-Schulz iteration for square matrices.
    
    Args:
        G: Input square matrix
        steps: Number of iteration steps
        eps: Small epsilon for numerical stability
        
    Returns:
        Orthogonalized matrix
    """
    # Coefficients from Muon paper
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # Normalize by Frobenius norm
    X = G / (jnp.linalg.norm(G, 'fro') + eps)
    
    # Newton-Schulz iterations using lax.fori_loop for JIT compatibility
    def iteration_body(i, X):
        A = X @ X.T
        B = b * A + c * A @ A
        return a * X + B @ X
    
    X = lax.fori_loop(0, steps, iteration_body, X)
    return X


class MuonClipState(NamedTuple):
    momentum: optax.Updates


def muonclip(
    learning_rate: float = 1e-3,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    ns_steps: int = 5,
    eps: float = 1e-7
) -> optax.GradientTransformation:
    """
    MuonClip optimizer for JAX/Optax.
    
    This optimizer applies:
    1. Muon updates with Newton-Schulz orthogonalization for 2D+ parameters
    2. Standard momentum for 1D parameters
    3. Weight decay to all parameters
    
    Note: QK-Clip should be applied separately after the optimizer step
    in your training loop, as it requires access to attention logits.
    
    Args:
        learning_rate: Learning rate (η in paper)
        momentum: Momentum coefficient (μ in paper)
        weight_decay: Weight decay coefficient (λ in paper)
        ns_steps: Newton-Schulz iteration steps
        eps: Numerical stability epsilon
        
    Returns:
        Optax GradientTransformation
    """
    
    def init_fn(params):
        return MuonClipState(
            momentum=tree_util.tree_map(jnp.zeros_like, params)
        )
    
    def update_fn(grads, state, params):
        
        def update_param(grad, momentum_buf, param):
            # Update momentum: Mt = μMt−1 + Gt
            new_momentum = momentum * momentum_buf + grad
            
            # Check parameter dimensionality
            if grad.ndim >= 2:  # 2D+ parameters - use Muon
                # Apply Newton-Schulz orthogonalization
                if grad.shape[0] == grad.shape[1]:  # Square matrix
                    orthogonal_update = newton_schulz_square(new_momentum, ns_steps, eps)
                else:  # Rectangular matrix
                    orthogonal_update = newton_schulz(new_momentum, ns_steps, eps)
                
                # RMS matching factor: √(max(n,m) × 0.2)
                n, m = param.shape[0], param.shape[1]
                rms_factor = jnp.sqrt(max(n, m) * 0.2)
                orthogonal_update = orthogonal_update * rms_factor
                
                # Compute update with weight decay
                param_update = -(orthogonal_update + weight_decay * param) * learning_rate
            else:  # 1D parameters - use standard momentum
                param_update = -(new_momentum + weight_decay * param) * learning_rate
            
            return param_update, new_momentum
        
        # Apply updates to each parameter
        updates = []
        new_momentum_buffers = []
        
        for grad, momentum_buf, param in zip(
            tree_util.tree_leaves(grads),
            tree_util.tree_leaves(state.momentum),
            tree_util.tree_leaves(params)
        ):
            update, new_mom = update_param(grad, momentum_buf, param)
            updates.append(update)
            new_momentum_buffers.append(new_mom)
        
        # Reconstruct trees
        updates = tree_util.tree_unflatten(tree_util.tree_structure(grads), updates)
        new_momentum = tree_util.tree_unflatten(tree_util.tree_structure(state.momentum), new_momentum_buffers)
        
        new_state = MuonClipState(momentum=new_momentum)
        
        return updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)


def apply_qk_clip_per_head(
    query_weights: jnp.ndarray,
    key_weights: jnp.ndarray,
    max_logits_per_head: jnp.ndarray,
    tau: float = 100.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply per-head QK-Clip following Algorithm 1, lines 11-16.
    
    Args:
        query_weights: [d_model, d_model] Query projection weights
        key_weights: [d_model, d_model] Key projection weights  
        max_logits_per_head: [num_heads,] Max logits per head
        tau: Threshold for clipping
        
    Returns:
        Clipped query and key weights
    """
    d_model = query_weights.shape[0]
    num_heads = len(max_logits_per_head)
    d_k = d_model // num_heads
    
    # Compute per-head scaling factors (Algorithm 1, line 12: γ ← τ/S^h_max)
    gamma_per_head = jnp.minimum(1.0, tau / max_logits_per_head)
    sqrt_gamma_per_head = jnp.sqrt(gamma_per_head)
    
    # Only apply scaling where threshold is exceeded (Algorithm 1, line 11)
    scaling_factors = jnp.where(max_logits_per_head > tau, sqrt_gamma_per_head, 1.0)
    
    # Reshape scaling factors to match weight dimensions [d_model,]
    # Each head gets d_k consecutive dimensions
    scaling_matrix = jnp.repeat(scaling_factors, d_k)  # [d_model,]
    
    # Apply √γ scaling to query and key weights (Algorithm 1, lines 13-14)
    clipped_query = query_weights * scaling_matrix[None, :]  # Broadcast over first dim
    clipped_key = key_weights * scaling_matrix[None, :]      # Broadcast over first dim
    
    return clipped_query, clipped_key


def apply_qk_clip_to_model(
    params: Dict[str, Any],
    attention_logits: Dict[str, jnp.ndarray],  # Now expects per-head arrays
    tau: float = 100.0
) -> Dict[str, Any]:
    """
    Apply per-head QK-Clip to model parameters.
    
    Args:
        params: Model parameters
        attention_logits: Dict mapping layer names to per-head max logits [num_heads,]
        tau: Clipping threshold
        
    Returns:
        Updated model parameters with QK-Clip applied
    """
    updated_params = params.copy()
    
    for layer_name, max_logits_per_head in attention_logits.items():
        if layer_name in params and jnp.any(max_logits_per_head > tau):
            layer_params = params[layer_name]
            
            # Apply QK-Clip if query and key weights exist
            if 'query' in layer_params and 'key' in layer_params:
                query_w = layer_params['query']['kernel']
                key_w = layer_params['key']['kernel']
                
                # Apply per-head clipping (Algorithm 1, lines 11-16)
                clipped_q, clipped_k = apply_qk_clip_per_head(
                    query_w, key_w, max_logits_per_head, tau
                )
                
                # Update parameters
                updated_params[layer_name]['query']['kernel'] = clipped_q
                updated_params[layer_name]['key']['kernel'] = clipped_k
    
    return updated_params


# Training utilities
class TrainState(train_state.TrainState):
    """Extended train state to track per-head attention logits."""
    attention_logits: Dict[str, jnp.ndarray] = None  # Now stores per-head arrays


def create_optimizer_separate_groups(
    rng: jax.random.PRNGKey,
    model: nn.Module,
    learning_rate: float = 1e-3,
    momentum: float = 0.95,
    weight_decay: float = 0.01
) -> TrainState:
    """
    Create training state with MuonClip optimizer.
    
    Args:
        rng: Random number generator key
        model: Flax model
        learning_rate: Learning rate
        momentum: Momentum coefficient
        weight_decay: Weight decay
        
    Returns:
        TrainState with MuonClip optimizer
    """
    # Initialize model
    dummy_input = jnp.ones((1, 32), dtype=jnp.int32)
    variables = model.init(rng, dummy_input, deterministic=True)
    
    # Create optimizer
    tx = muonclip(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        attention_logits={}
    )


def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    tau: float = 100.0
) -> Tuple[TrainState, float]:
    """
    Single training step with MuonClip and QK-Clip.
    
    Args:
        state: Current training state
        batch: Training batch with 'inputs' and 'targets'
        rng: Random key for dropout
        tau: QK-Clip threshold
        
    Returns:
        Tuple of (updated_state, loss)
    """
    
    def loss_fn(params):
        logits, attention_logits = state.apply_fn(
            {'params': params},
            batch['inputs'],
            deterministic=False,
            rngs={'dropout': rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            batch['targets'].reshape(-1)
        ).mean()
        return loss, attention_logits
    
    # Compute gradients and attention logits
    (loss, attention_logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Apply optimizer update
    state = state.apply_gradients(grads=grads)
    
    # Apply QK-Clip
    clipped_params = apply_qk_clip_to_model(state.params, attention_logits, tau)
    
    # Update state with clipped parameters and attention logits
    state = state.replace(params=clipped_params, attention_logits=attention_logits)
    
    return state, loss
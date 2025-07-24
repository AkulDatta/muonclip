import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List
import time
from functools import partial
from datasets import load_dataset
import requests
import tiktoken
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from muon_clip_jax import muonclip, apply_qk_clip_to_model, TrainState


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic=False):
        # Multi-head attention
        attn_out, max_logit_per_head = MultiHeadAttentionWithTracking(
            self.num_heads,
            self.d_model,
            self.dropout_rate
        )(x, mask, deterministic)
        x = nn.LayerNorm()(x + attn_out)
        
        # Feed-forward network
        ffn_out = nn.Sequential([
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dropout(self.dropout_rate, deterministic=deterministic),
            nn.Dense(self.d_model),
            nn.Dropout(self.dropout_rate, deterministic=deterministic)
        ])(x)
        x = nn.LayerNorm()(x + ffn_out)
        
        return x, max_logit_per_head


class MultiHeadAttentionWithTracking(nn.Module):
    num_heads: int
    d_model: int
    dropout_rate: float = 0.1
    
    def setup(self):
        self.d_k = self.d_model // self.num_heads
        self.scale = 1.0 / jnp.sqrt(self.d_k)
        
        self.query = nn.Dense(self.d_model, use_bias=False)
        self.key = nn.Dense(self.d_model, use_bias=False)
        self.value = nn.Dense(self.d_model, use_bias=False)
        self.output = nn.Dense(self.d_model, use_bias=False)
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def __call__(self, x, mask=None, deterministic=False):
        batch_size, seq_len = x.shape[:2]
        
        # Linear projections
        Q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 2, 1, 3)  # [batch, heads, seq, d_k]
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        # Track maximum logit per head (Algorithm 1, line 10)
        # scores shape: [batch, heads, seq, seq]
        max_logit_per_head = jnp.max(scores, axis=(0, 2, 3))  # [heads,]
        max_logit = jnp.max(max_logit_per_head)  # Overall max for backward compatibility
        
        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask[None, None, :, :], scores, float('-inf'))
        
        # Apply softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)
        
        # Apply attention to values
        context = jnp.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = self.output(context)
        
        return output, max_logit_per_head


class GPTModel(nn.Module):
    """Small GPT model"""
    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    max_seq_len: int
    dropout_rate: float = 0.1
    
    def setup(self):
        self.token_embedding = nn.Embed(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embed(self.max_seq_len, self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Use shared module for transformer blocks
        self.blocks = [
            TransformerBlock(self.d_model, self.num_heads, self.d_model * 4, self.dropout_rate)
            for _ in range(self.num_layers)
        ]
        
        self.ln_f = nn.LayerNorm()
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)
    
    def __call__(self, input_ids, deterministic=False):
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = jnp.arange(seq_len)[None, :]
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb, deterministic=deterministic)
        
        # Create causal mask
        mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1).astype(bool)
        mask = ~mask
        
        # Track attention logits for each layer and head
        attention_logits = {}
        
        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            x, max_logit_per_head = block(x, mask, deterministic)
            # Store per-head logits for per-head QK-Clip
            attention_logits[f'block_{i}'] = max_logit_per_head
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, attention_logits


def create_optimizer_muon(params, learning_rate=2e-4):
    """Create MuonClip optimizer (returns TrainState for JAX)"""
    tx = muonclip(
        learning_rate=learning_rate,
        momentum=0.95,
        weight_decay=0.1
    )
    
    return TrainState.create(
        apply_fn=None,  # Will be set later with model
        params=params,
        tx=tx,
        attention_logits={}
    )


def create_optimizer_adamw(params, learning_rate=3e-4):
    """Create AdamW optimizer (returns TrainState for JAX)"""
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=0.1)
    
    return train_state.TrainState.create(
        apply_fn=None,  # Will be set later with model
        params=params,
        tx=tx
    )


@partial(jit, static_argnames=['model', 'tau'])
def train_step_muon(model, state: TrainState, inputs: jnp.ndarray, targets: jnp.ndarray,
                    rng: jax.random.PRNGKey, tau: float = 100.0):
    """Training step with MuonClip optimizer"""
    
    def loss_fn(params):
        logits, attention_logits = model.apply(
            {'params': params},
            inputs,
            deterministic=False,
            rngs={'dropout': rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        ).mean()
        return loss, attention_logits
    
    # Compute loss and gradients
    (loss, attention_logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Apply optimizer update
    state = state.apply_gradients(grads=grads)
    
    # Apply QK-Clip to attention layers
    # Extract query/key weights and apply clipping
    clipped_params = state.params.copy()
    
    for block_idx in range(len(attention_logits)):
        block_name = f'blocks_{block_idx}'
        if block_name in state.params and f'block_{block_idx}' in attention_logits:
            max_logit_per_head = attention_logits[f'block_{block_idx}']  # Shape: [num_heads,]
            
            # Apply per-head clipping (Algorithm 1, line 11-17)
            # γ_h = min(1, τ/S^h_max) for each head h
            gamma_per_head = jnp.minimum(1.0, tau / max_logit_per_head)  # [num_heads,]
            sqrt_gamma_per_head = jnp.sqrt(gamma_per_head)  # [num_heads,]
            
            # Access the attention module parameters
            attn_params = state.params[block_name]['MultiHeadAttentionWithTracking_0']
            if 'query' in attn_params and 'key' in attn_params:
                query_weights = attn_params['query']['kernel']  # [d_model, d_model]
                key_weights = attn_params['key']['kernel']     # [d_model, d_model]
                
                # Reshape scaling factors for broadcasting: [num_heads,] -> [d_model, d_model]
                # Each head uses d_model//num_heads features
                d_k = query_weights.shape[1] // len(sqrt_gamma_per_head)
                sqrt_gamma_expanded = jnp.repeat(sqrt_gamma_per_head, d_k, axis=0)[:query_weights.shape[1]]
                sqrt_gamma_matrix = jnp.broadcast_to(sqrt_gamma_expanded[None, :], query_weights.shape)
                
                # Apply per-head scaling
                clipped_params[block_name]['MultiHeadAttentionWithTracking_0']['query']['kernel'] = query_weights * sqrt_gamma_matrix
                clipped_params[block_name]['MultiHeadAttentionWithTracking_0']['key']['kernel'] = key_weights * sqrt_gamma_matrix
    
    # Update state with clipped parameters
    state = state.replace(params=clipped_params, attention_logits=attention_logits)
    
    # Calculate gradient norm for monitoring
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    
    return state, loss, grad_norm, attention_logits


@partial(jit, static_argnames=['model'])
def train_step_adamw(model, state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray,
                     rng: jax.random.PRNGKey):
    """Training step with standard AdamW optimizer"""
    
    def loss_fn(params):
        logits, _ = model.apply(
            {'params': params},
            inputs,
            deterministic=False,
            rngs={'dropout': rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        ).mean()
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Calculate gradient norm
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    
    return state, loss, grad_norm


def load_shakespeare_data(key, batch_size, seq_len, num_batches=100):
    """Load and tokenize the tiny_shakespeare dataset"""
    # Download the dataset directly
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text
    
    # Use tiktoken for BPE tokenization
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize the entire text
    tokens = enc.encode(text)
    tokens = jnp.array(tokens)
    
    # Get vocabulary size
    vocab_size = enc.n_vocab
    
    # Create decode function for debugging
    decode = lambda x: enc.decode(x.tolist() if hasattr(x, 'tolist') else x)
    
    # Create batches
    data = []
    max_start = len(tokens) - seq_len - 1
    
    for i in range(num_batches):
        key, subkey = random.split(key)
        batch_inputs = []
        batch_targets = []
        
        for _ in range(batch_size):
            start_idx = random.randint(subkey, (), 0, max_start)
            # Input sequence
            inputs = tokens[start_idx:start_idx + seq_len]
            # Target is next token prediction
            targets = tokens[start_idx + 1:start_idx + seq_len + 1]
            
            batch_inputs.append(inputs)
            batch_targets.append(targets)
        
        batch_inputs = jnp.stack(batch_inputs)
        batch_targets = jnp.stack(batch_targets)
        data.append({'inputs': batch_inputs, 'targets': batch_targets})
    
    return data, vocab_size, enc, decode


def train_and_compare():
    """Train models with MuonClip and AdamW, comparing performance"""
    
    # Hyperparameters
    d_model = 256
    num_heads = 8
    num_layers = 4
    max_seq_len = 128
    batch_size = 16
    num_epochs = 5
    num_batches = 200
    
    # Initialize
    key = random.PRNGKey(42)
    key, model_key, data_key = random.split(key, 3)
    
    # Load Shakespeare data
    print("\nLoading tiny_shakespeare dataset...")
    train_data, vocab_size, enc, decode = load_shakespeare_data(data_key, batch_size, max_seq_len, num_batches)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Using GPT-2 tokenizer with BPE encoding")
    
    print("Initializing models...")
    # Create two identical models
    model_muon = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    model_adamw = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    
    # Initialize both models with same parameters
    dummy_input = jnp.ones((1, 32), dtype=jnp.int32)
    key_muon, key_adamw = random.split(model_key)
    variables_muon = model_muon.init(key_muon, dummy_input, deterministic=True)
    # Use same params for both models to ensure identical initialization
    variables_adamw = {'params': variables_muon['params'].copy()}
    
    # Create optimizers
    print("Creating optimizers...")
    state_muon = create_optimizer_muon(variables_muon['params'], learning_rate=2e-4)
    state_adamw = create_optimizer_adamw(variables_adamw['params'], learning_rate=3e-4)
    
    
    # Training metrics
    muon_losses = []
    adamw_losses = []
    muon_max_logits = []
    muon_grad_norms = []
    adamw_grad_norms = []
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training loop with progress bar
        epoch_muon_losses = []
        epoch_adamw_losses = []
        
        pbar = tqdm(enumerate(train_data), total=num_batches, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in pbar:
            key, rng_muon, rng_adam = random.split(key, 3)
            
            # Train MuonClip model
            state_muon, loss_muon, grad_norm_muon, attention_logits = train_step_muon(
                model_muon, state_muon, batch['inputs'], batch['targets'], rng_muon
            )
            
            # Train AdamW model
            state_adamw, loss_adamw, grad_norm_adamw = train_step_adamw(
                model_adamw, state_adamw, batch['inputs'], batch['targets'], rng_adam
            )
            
            # Record metrics
            muon_losses.append(float(loss_muon))
            adamw_losses.append(float(loss_adamw))
            epoch_muon_losses.append(float(loss_muon))
            epoch_adamw_losses.append(float(loss_adamw))
            muon_grad_norms.append(float(grad_norm_muon))
            adamw_grad_norms.append(float(grad_norm_adamw))
            
            # Track max attention logit
            if attention_logits:
                # v is now a per-head array, so we need to get the max across all heads
                max_logit = max(float(jnp.max(v)) for v in attention_logits.values())
                muon_max_logits.append(max_logit)
            else:
                muon_max_logits.append(0.0)
            
            # Update progress bar
            pbar.set_postfix({
                'MuonClip Loss': f"{float(loss_muon):.4f}",
                'AdamW Loss': f"{float(loss_adamw):.4f}",
                'Max Logit': f"{muon_max_logits[-1]:.2f}"
            })
        
        # Print epoch summary
        print(f"  Epoch {epoch + 1} Summary - "
              f"Avg MuonClip Loss: {np.mean(epoch_muon_losses):.4f}, "
              f"Avg AdamW Loss: {np.mean(epoch_adamw_losses):.4f}")
    
    print("\nTraining completed!")
    print("=" * 60)
    
    # Plot results
    plot_training_metrics(muon_losses, adamw_losses, muon_max_logits,
                         muon_grad_norms, adamw_grad_norms)
    
    return model_muon, model_adamw


def plot_training_metrics(muon_losses, adamw_losses, max_logits, muon_grads, adamw_grads):
    """Plot training metrics comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss comparison
    ax = axes[0, 0]
    ax.plot(muon_losses, label='MuonClip', alpha=0.8)
    ax.plot(adamw_losses, label='AdamW', alpha=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Max attention logits
    ax = axes[0, 1]
    ax.plot(max_logits, color='orange', alpha=0.8)
    ax.axhline(y=100, color='red', linestyle='--', label='QK-Clip Threshold (τ=100)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Max Attention Logit')
    ax.set_title('Maximum Attention Logits (MuonClip)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gradient norms
    ax = axes[1, 0]
    ax.plot(muon_grads, label='MuonClip', alpha=0.8)
    ax.plot(adamw_grads, label='AdamW', alpha=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss ratio
    ax = axes[1, 1]
    loss_ratio = np.array(muon_losses) / (np.array(adamw_losses) + 1e-8)
    ax.plot(loss_ratio, color='green', alpha=0.8)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss Ratio (MuonClip / AdamW)')
    ax.set_title('Relative Performance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('muonclip_jax_comparison.png', dpi=150)
    print("\nTraining metrics saved to 'muonclip_jax_comparison.png'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Final MuonClip Loss: {muon_losses[-1]:.4f}")
    print(f"  Final AdamW Loss: {adamw_losses[-1]:.4f}")
    print(f"  Max Attention Logit (peak): {max(max_logits):.2f}")
    print(f"  Max Attention Logit (final): {max_logits[-1]:.2f}")
    print(f"  Avg Gradient Norm - MuonClip: {np.mean(muon_grads):.4f}")
    print(f"  Avg Gradient Norm - AdamW: {np.mean(adamw_grads):.4f}")


if __name__ == "__main__":
    print("MuonClip JAX Example - Transformer Training")
    print("=" * 60)
    
    # Run training comparison
    muon_state, adamw_state = train_and_compare()
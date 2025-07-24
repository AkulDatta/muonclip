import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import math
import time
from datasets import load_dataset
from transformers import AutoTokenizer
import requests
import tiktoken
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from muon_clip_pytorch import MuonClip, create_optimizer_separate_groups


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and FFN"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttentionWithTracking(d_model, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual  
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class MultiHeadAttentionWithTracking(nn.Module):
    """Multi-head attention with max logit tracking for QK-Clip"""
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.dropout_rate = dropout_rate
        
        # Linear projections
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # For tracking max logits
        self.max_logits = []
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        
        # Linear projections and reshape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, seq, d_k]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Track maximum logit per head (Algorithm 1, line 10)
        if self.training:
            # scores shape: [batch, heads, seq, seq]
            max_logit_per_head = scores.max(dim=-1)[0].max(dim=-1)[0].max(dim=0)[0]  # [heads,]
            self.max_logits.extend(max_logit_per_head.tolist())  # Store per-head logits
        
        # Apply mask if provided  
        if mask is not None:
            # mask is True where attention is allowed, False where it should be masked
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.output(context)
        
        return output



class GPTModel(nn.Module):
    """Small GPT model"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use Lecun normal initialization to match JAX/Flax defaults
            fan_in = module.weight.shape[1]
            std = (1.0 / fan_in) ** 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use Lecun normal initialization for embeddings too
            fan_in = module.weight.shape[1]
            std = (1.0 / fan_in) ** 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices  
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        mask = ~mask  # Invert for attention
        
        # Track attention logits for each layer
        attention_logits = {}
        
        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            # Clear previous max logits
            block.attention.max_logits.clear()
            x = block(x, mask)
            # Get per-head max logits from this layer (for per-head QK-Clip)
            if block.attention.max_logits:
                # max_logits now contains per-head values
                attention_logits[f'block_{i}'] = block.attention.max_logits.copy()
            else:
                attention_logits[f'block_{i}'] = [0.0] * self.num_heads
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, attention_logits
    
    def get_attention_layers(self) -> List[Tuple[str, nn.Module]]:
        layers = []
        for i, block in enumerate(self.blocks):
            layers.append((f"block_{i}_attention", block.attention))
        return layers


def create_optimizer_muon(model, learning_rate=2e-4):
    """Create MuonClip optimizer for the model"""
    muon_opt = MuonClip(
        model.parameters(),
        lr=learning_rate,
        momentum=0.95,
        weight_decay=0.1,
        tau=100.0
    )
    # Set model reference for QK-Clip
    muon_opt.set_model(model)
    return muon_opt


def create_optimizer_adamw(model, learning_rate=3e-4):
    """Create AdamW optimizer for the model"""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)


def load_shakespeare_data(batch_size: int, seq_len: int, num_batches: int = 100):
    """Load and tokenize the tiny_shakespeare dataset"""
    # Download the dataset directly
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text
    
    # Use tiktoken for BPE tokenization
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize the entire text
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    # Get vocabulary size
    vocab_size = enc.n_vocab
    
    # Create decode function for debugging
    decode = lambda x: enc.decode(x.tolist() if isinstance(x, torch.Tensor) else x)
    
    # Create batches
    data = []
    max_start = len(tokens) - seq_len - 1
    
    for _ in range(num_batches):
        batch_inputs = []
        batch_targets = []
        
        for _ in range(batch_size):
            start_idx = torch.randint(0, max_start, (1,)).item()
            # Input sequence
            inputs = tokens[start_idx:start_idx + seq_len]
            # Target is next token prediction
            targets = tokens[start_idx + 1:start_idx + seq_len + 1]
            
            batch_inputs.append(inputs)
            batch_targets.append(targets)
        
        batch_inputs = torch.stack(batch_inputs)
        batch_targets = torch.stack(batch_targets)
        data.append((batch_inputs, batch_targets))
    
    return data, vocab_size, enc, decode


def train_step_muon(model, inputs, targets, optimizer):
    """Training step with MuonClip optimizer"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits, attention_logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Calculate gradient norm before step
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    # Optimizer step (includes QK-Clip)
    optimizer.step()
    
    # Get max attention logit
    max_logit = 0.0
    if attention_logits:
        for layer_logits in attention_logits.values():
            if isinstance(layer_logits, list):
                max_logit = max(max_logit, max(layer_logits))
            else:
                max_logit = max(max_logit, layer_logits)
    
    return loss.item(), grad_norm, max_logit, attention_logits


def train_step_adamw(model, inputs, targets, optimizer):
    """Training step with standard AdamW optimizer"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits, _ = model(inputs)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Calculate gradient norm
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    # Optimizer step
    optimizer.step()
    
    return loss.item(), grad_norm


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
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Shakespeare data
    print("\nLoading tiny_shakespeare dataset...")
    train_data, vocab_size, enc, decode = load_shakespeare_data(batch_size, max_seq_len, num_batches)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Using GPT-2 tokenizer with BPE encoding")
    
    # Create two identical models
    print("\nInitializing models...")
    model_muon = GPTModel(vocab_size, d_model, num_heads, num_layers, max_seq_len).to(device)
    model_adamw = GPTModel(vocab_size, d_model, num_heads, num_layers, max_seq_len).to(device)
    
    # Copy weights to ensure identical initialization
    model_adamw.load_state_dict(model_muon.state_dict())
    
    # Create optimizers
    print("Creating optimizers...")
    
    # MuonClip setup
    muon_opt = create_optimizer_muon(model_muon, learning_rate=2e-4)
    
    # AdamW setup
    adamw_opt = create_optimizer_adamw(model_adamw, learning_rate=3e-4)
    
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
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Train MuonClip model
            loss_muon, muon_grad_norm, max_logit, attention_logits_muon = train_step_muon(
                model_muon, inputs, targets, muon_opt
            )
            
            # Train AdamW model
            loss_adamw, adamw_grad_norm = train_step_adamw(
                model_adamw, inputs, targets, adamw_opt
            )
            
            # Record metrics
            muon_losses.append(loss_muon)
            adamw_losses.append(loss_adamw)
            epoch_muon_losses.append(loss_muon)
            epoch_adamw_losses.append(loss_adamw)
            muon_max_logits.append(max_logit)
            muon_grad_norms.append(muon_grad_norm)
            adamw_grad_norms.append(adamw_grad_norm)
            
            # Update progress bar
            pbar.set_postfix({
                'MuonClip Loss': f"{loss_muon:.4f}",
                'AdamW Loss': f"{loss_adamw:.4f}",
                'Max Logit': f"{max_logit:.2f}"
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
    plt.savefig('muonclip_pytorch_comparison.png', dpi=150)
    print("\nTraining metrics saved to 'muonclip_pytorch_comparison.png'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Final MuonClip Loss: {muon_losses[-1]:.4f}")
    print(f"  Final AdamW Loss: {adamw_losses[-1]:.4f}")
    print(f"  Max Attention Logit (peak): {max(max_logits):.2f}")
    print(f"  Max Attention Logit (final): {max_logits[-1]:.2f}")
    print(f"  Avg Gradient Norm - MuonClip: {np.mean(muon_grads):.4f}")
    print(f"  Avg Gradient Norm - AdamW: {np.mean(adamw_grads):.4f}")


if __name__ == "__main__":
    print("MuonClip PyTorch Example - Transformer Training")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run training comparison
    muon_model, adamw_model = train_and_compare()
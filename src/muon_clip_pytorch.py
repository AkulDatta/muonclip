import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Callable


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration for matrix orthogonalization.
    
    Args:
        G: Input matrix tensor
        steps: Number of iteration steps
        eps: Small epsilon for numerical stability
        
    Returns:
        Orthogonalized matrix
    """
    # Coefficients from Muon paper
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Convert to float for precision
    X = G.float()
    X /= (X.norm() + eps)
    
    # Handle rectangular matrices by transposing
    if G.size(0) > G.size(1):
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
    
    return X.to(G.dtype)


class MuonClip(torch.optim.Optimizer):
    """
    MuonClip Optimizer - Combines Muon optimizer with QK-Clip for stable LLM training.
    
    This optimizer applies:
    1. Muon updates with Newton-Schulz orthogonalization for 2D+ parameters
    2. Standard momentum for 1D parameters
    3. QK-Clip to prevent attention logit explosion
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum: Momentum coefficient μ (default: 0.95)
        weight_decay: Weight decay coefficient λ (default: 0.01)
        tau: QK-Clip threshold τ (default: 100.0)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        eps: Numerical stability epsilon (default: 1e-7)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        tau: float = 100.0,
        ns_steps: int = 5,
        eps: float = 1e-7
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < tau:
            raise ValueError(f"Invalid tau value: {tau}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            tau=tau,
            ns_steps=ns_steps,
            eps=eps
        )
        super(MuonClip, self).__init__(params, defaults)
        
        # For QK-Clip functionality
        self.model = None
        self.attention_layers = []
    
    def set_model(self, model: nn.Module):
        """
        Set model reference for QK-Clip functionality.
        
        Args:
            model: PyTorch model containing attention layers
        """
        self.model = model
        if hasattr(model, 'get_attention_layers'):
            self.attention_layers = model.get_attention_layers()
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Step 1: Muon optimizer step
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                param_state = self.state[p]
                
                # Initialize momentum buffer
                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = param_state['momentum_buffer']
                
                # Apply momentum: Mt = μMt−1 + Gt
                buf.mul_(momentum).add_(grad)
                
                if p.ndim >= 2:  # 2D+ parameters - use Muon
                    # Apply Newton-Schulz orthogonalization
                    if p.ndim > 2:
                        # Flatten to 2D for orthogonalization
                        original_shape = buf.shape
                        buf_2d = buf.view(buf.shape[0], -1)
                        orthogonal_update = newton_schulz(buf_2d, ns_steps, eps)
                        orthogonal_update = orthogonal_update.view(original_shape)
                    else:
                        orthogonal_update = newton_schulz(buf, ns_steps, eps)
                    
                    # RMS matching factor: √(max(n,m) × 0.2)
                    n, m = p.shape[0], p.shape[1] if p.ndim > 1 else 1
                    rms_factor = math.sqrt(max(n, m) * 0.2)
                    orthogonal_update *= rms_factor
                    
                    # Update parameters: Wt = Wt−1 − η(Ot + λWt−1)
                    p.data.add_(orthogonal_update + weight_decay * p.data, alpha=-lr)
                    
                else:  # 1D parameters - use standard momentum
                    p.data.add_(buf + weight_decay * p.data, alpha=-lr)
        
        # Step 2: Apply QK-Clip
        self._apply_qk_clip()
        
        return loss
    
    def _apply_qk_clip(self):
        """Apply QK-Clip to attention layers to prevent logit explosion."""
        if not self.attention_layers:
            return
        
        tau = self.param_groups[0]['tau']  # Use tau from first group
        
        for layer_name, attention_layer in self.attention_layers:
            if hasattr(attention_layer, 'max_logits') and attention_layer.max_logits:
                # max_logits now contains per-head values
                max_logits_per_head = attention_layer.max_logits
                num_heads = len(max_logits_per_head)
                
                # Apply per-head QK-Clip (Algorithm 1, line 11-17)
                if hasattr(attention_layer, 'query') and hasattr(attention_layer, 'key'):
                    query_weight = attention_layer.query.weight.data  # [d_model, d_model]
                    key_weight = attention_layer.key.weight.data      # [d_model, d_model]
                    
                    d_model = query_weight.shape[0]
                    d_k = d_model // num_heads
                    
                    # Apply per-head scaling
                    for head_idx, max_logit in enumerate(max_logits_per_head):
                        if max_logit > tau:
                            gamma = tau / max_logit
                            sqrt_gamma = math.sqrt(gamma)
                            
                            # Apply scaling to this head's parameters
                            start_idx = head_idx * d_k
                            end_idx = start_idx + d_k
                            
                            query_weight[:, start_idx:end_idx] *= sqrt_gamma
                            key_weight[:, start_idx:end_idx] *= sqrt_gamma


def apply_qk_clip_per_head(
    query_weights: torch.Tensor,
    key_weights: torch.Tensor,
    max_logits_per_head: List[float],
    tau: float = 100.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply per-head QK-Clip following Algorithm 1, lines 11-16.
    
    Args:
        query_weights: [d_model, d_model] Query projection weights
        key_weights: [d_model, d_model] Key projection weights  
        max_logits_per_head: List of max logits per head
        tau: Threshold for clipping
        
    Returns:
        Clipped query and key weights
    """
    d_model = query_weights.shape[0]
    num_heads = len(max_logits_per_head)
    d_k = d_model // num_heads
    
    clipped_query = query_weights.clone()
    clipped_key = key_weights.clone()
    
    # Apply per-head clipping (Algorithm 1, lines 11-16)
    for head_idx, max_logit in enumerate(max_logits_per_head):
        if max_logit > tau:
            gamma = tau / max_logit
            sqrt_gamma = math.sqrt(gamma)
            
            # Apply scaling to this head's parameters
            start_idx = head_idx * d_k
            end_idx = start_idx + d_k
            
            clipped_query[:, start_idx:end_idx] *= sqrt_gamma
            clipped_key[:, start_idx:end_idx] *= sqrt_gamma
    
    return clipped_query, clipped_key


def apply_qk_clip_to_model(
    model: nn.Module,
    attention_logits: dict,
    tau: float = 100.0
) -> None:
    """
    Apply per-head QK-Clip to model parameters.
    
    Args:
        model: PyTorch model with attention layers
        attention_logits: Dict mapping layer names to per-head max logits
        tau: Clipping threshold
    """
    if hasattr(model, 'get_attention_layers'):
        attention_layers = model.get_attention_layers()
        
        for layer_name, attention_layer in attention_layers:
            if layer_name in attention_logits:
                max_logits_per_head = attention_logits[layer_name]
                
                if hasattr(attention_layer, 'query') and hasattr(attention_layer, 'key'):
                    query_weight = attention_layer.query.weight.data
                    key_weight = attention_layer.key.weight.data
                    
                    clipped_q, clipped_k = apply_qk_clip_per_head(
                        query_weight, key_weight, max_logits_per_head, tau
                    )
                    
                    attention_layer.query.weight.data = clipped_q
                    attention_layer.key.weight.data = clipped_k


# Example multi-head attention layer with QK-Clip support
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with QK-Clip support.
    
    This implementation tracks maximum attention logits for QK-Clip.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.max_logits = 0.0  # Track maximum attention logits
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections in batch from d_model => h x d_k
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Track maximum logits for QK-Clip
        with torch.no_grad():
            self.max_logits = torch.max(scores).item()
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.output(context)
        
        return output


# Example usage functions
def create_optimizer_separate_groups(model: nn.Module, lr: float = 1e-3, **kwargs) -> Tuple[MuonClip, torch.optim.AdamW]:
    """
    Create MuonClip for 2D+ parameters and AdamW for 1D parameters.
    
    This is the recommended approach from the paper for optimal performance.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        **kwargs: Additional arguments for MuonClip
        
    Returns:
        Tuple of (MuonClip optimizer, AdamW optimizer)
    """
    # Separate parameters by dimensionality
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    
    # Create optimizers
    muon_optimizer = MuonClip(muon_params, lr=lr, **kwargs)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=lr * 0.3, weight_decay=kwargs.get('weight_decay', 0.01))
    
    # Set model reference for QK-Clip
    muon_optimizer.set_model(model)
    
    return muon_optimizer, adamw_optimizer


def train_step(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    muon_opt: MuonClip,
    adamw_opt: Optional[torch.optim.AdamW] = None,
    criterion: Optional[nn.Module] = None
) -> float:
    """
    Single training step with MuonClip.
    
    Args:
        model: PyTorch model
        data: Input data
        target: Target labels
        muon_opt: MuonClip optimizer
        adamw_opt: Optional AdamW optimizer for 1D params
        criterion: Loss function (default: CrossEntropyLoss)
        
    Returns:
        Loss value
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Forward pass
    output = model(data)
    loss = criterion(output, target)
    
    # Backward pass
    muon_opt.zero_grad()
    if adamw_opt:
        adamw_opt.zero_grad()
    
    loss.backward()
    
    # Optimizer steps
    muon_opt.step()
    if adamw_opt:
        adamw_opt.step()
    
    return loss.item()
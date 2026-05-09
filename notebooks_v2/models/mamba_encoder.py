"""
Mamba Encoder for Time Series Anomaly Detection

A Selective State Space Model (S6) implementation for contextual anomaly detection.
Based on: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

Key Innovation: The selectivity mechanism allows the model to dynamically decide what
information to remember or forget based on the current input - perfect for detecting
when new observations violate learned contextual expectations.

Architecture:
    Input (B, L, 1) → Embedding → [Mamba Block] x N → Pooling → Classification Head
    
    Where each Mamba Block contains:
        - Linear projection to expanded dimension
        - 1D Convolution for local context
        - Selective SSM (the core innovation)
        - Output projection with residual connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - The core of Mamba.
    
    Unlike traditional SSMs with fixed dynamics, the selectivity mechanism makes
    the state transition parameters (Δ, B, C) input-dependent. This allows the
    model to selectively propagate or forget information based on content.
    
    State Space Equations:
        h_t = Ā * h_{t-1} + B̄ * x_t    (state update)
        y_t = C * h_t                    (output)
    
    Where Ā = exp(Δ * A) and B̄ = Δ * B are discretized versions,
    and Δ, B, C are computed from the input (selective).
    
    Args:
        d_model: Model dimension (input/output size)
        d_state: SSM state dimension (N in the paper, controls memory capacity)
        d_conv: Local convolution width
        expand: Expansion factor for inner dimension
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Input projection: project to 2x inner dim (for x and z paths)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Local convolution for short-range dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise convolution
        )
        
        # Selective parameters projection
        # Projects input to: Δ (timestep), B (input matrix), C (output matrix)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # dt, B, C
        
        # Δ (delta/timestep) projection and initialization
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize dt bias for stable training
        dt_init_std = (1.0 / d_model) ** 0.5 * dt_scale
        if dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # A parameter (state transition) - initialized as negative for stability
        # This is the only parameter that is NOT input-dependent
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # Keep in log space for numerical stability
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Selective SSM.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            
        Returns:
            Output tensor of shape (B, L, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection: split into x and z (gate) paths
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_path, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Local convolution (causal)
        x_conv = x_path.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal: trim to original length
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)  # SiLU activation
        
        # Compute selective parameters from input
        x_proj_out = self.x_proj(x_conv)  # (B, L, d_state * 2 + 1)
        dt, B, C = x_proj_out.split([1, self.d_state, self.d_state], dim=-1)
        
        # Process Δ (timestep/gate) - controls how much to update state
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)  # Ensure positive
        
        # Get A from log space
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability
        
        # Run the selective scan (the core SSM computation)
        y = self._selective_scan(x_conv, dt, A, B, C)
        
        # Apply skip connection with D
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gate with z path (SiLU gating)
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def _selective_scan(
        self,
        x: torch.Tensor,      # (B, L, d_inner)
        dt: torch.Tensor,     # (B, L, d_inner)
        A: torch.Tensor,      # (d_inner, d_state)
        B: torch.Tensor,      # (B, L, d_state)
        C: torch.Tensor,      # (B, L, d_state)
    ) -> torch.Tensor:
        """
        Selective scan implementation - the heart of Mamba.
        
        This is a sequential scan that could be parallelized with associative scan
        for training efficiency, but we use the sequential version for clarity.
        
        The selectivity comes from dt, B, C being input-dependent:
        - Large dt: integrate more of the new input (forget less of history)
        - Small dt: ignore the new input (preserve history)
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize A and B using dt
        # Ā = exp(Δ * A), B̄ = Δ * B
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        # Sequential scan
        outputs = []
        for t in range(seq_len):
            # State update: h_t = Ā * h_{t-1} + B̄ * x_t
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            
            # Output: y_t = C * h_t
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        return y


class MambaBlock(nn.Module):
    """
    A single Mamba block with residual connection and layer normalization.
    
    Architecture:
        x → LayerNorm → SelectiveSSM → + → output
        └─────────────────────────────────┘
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            
        Returns:
            Output tensor of shape (B, L, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return x + residual


class MambaAnomalyClassifier(nn.Module):
    """
    Mamba-based Anomaly Classifier for Time Series.
    
    This model takes raw univariate time series windows and classifies them
    as normal (0) or anomalous (1) using self-supervised learning.
    
    Architecture:
        Raw Signal → Input Embedding → Mamba Blocks → Pooling → MLP → Sigmoid
        
    The Mamba blocks learn contextual patterns, and the classification head
    learns to distinguish normal patterns from synthetic anomalies.
    
    Args:
        input_dim: Input feature dimension (1 for univariate)
        d_model: Model/embedding dimension
        d_state: SSM state dimension (memory capacity)
        d_conv: Local convolution width
        expand: Expansion factor for inner dimension
        n_layers: Number of Mamba blocks
        dropout: Dropout rate
        pool_strategy: How to aggregate sequence for classification
            - 'last': Use only the last hidden state
            - 'mean': Average all hidden states
            - 'max': Max pool all hidden states
            - 'attention': Learned attention pooling
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
        pool_strategy: str = 'last',
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pool_strategy = pool_strategy
        
        # Input embedding: project raw signal to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm_f = nn.LayerNorm(d_model)
        
        # Attention pooling (if used)
        if pool_strategy == 'attention':
            self.attention_weights = nn.Linear(d_model, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for anomaly classification.
        
        Args:
            x: Input tensor of shape (B, L) or (B, L, 1)
            return_features: If True, also return the pooled features
            
        Returns:
            logits: Raw classification logits of shape (B, 1)
            features: (optional) Pooled features of shape (B, d_model)
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L) → (B, L, 1)
        
        # Input projection
        x = self.input_proj(x)  # (B, L, d_model)
        
        # Pass through Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm_f(x)  # (B, L, d_model)
        
        # Pooling for classification
        features = self._pool(x)  # (B, d_model)
        
        # Classification
        logits = self.classifier(features)  # (B, 1)
        
        if return_features:
            return logits, features
        return logits
    
    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence representations for classification.
        
        Args:
            x: Sequence tensor of shape (B, L, d_model)
            
        Returns:
            Pooled tensor of shape (B, d_model)
        """
        if self.pool_strategy == 'last':
            return x[:, -1, :]
        elif self.pool_strategy == 'mean':
            return x.mean(dim=1)
        elif self.pool_strategy == 'max':
            return x.max(dim=1)[0]
        elif self.pool_strategy == 'attention':
            # Learned attention weights
            weights = self.attention_weights(x)  # (B, L, 1)
            weights = F.softmax(weights, dim=1)
            return (x * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pool strategy: {self.pool_strategy}")
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get anomaly probabilities.
        
        Args:
            x: Input tensor of shape (B, L) or (B, L, 1)
            
        Returns:
            Probability tensor of shape (B,)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary anomaly predictions.
        
        Args:
            x: Input tensor of shape (B, L) or (B, L, 1)
            threshold: Classification threshold
            
        Returns:
            Binary predictions of shape (B,)
        """
        proba = self.predict_proba(x)
        return (proba >= threshold).long()


# ============================================================================
# Utility functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_mamba_classifier(
    window_size: int = 100,
    d_model: int = 64,
    n_layers: int = 4,
    **kwargs
) -> MambaAnomalyClassifier:
    """
    Factory function to create a Mamba Anomaly Classifier.
    
    Args:
        window_size: Length of input windows (for documentation, not used in model)
        d_model: Model dimension
        n_layers: Number of Mamba blocks
        **kwargs: Additional arguments passed to MambaAnomalyClassifier
        
    Returns:
        Configured MambaAnomalyClassifier instance
    """
    model = MambaAnomalyClassifier(
        input_dim=1,
        d_model=d_model,
        n_layers=n_layers,
        **kwargs
    )
    
    print(f"Created MambaAnomalyClassifier:")
    print(f"  - Window size: {window_size}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Number of layers: {n_layers}")
    print(f"  - Total parameters: {count_parameters(model):,}")
    
    return model


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create model
    model = create_mamba_classifier(
        window_size=100,
        d_model=64,
        n_layers=4,
        d_state=16,
        pool_strategy='last'
    ).to(device)
    
    # Test forward pass
    batch_size = 8
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 1).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
        proba = model.predict_proba(x)
        preds = model.predict(x)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probabilities: {proba[:4].cpu().numpy()}")
    print(f"  Predictions: {preds[:4].cpu().numpy()}")
    print("\n✓ Model test passed!")

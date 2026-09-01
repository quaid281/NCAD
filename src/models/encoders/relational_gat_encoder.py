"""Spatial-Temporal Relational Graph Attention Network (GAT) Encoder for NCAD-CS.

Combines causal temporal convolutions with dynamic inter-variable graph attention
to capture both temporal patterns and cross-sensor relational dependencies.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFirstLayerNorm(nn.Module):
    """LayerNorm across channels when input is (batch, channels, time)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class CausalTCNBlock(nn.Module):
    """Two-layer causal dilated TCN block with residual connection and LayerNorm."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.10):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = TimeFirstLayerNorm(channels)
        self.norm2 = TimeFirstLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        x = F.pad(inputs, (self.padding, 0))
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout(x)

        x = F.pad(x, (self.padding, 0))
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return F.gelu(x + residual)


class RelationalGraphAttentionLayer(nn.Module):
    """Multi-Head Graph Attention layer modeling dynamic inter-variable dependencies.
    
    Operates on node features of shape (batch, num_nodes, node_dim).
    Learns dynamic pairwise attention weights alpha_{ij} with node identity embeddings.
    """

    def __init__(
        self,
        node_dim: int,
        num_heads: int = 4,
        dropout: float = 0.10,
        use_node_identity: bool = True,
        max_nodes: int = 256,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = max(node_dim // num_heads, 8)
        self.all_head_dim = self.head_dim * num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(node_dim, self.all_head_dim)
        self.k_proj = nn.Linear(node_dim, self.all_head_dim)
        self.v_proj = nn.Linear(node_dim, self.all_head_dim)
        self.out_proj = nn.Linear(self.all_head_dim, node_dim)

        self.use_node_identity = use_node_identity
        if use_node_identity:
            self.node_identity = nn.Parameter(torch.randn(max_nodes, self.all_head_dim) * 0.02)
        else:
            self.node_identity = None

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(node_dim)
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node states of shape (batch, num_nodes, node_dim).
            return_attention: If True, also return attention weights.
            
        Returns:
            Updated node states (batch, num_nodes, node_dim) or (states, attention_weights).
        """
        B, N, D = x.shape

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_node_identity and self.node_identity is not None:
            ident = self.node_identity[:N].unsqueeze(0)  # (1, N, all_head_dim)
            q = q + ident
            k = k + ident

        # Reshape for multi-head attention: (B, H, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product graph attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # Message aggregation
        msg = torch.matmul(attn_weights_drop, v)  # (B, H, N, head_dim)
        msg = msg.transpose(1, 2).contiguous().view(B, N, self.all_head_dim)
        msg = self.out_proj(msg)

        # Residual connection + LayerNorm
        h1 = self.norm(x + self.dropout(msg))
        # Feed-forward block + LayerNorm
        h2 = self.norm2(h1 + self.ffn(h1))

        if return_attention:
            return h2, attn_weights
        return h2


class RelationalGATEncoder(nn.Module):
    """Spatial-Temporal Relational Graph Attention Network Encoder.
    
    1. Extracts individual temporal dynamics per variable using causal dilated convolutions.
    2. Models inter-variable relational topological dependencies using stacked Relational GAT layers.
    3. Aggregates multi-pathway temporal and statistical-spectral representations into the latent space.
    """

    architecture: str = "relational_gat"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        filters: int = 64,
        tcn_layers: int = 4,
        gat_layers: int = 2,
        gat_heads: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.filters = filters
        self.tcn_layers = tcn_layers
        self.gat_layers = gat_layers
        self.kernel_size = kernel_size

        # Temporal pathway: causal dilated TCN
        self.input_projection = nn.Conv1d(input_dim, filters, kernel_size=1)
        self.tcn_blocks = nn.ModuleList(
            [
                CausalTCNBlock(
                    channels=filters,
                    kernel_size=kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
                for layer in range(tcn_layers)
            ]
        )

        # Spatial-relational pathway: Graph Attention over variables
        # We project temporal representations per variable into graph node features
        self.node_proj = nn.Linear(filters, filters)
        self.gat_blocks = nn.ModuleList(
            [
                RelationalGraphAttentionLayer(
                    node_dim=filters,
                    num_heads=gat_heads,
                    dropout=dropout,
                    use_node_identity=True,
                    max_nodes=max(input_dim, filters, 128),
                )
                for _ in range(gat_layers)
            ]
        )

        # Temporal trajectory pooling
        self.temporal_dense = nn.Sequential(
            nn.Linear(filters * 3, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
        )

        # Statistical & spectral moments pooling
        self.stat_dense = nn.Sequential(
            nn.Linear(filters * 6, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
        )

        # Relational graph topology pooling
        self.relational_dense = nn.Sequential(
            nn.Linear(filters, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
        )

        # Latent bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.Linear(128 + 128 + 64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, inputs: torch.Tensor, return_attention: bool = False) -> torch.Tensor | Tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.
        
        Args:
            inputs: Tensor of shape (batch, sequence_length, input_dim).
            return_attention: If True, also return relational attention matrices.
            
        Returns:
            Latent embedding (batch, latent_dim).
        """
        if inputs.ndim != 3:
            raise ValueError(f"Expected tensor with shape (batch, sequence, features), got {inputs.shape}")

        B, T, D = inputs.shape

        # 1. Temporal Causal Convolutions: (B, T, D) -> (B, filters, T)
        x_t = inputs.transpose(1, 2)
        x_t = self.input_projection(x_t)
        for block in self.tcn_blocks:
            x_t = block(x_t)

        # 2. Extract temporal trajectory moments
        last_features = x_t[:, :, -1]
        max_features = torch.max(x_t, dim=2).values
        avg_features = torch.mean(x_t, dim=2)
        temporal_emb = self.temporal_dense(torch.cat([last_features, max_features, avg_features], dim=1))

        # 3. Statistical-spectral features
        stat_emb = self.stat_dense(self._statistical_spectral_features(x_t))

        # 4. Spatial Relational Graph Attention
        if x_t.shape[2] > 64:
            graph_nodes = F.adaptive_avg_pool1d(x_t, 32).transpose(1, 2)  # (B, 32, filters)
        else:
            graph_nodes = x_t.transpose(1, 2)  # (B, T, filters)

        graph_nodes = self.node_proj(graph_nodes)  # (B, N, filters)

        attentions = []
        for gat in self.gat_blocks:
            if return_attention:
                graph_nodes, att = gat(graph_nodes, return_attention=True)
                attentions.append(att)
            else:
                graph_nodes = gat(graph_nodes)

        relational_emb = self.relational_dense(graph_nodes.mean(dim=1))

        # 5. Bottleneck fusion: Temporal + Statistical + Relational
        fused = torch.cat([temporal_emb, stat_emb, relational_emb], dim=1)
        latent = self.bottleneck(fused)
        latent = torch.nan_to_num(latent)

        if return_attention:
            return latent, attentions
        return latent

    @staticmethod
    def _statistical_spectral_features(x: torch.Tensor) -> torch.Tensor:
        """Compute statistical and spectral moments along the time dimension."""
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2, unbiased=False)
        centered = x - mean.unsqueeze(-1)
        safe_std = torch.clamp(std, min=1e-6)
        skew = torch.mean(centered**3, dim=2) / (safe_std**3)
        kurtosis = torch.mean(centered**4, dim=2) / (safe_std**4)
        energy = torch.mean(x**2, dim=2)

        signs = torch.sign(x)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        if x.shape[2] > 1:
            zero_crossing = torch.mean(torch.abs(signs[:, :, 1:] - signs[:, :, :-1]), dim=2) * 0.5
        else:
            zero_crossing = torch.zeros_like(mean)

        features = torch.cat([mean, std, skew, kurtosis, energy, zero_crossing], dim=1)
        return torch.nan_to_num(features)

"""Hybrid-pooling TCN encoder for NCAD-CS."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFirstLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class CausalTCNBlock(nn.Module):
    """Two-layer causal TCN block with residual connection and layer norm."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
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


class HybridTCNEncoder(nn.Module):
    """Shared encoder used for full windows, context windows, and reference contexts."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        filters: int = 64,
        tcn_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.filters = filters
        self.tcn_layers = tcn_layers
        self.input_projection = nn.Conv1d(input_dim, filters, kernel_size=1)
        self.blocks = nn.ModuleList(
            [CausalTCNBlock(filters, kernel_size=kernel_size, dilation=2**layer, dropout=dropout) for layer in range(tcn_layers)]
        )

        self.temporal_dense = nn.Sequential(
            nn.Linear(filters * 3, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
        )
        self.stat_dense = nn.Sequential(
            nn.Linear(filters * 6, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Expected tensor with shape (batch, sequence, features).")

        x = inputs.transpose(1, 2)
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)

        last_features = x[:, :, -1]
        max_features = torch.max(x, dim=2).values
        avg_features = torch.mean(x, dim=2)
        temporal_features = self.temporal_dense(torch.cat([last_features, max_features, avg_features], dim=1))

        stat_features = self.stat_dense(self._statistical_spectral_features(x))
        latent = self.bottleneck(torch.cat([temporal_features, stat_features], dim=1))
        return torch.nan_to_num(latent)

    @staticmethod
    def _statistical_spectral_features(x: torch.Tensor) -> torch.Tensor:
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


def contrastive_loss(z_full: torch.Tensor, z_context: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Paper contrastive objective: positives close, synthetic negatives at least margin apart."""

    distances = torch.linalg.norm(z_full - z_context, dim=1)
    positive_loss = (1.0 - labels) * distances.pow(2)
    negative_loss = labels * F.relu(margin - distances).pow(2)
    return torch.mean(positive_loss + negative_loss)

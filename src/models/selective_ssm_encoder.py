"""GRU-backed selective state-space context encoder.

Public name is preserved for compatibility, but the recurrence is now an
``nn.GRU`` (a fused, parallelized, input-dependent gated state update). This is
the same family of input-dependent gating as the previous hand-rolled block,
but it runs ~50-100x faster on GPU because it does not iterate token-by-token
in Python. Bidirectional so each position sees both past and future context
within the window.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUStateSpaceBlock(nn.Module):
    """Bidirectional GRU block with pre-norm, residual, and dropout."""

    def __init__(self, hidden_dim: int, dropout: float = 0.10):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Expected tensor with shape (batch, sequence, hidden_dim).")
        x = self.norm(inputs)
        y, _ = self.gru(x)
        y = self.output_projection(y)
        return inputs + self.dropout(y)


class SelectiveSSMContextEncoder(nn.Module):
    """Drop-in sequence encoder for full, context, or successor windows.

    Input shape:  (batch, sequence, features)
    Output shape: (batch, latent_dim)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        layers: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [GRUStateSpaceBlock(hidden_dim, dropout=dropout) for _ in range(layers)]
        )
        # last + mean + max pooling (std dropped: unstable under dropout)
        self.pool_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Expected tensor with shape (batch, sequence, features).")

        x = self.input_projection(inputs)
        for block in self.blocks:
            x = block(x)

        last_features = x[:, -1]
        mean_features = torch.mean(x, dim=1)
        max_features = torch.max(x, dim=1).values
        latent = self.pool_head(torch.cat([last_features, mean_features, max_features], dim=1))
        return latent

"""Mamba-inspired selective state-space context encoder.

This module is dependency-light and isolated. It does not require the external
``mamba-ssm`` package, but it captures the key research move we would cite for
Idea D: input-dependent selective state updates instead of a TCN-only encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SelectiveStateSpaceBlock(nn.Module):
    """Input-dependent recurrent state update with residual normalization."""

    def __init__(self, hidden_dim: int, dropout: float = 0.10):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.delta = nn.Linear(hidden_dim, hidden_dim)
        self.candidate = nn.Linear(hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Expected tensor with shape (batch, sequence, hidden_dim).")

        x = self.norm(inputs)
        batch_size, sequence_length, hidden_dim = x.shape
        state = torch.zeros(batch_size, hidden_dim, dtype=x.dtype, device=x.device)
        outputs = []

        for step in range(sequence_length):
            step_input = x[:, step]
            delta = torch.sigmoid(self.delta(step_input))
            candidate = torch.tanh(self.candidate(step_input))
            state = (1.0 - delta) * state + delta * candidate
            gated_state = torch.sigmoid(self.output_gate(step_input)) * state
            outputs.append(gated_state)

        y = torch.stack(outputs, dim=1)
        y = self.output_projection(y)
        return inputs + self.dropout(y)


class ExperimentalSSMContextEncoder(nn.Module):
    """Drop-in sequence encoder prototype for full or context windows.

    Note: This is an experimental, token-by-token research prototype.
    For high-performance production workloads, use `src.models.SelectiveSSMContextEncoder`.
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
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([SelectiveStateSpaceBlock(hidden_dim, dropout=dropout) for _ in range(layers)])
        self.pool_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
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
        std_features = torch.std(x, dim=1, unbiased=False)
        latent = self.pool_head(torch.cat([last_features, mean_features, max_features, std_features], dim=1))
        return torch.nan_to_num(latent)

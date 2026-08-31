"""TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data (Tuli et al., VLDB 2022).

Two-phase adversarial Transformer architecture utilizing sub-series attention and focus scoring
to detect subtle deviations and contextual anomalies.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 128, dropout: float = 0.10):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor = None) -> torch.Tensor:
        kv = x if memory is None else memory
        attn_out, _ = self.attn(x, kv, kv)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TranAD(nn.Module):
    """TranAD: Deep Transformer Networks with Adversarial Training (VLDB 2022)."""

    def __init__(
        self,
        c_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model

        self.embedding = nn.Linear(c_in, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(e_layers)
        ])

        # Decoder Phase 1 (Coarse Reconstruction)
        self.decoder1_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(d_layers)
        ])
        self.proj1 = nn.Linear(d_model, c_in)

        # Decoder Phase 2 (Adversarial Fine Reconstruction)
        self.decoder2_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(d_layers)
        ])
        self.proj2 = nn.Linear(d_model, c_in)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and two decoders.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels)
            
        Returns:
            rec1: Coarse reconstruction (batch, seq_len, channels)
            rec2: Fine adversarial reconstruction (batch, seq_len, channels)
        """
        enc = self.pos_enc(self.embedding(x))
        for blk in self.encoder_blocks:
            enc = blk(enc)

        # Phase 1
        dec1 = enc
        for blk in self.decoder1_blocks:
            dec1 = blk(dec1, memory=enc)
        rec1 = self.proj1(dec1)

        # Phase 2 (conditioned on enc + rec1 features)
        dec2 = self.pos_enc(self.embedding(rec1))
        for blk in self.decoder2_blocks:
            dec2 = blk(dec2, memory=enc)
        rec2 = self.proj2(dec2)

        return rec1, rec2

    @staticmethod
    def adversarial_loss(
        rec1: torch.Tensor, rec2: torch.Tensor, target: torch.Tensor, epoch: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adversarial two-phase training loss (VLDB 2022).

        Phase 1: Reconstruction loss for Decoder 1: (1 / n) * MSE(rec1, target).
        Phase 2: Epoch-weighted composite loss for Decoder 2:
                 (1 / n) * MSE(rec1, target) + (1 - 1 / n) * MSE(rec2, target).
        """
        n = max(epoch, 1)
        l1 = (1.0 / n) * torch.mean((rec1 - target) ** 2)
        l2 = (1.0 / n) * torch.mean((rec1 - target) ** 2) + (1.0 - 1.0 / n) * torch.mean((rec2 - target) ** 2)
        return l1, l2

    def compute_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute composite reconstruction error: 0.5 * ||x - rec1||^2 + 0.5 * ||x - rec2||^2."""
        self.eval()
        with torch.no_grad():
            rec1, rec2 = self.forward(x)
            score1 = torch.mean((rec1 - x) ** 2, dim=-1)
            score2 = torch.mean((rec2 - x) ** 2, dim=-1)
            return 0.5 * score1 + 0.5 * score2

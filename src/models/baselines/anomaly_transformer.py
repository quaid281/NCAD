"""Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy (Xu et al., ICLR 2022).

Computes Series-Association (Self-Attention) and Prior-Association (Gaussian-kernel Attention),
measuring Association Discrepancy via symmetric KL-divergence to distinguish normal patterns
from anomalies.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.05):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class AnomalyAttention(nn.Module):
    """Computes Series-Association S and learns Prior-Association P with Gaussian scale sigma."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.sigma_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            
        Returns:
            out: (batch, seq_len, d_model)
            series_prob: (batch, n_heads, seq_len, seq_len)
            prior_prob: (batch, n_heads, seq_len, seq_len)
        """
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # 1. Series Association (Self-Attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        series_prob = F.softmax(scores, dim=-1)  # (B, H, L, L)
        series_prob_dropped = self.dropout(series_prob)
        out = torch.matmul(series_prob_dropped, v)  # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)

        # 2. Prior Association (Gaussian distribution centered at each timestep)
        # Learnable sigma > 0 for each head and timestep
        sigma = F.softplus(self.sigma_proj(x)) + 1e-4  # (B, L, H)
        sigma = sigma.transpose(1, 2).unsqueeze(-1)  # (B, H, L, 1)

        # Distance matrix |i - j|
        indices = torch.arange(L, device=x.device, dtype=torch.float32)
        dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))  # (L, L)
        dist = dist.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, L, L)  # (B, H, L, L)

        # Gaussian density: 1 / (sqrt(2pi) * sigma) * exp(-dist^2 / (2 * sigma^2))
        gaussian = torch.exp(- (dist ** 2) / (2.0 * (sigma ** 2) + 1e-6))
        prior_prob = gaussian / (gaussian.sum(dim=-1, keepdim=True) + 1e-8)  # normalize

        return out, series_prob, prior_prob


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 128, dropout: float = 0.05):
        super().__init__()
        self.attention = AnomalyAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, series, prior = self.attention(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, series, prior


class AnomalyTransformer(nn.Module):
    """Anomaly Transformer Network (ICLR 2022)."""

    def __init__(
        self,
        c_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.embedding = DataEmbedding(c_in, d_model, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                AnomalyTransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(e_layers)
            ]
        )
        self.projection = nn.Linear(d_model, c_in)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels)
            
        Returns:
            reconstruction: (batch, seq_len, channels)
            series_list: list of series-association attention maps per layer
            prior_list: list of prior-association attention maps per layer
        """
        enc_out = self.embedding(x)
        series_list = []
        prior_list = []

        for block in self.blocks:
            enc_out, series, prior = block(enc_out)
            series_list.append(series)
            prior_list.append(prior)

        reconstruction = self.projection(enc_out)
        return reconstruction, series_list, prior_list

    @staticmethod
    def association_discrepancy(
        prior_list: List[torch.Tensor], series_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute symmetric KL divergence between Prior and Series associations.
        
        KL(P || S) + KL(S || P)
        """
        discrepancies = []
        for prior, series in zip(prior_list, series_list):
            kl_ps = torch.sum(prior * (torch.log(prior + 1e-8) - torch.log(series + 1e-8)), dim=-1)
            kl_sp = torch.sum(series * (torch.log(series + 1e-8) - torch.log(prior + 1e-8)), dim=-1)
            sym_kl = (kl_ps + kl_sp).mean(dim=1)  # average over heads -> (B, L)
            discrepancies.append(sym_kl)

        return torch.stack(discrepancies, dim=0).mean(dim=0)

    def minimax_losses(
        self, x: torch.Tensor, lambda_weight: float = 3.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Official two-phase Minimax Loss for Anomaly Transformer (ICLR 2022).

        Phase 1 (Prior update): Minimize discrepancy with detached series to approximate series association.
        Phase 2 (Series update): Maximize discrepancy with detached prior to enlarge association distance.

        Returns:
            loss_prior: Loss for Phase 1 (Prior association optimization with detached series)
            loss_series: Loss for Phase 2 (Series association optimization with detached prior)
        """
        rec, series_list, prior_list = self.forward(x)
        rec_loss = torch.mean((rec - x) ** 2)

        series_detached = [s.detach() for s in series_list]
        ass_dis_prior = self.association_discrepancy(prior_list, series_detached)
        loss_prior = rec_loss + lambda_weight * ass_dis_prior.mean()

        prior_detached = [p.detach() for p in prior_list]
        ass_dis_series = self.association_discrepancy(prior_detached, series_list)
        loss_series = rec_loss - lambda_weight * ass_dis_series.mean()

        return loss_prior, loss_series

    def compute_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score for input window: Softmax(-AssDis) * ||x - x_rec||_2."""
        self.eval()
        with torch.no_grad():
            rec, series_list, prior_list = self.forward(x)
            rec_loss = torch.mean((rec - x) ** 2, dim=-1)  # (B, L)
            ass_dis = self.association_discrepancy(prior_list, series_list)  # (B, L)
            discrepancy_weight = F.softmax(-ass_dis, dim=-1)  # (B, L)
            score = discrepancy_weight * rec_loss  # (B, L)
            return score

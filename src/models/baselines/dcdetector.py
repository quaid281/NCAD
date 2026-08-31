"""DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection (Yang et al., KDD 2023).

Replaces reconstruction objectives with a pure self-supervised contrastive representation framework.
Combines Patch-wise Attention (temporal) and Inception/Channel-wise Attention with multi-scale
contrastive learning.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Cuts multivariate time series into patches and projects to latent dimension."""

    def __init__(self, c_in: int, patch_size: int, d_model: int, dropout: float = 0.05):
        super().__init__()
        self.patch_size = patch_size
        self.c_in = c_in
        self.d_model = d_model
        self.proj = nn.Linear(patch_size * c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # x: (B, L, C)
        B, L, C = x.shape
        # Pad if L is not divisible by patch_size
        pad_len = (self.patch_size - (L % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        padded_L = x.shape[1]
        num_patches = padded_L // self.patch_size
        # Reshape to (B, num_patches, patch_size * C)
        patches = x.view(B, num_patches, self.patch_size * C)
        emb = self.proj(patches)  # (B, num_patches, d_model)
        return self.dropout(emb), num_patches


class DualAttentionBlock(nn.Module):
    """Dual Attention block performing patch-wise temporal attention and channel-wise feature mixing."""

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 128, dropout: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Patch-wise Multi-Head Attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(self.dropout(attn), v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class DCdetector(nn.Module):
    """DCdetector: Dual Attention Contrastive Representation Learning (KDD 2023)."""

    def __init__(
        self,
        c_in: int,
        patch_size1: int = 8,
        patch_size2: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.c_in = c_in
        self.patch_size1 = patch_size1
        self.patch_size2 = patch_size2
        self.d_model = d_model

        # Branch 1: Fine-scale patches
        self.patch_emb1 = PatchEmbedding(c_in, patch_size1, d_model, dropout=dropout)
        self.blocks1 = nn.ModuleList([
            DualAttentionBlock(d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(e_layers)
        ])

        # Branch 2: Coarse-scale patches
        self.patch_emb2 = PatchEmbedding(c_in, patch_size2, d_model, dropout=dropout)
        self.blocks2 = nn.ModuleList([
            DualAttentionBlock(d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(e_layers)
        ])

        # Latent projection heads
        self.proj_head1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.proj_head2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward_branch(self, x: torch.Tensor, branch: int = 1) -> Tuple[torch.Tensor, int]:
        if branch == 1:
            emb, n_patches = self.patch_emb1(x)
            for blk in self.blocks1:
                emb = blk(emb)
            z = self.proj_head1(emb)
        else:
            emb, n_patches = self.patch_emb2(x)
            for blk in self.blocks2:
                emb = blk(emb)
            z = self.proj_head2(emb)
        return z, n_patches

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through dual multi-scale contrastive branches.
        
        Args:
            x: (B, L, C)
            
        Returns:
            z1: representations from branch 1 (B, N1, d_model)
            z2: representations from branch 2 (B, N2, d_model)
        """
        z1, _ = self.forward_branch(x, branch=1)
        z2, _ = self.forward_branch(x, branch=2)
        return z1, z2

    @staticmethod
    def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Cross-scale multi-scale contrastive loss (KDD 2023).
        
        Maximizes representation consistency across global temporal views.
        """
        # Pool representations across patches: (B, d_model)
        h1 = F.normalize(z1.mean(dim=1), dim=-1)
        h2 = F.normalize(z2.mean(dim=1), dim=-1)

        # Cosine similarity matrix across batch
        sim_matrix = torch.matmul(h1, h2.T) / 0.1  # temperature = 0.1
        labels = torch.arange(len(h1), device=z1.device)
        loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2.0
        return loss

    def compute_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute representation discrepancy anomaly score between dual multiscale branches."""
        self.eval()
        with torch.no_grad():
            B, L, C = x.shape
            z1, n_p1 = self.forward_branch(x, branch=1)
            z2, n_p2 = self.forward_branch(x, branch=2)

            # Upsample patch representations back to sequence length L
            # z1: (B, N1, d_model) -> interpolate -> (B, L, d_model)
            z1_up = F.interpolate(z1.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)
            z2_up = F.interpolate(z2.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)

            # Cosine distance / Euclidean discrepancy in representation space
            z1_norm = F.normalize(z1_up, dim=-1)
            z2_norm = F.normalize(z2_up, dim=-1)
            discrepancy = 1.0 - torch.sum(z1_norm * z2_norm, dim=-1)  # (B, L)
            return discrepancy

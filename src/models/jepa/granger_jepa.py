"""Granger Causal Cross-Channel Joint Embedding Predictive Architecture (GrangerCausalJEPA).

Rooted in Causal Inference and Inter-Sensor Physical Coupling:
1. Encodes each multivariate channel independently into channel embeddings:
       H_ctx in R^{B x C x D},  H_tgt in R^{B x C x D}
2. Causal Cross-Attention Matrix G in [0, 1]^{C x C} where G_ij = P(channel_j drives channel_i):
       H_hat_tgt = G @ H_ctx
3. Dual Physical Anomaly Scoring:
       Channel Prediction Residual (Point Anomaly):    S_channel = 1/C * sum_c ||h_tgt^(c) - h_hat_tgt^(c)||_2
       Causal Graph Disruption (Contextual Anomaly):   S_graph   = ||G - G_nominal||_F
4. Total Anomaly Score:
       S = S_channel + alpha * S_graph
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss


class ChannelTCNEncoder(nn.Module):
    """Encodes each channel's 1D time series into a channel embedding in R^D."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tensor of shape (B, L, C) -> (B, C, D)."""
        B, L, C = x.shape
        # Permute to (B*C, 1, L)
        x_flat = x.permute(0, 2, 1).contiguous().view(B * C, 1, L)
        h_flat = self.conv(x_flat).view(B * C, -1)  # (B*C, D)
        return h_flat.view(B, C, -1)                # (B, C, D)


class CausalCrossAttentionHead(nn.Module):
    """Predicts inter-channel causal attention matrix G and future channel states."""

    def __init__(self, latent_dim: int = 32, num_heads: int = 4, dropout: float = 0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, H_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute predicted target channel states and causal coupling matrix G.
        
        Args:
            H_ctx: Channel embeddings, shape (B, C, D)
            
        Returns:
            H_pred: Predicted target channel embeddings, shape (B, C, D)
            G: Causal adjacency matrix, shape (B, C, C)
        """
        B, C, D = H_ctx.shape
        Q = self.q_proj(H_ctx)  # (B, C, D)
        K = self.k_proj(H_ctx)  # (B, C, D)
        V = self.v_proj(H_ctx)  # (B, C, D)

        # Causal attention weights G in [0, 1]^{C x C}
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(D)  # (B, C, C)
        G = F.softmax(scores, dim=-1)                          # (B, C, C)

        H_pred = self.out_proj(torch.bmm(G, V))                # (B, C, D)
        return H_pred, G


class GrangerCausalJEPAModel(JEPABase):
    """Granger Causal Cross-Channel JEPA (GrangerCausalJEPA)."""

    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 32,
        num_heads: int = 4,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        self.context_encoder = ChannelTCNEncoder(in_channels=1, latent_dim=latent_dim)
        self.target_encoder = self.init_target_encoder(self.context_encoder)

        self.causal_head = CausalCrossAttentionHead(
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Buffer for nominal causal adjacency matrix
        self.register_buffer("G_nominal", torch.zeros(1, 1, 1))
        self.register_buffer("G_nominal_fitted", torch.tensor(False))
        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        H_ctx = self.context_encoder(context_windows)  # (B, C, D)
        H_pred, G = self.causal_head(H_ctx)           # (B, C, D), (B, C, C)

        if target_windows is None:
            return H_ctx.mean(dim=1), None, H_pred, G

        self.target_encoder.eval()
        with torch.no_grad():
            H_tgt = self.target_encoder(target_windows)  # (B, C, D)

        return H_ctx, H_tgt, H_pred, G

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        pred_weight: float = 1.0,
        sparse_weight: float = 0.05,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute Granger causal prediction loss + graph sparsity + VICReg."""
        H_ctx, H_tgt, H_pred, G = self.forward(ctx, tgt)

        # 1. Causal Channel Prediction Loss: ||H_tgt - H_pred||_F^2
        loss_pred = torch.mean(torch.sum((H_tgt - H_pred) ** 2, dim=(-1, -2)))

        # 2. Graph Sparsity / Structure Regularization: encourage sparse physical couplings
        loss_sparse = torch.mean(torch.sum(torch.abs(G - torch.eye(G.shape[1], device=ctx.device)), dim=(-1, -2)))

        # 3. Representation Non-Collapse (VICReg on pooled context)
        z_ctx_pooled = H_ctx.mean(dim=1)  # (B, D)
        std_c = torch.sqrt(torch.var(z_ctx_pooled, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_ctx_pooled, eps=eps)

        total_loss = (
            pred_weight * loss_pred
            + sparse_weight * loss_sparse
            + var_weight * var_c
            + cov_weight * cov_c
        )

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_pred": float(loss_pred.item()),
            "loss_sparse": float(loss_sparse.item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        alpha_graph: float = 0.30,
        **kwargs,
    ) -> torch.Tensor:
        """Compute decoupled dual anomaly score: Channel Prediction + Causal Graph Disruption."""
        self.eval()
        H_ctx = self.context_encoder(context_windows)
        H_tgt = self.target_encoder(observed_target_windows)

        H_pred, G = self.causal_head(H_ctx)

        # 1. Channel Prediction Discrepancy (Point Anomaly / Local Sensor Fault)
        # Average Euclidean distance across channels
        channel_dists = torch.sqrt(torch.sum((H_tgt - H_pred) ** 2, dim=-1) + 1e-8)  # (B, C)
        score_channel = torch.mean(channel_dists, dim=-1)                             # (B,)

        # 2. Causal Coupling Disruption (Contextual Anomaly / Broken Interaction)
        if bool(self.G_nominal_fitted.item()) and self.G_nominal.shape == G.shape[1:]:
            graph_diff = G - self.G_nominal.unsqueeze(0)
            score_graph = torch.sqrt(torch.sum(graph_diff ** 2, dim=(-1, -2)) + 1e-8)
        else:
            # Deviation from identity / baseline
            I_C = torch.eye(G.shape[1], device=G.device).unsqueeze(0)
            score_graph = torch.sqrt(torch.sum((G - I_C) ** 2, dim=(-1, -2)) + 1e-8)

        total_score = score_channel + alpha_graph * score_graph
        return total_score

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        # Fit nominal causal adjacency matrix G_nominal
        self.eval()
        all_G = []
        for i in range(0, len(context_windows), batch_size):
            ctx_b = context_windows[i : i + batch_size]
            H_ctx = self.context_encoder(ctx_b)
            _, G = self.causal_head(H_ctx)
            all_G.append(G)
        G_cat = torch.cat(all_G, dim=0)
        self.G_nominal = torch.mean(G_cat, dim=0)  # (C, C)
        self.G_nominal_fitted.copy_(torch.tensor(True))

        def residual_fn(ctx_b, tgt_b):
            H_ctx = self.context_encoder(ctx_b)
            H_tgt = self.target_encoder(tgt_b)
            H_pred, _ = self.causal_head(H_ctx)
            # Pool channel residuals into (B, D)
            return torch.mean(H_tgt - H_pred, dim=1)

        fit_covariance_batched(
            self,
            context_windows,
            target_windows,
            residual_fn=residual_fn,
            dim=self.latent_dim,
            batch_size=batch_size,
            reg=reg,
            precision_buffer=self.precision_matrix,
            residual_mean_buffer=self.residual_mean,
            fitted_buffer=self.precision_fitted,
        )


GrangerCausalJEPA = GrangerCausalJEPAModel

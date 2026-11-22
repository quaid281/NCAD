"""Episodic Memory Bank Cross-Attention Latent World Model JEPA (MemoryBankLatentWorldJEPA).

Maintains a lossless episodic keyframe memory bank in latent space:
1. Stores the Context Trajectory as Memory Keys and Values:
       K_mem = KeyNet(Z_ctx),  V_mem = ValNet(Z_ctx) in R^{B x C x D}
2. Cross-Attention Querying during Autonomous Rollout:
       Q_tau = QueryNet(z_hat_{C+tau}) in R^{B x 1 x D}
       Attn = softmax(Q_tau @ K_mem.T / sqrt(D)) in R^{B x 1 x C}
       c_mem = Attn @ V_mem in R^{B x 1 x D}
3. Memory-Guided Transition:
       z_hat_{C+tau+1} = z_hat_{C+tau} + TransitionNet([z_hat_{C+tau}, c_mem])
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss
from src.models.jepa.latent_world_jepa import SpatialFrameEncoder


class LatentMemoryBankCore(nn.Module):
    """Memory bank with cross-attention and recurrent transition."""

    def __init__(self, latent_dim: int = 32, num_heads: int = 4, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.q_proj = nn.Linear(latent_dim, latent_dim)

        self.recurrent_cell = nn.GRUCell(latent_dim * 2, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode_memory_bank(self, Z_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute K_mem and V_mem from context trajectory (B, C, D)."""
        K_mem = self.k_proj(Z_ctx)
        V_mem = self.v_proj(Z_ctx)
        return K_mem, V_mem

    def step_with_memory(
        self,
        z_t: torch.Tensor,
        h_prev: torch.Tensor,
        K_mem: torch.Tensor,
        V_mem: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step with cross-attention memory retrieval: z_t in (B, D)."""
        B, D = z_t.shape
        Q = self.q_proj(z_t).unsqueeze(1)  # (B, 1, D)

        # Cross-attention weights over context keyframes
        scores = torch.bmm(Q, K_mem.transpose(1, 2)) / np.sqrt(D)  # (B, 1, C)
        attn = F.softmax(scores, dim=-1)                            # (B, 1, C)
        c_mem = torch.bmm(attn, V_mem).squeeze(1)                   # (B, D)

        # Fuse state + memory into recurrent update
        in_cat = torch.cat([z_t, c_mem], dim=-1)                    # (B, 2*D)
        h_next = self.recurrent_cell(in_cat, h_prev)                # (B, H)
        delta_z = self.readout(h_next)
        z_hat_next = z_t + delta_z
        return z_hat_next, h_next


class MemoryBankLatentWorldJEPAModel(JEPABase):
    """Episodic Memory Bank Cross-Attention Latent World Model JEPA."""

    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 32,
        num_heads: int = 4,
        hidden_dim: int = 64,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay

        self.context_encoder = SpatialFrameEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.target_encoder = self.init_target_encoder(self.context_encoder)

        self.memory_core = LatentMemoryBankCore(
            latent_dim=latent_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.register_mahalanobis_buffers(latent_dim)

    def forward_latent_trajectory(
        self,
        context_windows: torch.Tensor,
        horizon: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C_len, _ = context_windows.shape
        Z_ctx = self.context_encoder(context_windows)  # (B, C, D)

        K_mem, V_mem = self.memory_core.encode_memory_bank(Z_ctx)
        h = torch.zeros(B, self.hidden_dim, device=context_windows.device)
        Z_ctx_pred = []

        # Phase 1: Context Tracking with Memory Retrieval
        for t in range(C_len):
            z_t = Z_ctx[:, t]
            z_hat_next, h = self.memory_core.step_with_memory(z_t, h, K_mem, V_mem)
            Z_ctx_pred.append(z_hat_next)

        Z_ctx_pred = torch.stack(Z_ctx_pred, dim=1)  # (B, C, D)

        # Phase 2: Autonomous Suspect Rollout with Cross-Attention Memory Retrieval
        Z_sus_pred = []
        curr_z = Z_ctx_pred[:, -1]
        for tau in range(horizon):
            Z_sus_pred.append(curr_z)
            curr_z, h = self.memory_core.step_with_memory(curr_z, h, K_mem, V_mem)

        Z_sus_pred = torch.stack(Z_sus_pred, dim=1)  # (B, H, D)
        return Z_ctx, Z_ctx_pred, Z_sus_pred

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        horizon = target_windows.shape[1] if target_windows is not None else 64
        Z_ctx, Z_ctx_pred, Z_sus_pred = self.forward_latent_trajectory(context_windows, horizon=horizon)

        if target_windows is None:
            return Z_ctx.mean(dim=1), None, Z_sus_pred.mean(dim=1), Z_sus_pred

        self.target_encoder.eval()
        with torch.no_grad():
            Z_tgt = self.target_encoder(target_windows)  # (B, H, D)

        return Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        track_weight: float = 0.5,
        rollout_weight: float = 1.0,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred = self.forward(ctx, tgt)

        loss_track = torch.mean(torch.sum((Z_ctx[:, 1:] - Z_ctx_pred[:, :-1]) ** 2, dim=-1))
        loss_rollout = torch.mean(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1))

        z_flat = Z_ctx.contiguous().view(-1, self.latent_dim)
        std_c = torch.sqrt(torch.var(z_flat, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_flat, eps=eps)

        total_loss = (
            track_weight * loss_track
            + rollout_weight * loss_rollout
            + var_weight * var_c
            + cov_weight * cov_c
        )

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_track": float(loss_track.item()),
            "loss_rollout": float(loss_rollout.item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        horizon = observed_target_windows.shape[1]
        _, _, Z_sus_pred = self.forward_latent_trajectory(context_windows, horizon=horizon)
        Z_tgt = self.target_encoder(observed_target_windows)

        step_residuals = torch.sqrt(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1) + 1e-8)
        mean_res = torch.mean(step_residuals, dim=-1)
        max_res = torch.max(step_residuals, dim=-1)[0]
        return mean_res + 0.5 * max_res

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            horizon = tgt_b.shape[1]
            _, _, Z_sus_pred = self.forward_latent_trajectory(ctx_b, horizon=horizon)
            Z_tgt = self.target_encoder(tgt_b)
            return torch.mean(Z_tgt - Z_sus_pred, dim=1)

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


MemoryBankLatentWorldJEPA = MemoryBankLatentWorldJEPAModel

"""Multi-Scale Dilated Latent World Model JEPA (MultiScaleLatentWorldJEPA).

Decomposes pure latent-space temporal recurrence into 3 multi-rate dilated tracks:
1. Fast Latent Track (d=1):   Instantaneous momentum and high-frequency vibrations (Point Anomalies).
2. Medium Latent Track (d=4): Intermediate flow, hydraulic, and mechanical dynamics.
3. Slow Latent Track (d=16):  Long-term thermal, battery, and macro regime drift (Contextual Anomalies).
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


class MultiScaleLatentCore(nn.Module):
    """Multi-scale dilated latent recurrent core."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # 3 dilated recurrent tracks
        self.cell_fast = nn.GRUCell(latent_dim, hidden_dim)  # step = 1
        self.cell_med = nn.GRUCell(latent_dim, hidden_dim)   # step = 4
        self.cell_slow = nn.GRUCell(latent_dim, hidden_dim)  # step = 16

        self.fuse_readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def step(
        self,
        z_t: torch.Tensor,
        h_fast: torch.Tensor,
        h_med: torch.Tensor,
        h_slow: torch.Tensor,
        step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Fast track always updates
        h_fast_next = self.cell_fast(z_t, h_fast)

        # Medium track updates every 4 steps
        if step_idx % 4 == 0:
            h_med_next = self.cell_med(z_t, h_med)
        else:
            h_med_next = h_med

        # Slow track updates every 16 steps
        if step_idx % 16 == 0:
            h_slow_next = self.cell_slow(z_t, h_slow)
        else:
            h_slow_next = h_slow

        h_cat = torch.cat([h_fast_next, h_med_next, h_slow_next], dim=-1)
        delta_z = self.fuse_readout(h_cat)
        z_hat_next = z_t + delta_z
        return z_hat_next, h_fast_next, h_med_next, h_slow_next


class MultiScaleLatentWorldJEPAModel(JEPABase):
    """Multi-Scale Dilated Latent World Model JEPA."""

    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 32,
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

        self.latent_core = MultiScaleLatentCore(
            latent_dim=latent_dim,
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

        h_fast = torch.zeros(B, self.hidden_dim, device=context_windows.device)
        h_med = torch.zeros(B, self.hidden_dim, device=context_windows.device)
        h_slow = torch.zeros(B, self.hidden_dim, device=context_windows.device)
        Z_ctx_pred = []

        # Phase 1: Context Tracking across 3 dilated scales
        for t in range(C_len):
            z_t = Z_ctx[:, t]
            z_hat_next, h_fast, h_med, h_slow = self.latent_core.step(
                z_t, h_fast, h_med, h_slow, step_idx=t
            )
            Z_ctx_pred.append(z_hat_next)

        Z_ctx_pred = torch.stack(Z_ctx_pred, dim=1)  # (B, C, D)

        # Phase 2: Autonomous Suspect Rollout
        Z_sus_pred = []
        curr_z = Z_ctx_pred[:, -1]
        for tau in range(horizon):
            Z_sus_pred.append(curr_z)
            curr_z, h_fast, h_med, h_slow = self.latent_core.step(
                curr_z, h_fast, h_med, h_slow, step_idx=C_len + tau
            )

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


MultiScaleLatentWorldJEPA = MultiScaleLatentWorldJEPAModel

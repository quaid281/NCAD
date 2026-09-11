"""Selective Gated Memory Latent World Model JEPA (SelectiveLatentWorldJEPA).

Applies Selective State-Space (Mamba/S6) dynamics strictly within latent space:
1. Input-Dependent Timescale Parameter:
       Delta_t = softplus(W_Delta @ z_t + b_Delta) in R^D
2. Discretization & Memory Control:
       A_bar_t = exp(Delta_t * A_nom)  (Memory decay)
       B_bar_t = Delta_t * B_t         (Input injection)
       h_t = A_bar_t * h_{t-1} + B_bar_t * z_t
3. Autonomously freezes memory during steady states (Delta_t -> 0)
   and flushes memory during abrupt regime transitions (Delta_t -> infty).
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


class SelectiveLatentCore(nn.Module):
    """Selective State-Space recurrent core in latent space."""

    def __init__(self, latent_dim: int = 32, d_state: int = 8, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_state = d_state

        # Parameterized continuous A matrix: A_nom in [-inf, 0]
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.1, 1.0, d_state).unsqueeze(0).expand(latent_dim, -1)))

        # Dynamic projection from z_t
        self.delta_proj = nn.Linear(latent_dim, latent_dim)
        self.B_proj = nn.Linear(latent_dim, d_state)
        self.C_proj = nn.Linear(latent_dim, d_state)

        self.layer_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def step(self, z_t: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single selective step: z_t in (B, D), h_prev in (B, D, N)."""
        B, D = z_t.shape
        N = self.d_state

        # 1. Compute dynamic selection parameters
        delta_t = torch.clamp(F.softplus(self.delta_proj(z_t)), min=1e-4, max=5.0)  # (B, D)
        B_t = torch.tanh(self.B_proj(z_t))                                          # (B, N)
        C_t = torch.tanh(self.C_proj(z_t))                                          # (B, N)

        # 2. Continuous-to-discrete parameterization: A_bar strictly in (0, 1]
        A_nom = -torch.exp(torch.clamp(self.A_log, min=-5.0, max=2.0))             # (D, N)
        decay_exp = torch.clamp(delta_t.unsqueeze(-1) * A_nom.unsqueeze(0), min=-20.0, max=0.0)
        A_bar = torch.exp(decay_exp)                                                # (B, D, N) in (0, 1]
        B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)                           # (B, D, N)

        # 3. State update
        h_next = A_bar * h_prev + B_bar * z_t.unsqueeze(-1)                         # (B, D, N)
        h_next = torch.clamp(h_next, min=-50.0, max=50.0)

        # 4. State readout
        y = torch.sum(h_next * C_t.unsqueeze(1), dim=-1) / np.sqrt(N)              # (B, D)
        y_norm = self.layer_norm(y)
        delta_z = self.out_proj(y_norm)                                             # (B, D)
        z_hat_next = z_t + delta_z
        return z_hat_next, h_next



class SelectiveLatentWorldJEPAModel(JEPABase):
    """Selective Gated Memory Latent World Model JEPA."""

    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 32,
        d_state: int = 8,
        hidden_dim: int = 64,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.d_state = d_state
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay

        self.context_encoder = SpatialFrameEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.target_encoder = self.init_target_encoder(self.context_encoder)

        self.latent_core = SelectiveLatentCore(
            latent_dim=latent_dim,
            d_state=d_state,
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

        h = torch.zeros(B, self.latent_dim, self.d_state, device=context_windows.device)
        Z_ctx_pred = []

        # Phase 1: Context Tracking with Selective Gated Retention
        for t in range(C_len):
            z_t = Z_ctx[:, t]
            z_hat_next, h = self.latent_core.step(z_t, h)
            Z_ctx_pred.append(z_hat_next)

        Z_ctx_pred = torch.stack(Z_ctx_pred, dim=1)  # (B, C, D)

        # Phase 2: Autonomous Suspect Rollout
        Z_sus_pred = []
        curr_z = Z_ctx_pred[:, -1]
        for tau in range(horizon):
            Z_sus_pred.append(curr_z)
            curr_z, h = self.latent_core.step(curr_z, h)

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


SelectiveLatentWorldJEPA = SelectiveLatentWorldJEPAModel

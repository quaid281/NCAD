"""Latent-Space World Model Joint Embedding Predictive Architecture (LatentWorldJEPA).

Places 100% of temporal learning, recurrence, and autonomous trajectory rollouts
strictly inside the latent space:
1. Instantaneous Spatial Lifting:
       z_t = E_spatial(x_t) in R^D (per-timestep MLP, strictly ZERO temporal pooling).
2. Pure Latent Recurrent Core:
       h_t = LatentGRUCell(z_t, h_{t-1})
       z_hat_{t+1} = z_t + ReadoutNet(h_t)
3. Autonomous Latent Rollout:
       In the suspect window, the model unrolls forward step-by-step purely in latent space
       without looking at future raw sensor inputs:
       z_hat_{C+1} -> z_hat_{C+2} -> ... -> z_hat_{C+H}
4. Dense Per-Timestep Scoring:
       e_tau = ||z_tgt,tau - z_hat_sus,tau||_2
       S_point   = max_tau e_tau (instantaneous spike)
       S_context = 1/H * sum_tau e_tau * (1 + 0.1 * tau) (divergence over rollout)
       S_total   = mean(e_tau) + 0.5 * max_tau(e_tau)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss


class SpatialFrameEncoder(nn.Module):
    """Instantaneous spatial frame encoder: x_t in R^C -> z_t in R^D."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes time series tensor of shape (B, T, C) -> (B, T, D) instantaneously."""
        B, T, C = x.shape
        x_flat = x.contiguous().view(B * T, C)
        z_flat = self.net(x_flat)
        return z_flat.view(B, T, self.latent_dim)


class LatentRecurrentCore(nn.Module):
    """Recurrent transition core operating strictly within latent space R^D."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.cell = nn.GRUCell(latent_dim, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def step(self, z_t: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single 1-step transition in latent space.
        
        Args:
            z_t: Current latent vector, shape (B, D)
            h_prev: Previous recurrent hidden state, shape (B, H)
            
        Returns:
            z_hat_next: Predicted next latent state, shape (B, D)
            h_next: Updated recurrent hidden state, shape (B, H)
        """
        h_next = self.cell(z_t, h_prev)
        delta_z = self.readout(h_next)
        z_hat_next = z_t + delta_z
        return z_hat_next, h_next


class LatentWorldJEPAModel(JEPABase):
    """Latent-Space World Model JEPA (LatentWorldJEPA)."""

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

        self.latent_core = LatentRecurrentCore(
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
        """Track context in latent space, then roll forward autonomously across horizon."""
        B, C_len, _ = context_windows.shape
        Z_ctx = self.context_encoder(context_windows)  # (B, C, D)

        h = torch.zeros(B, self.hidden_dim, device=context_windows.device)
        Z_ctx_pred = []

        # Phase 1: Context Tracking
        for t in range(C_len):
            z_t = Z_ctx[:, t]
            z_hat_next, h = self.latent_core.step(z_t, h)
            Z_ctx_pred.append(z_hat_next)

        Z_ctx_pred = torch.stack(Z_ctx_pred, dim=1)  # (B, C, D)

        # Phase 2: Autonomous Suspect Rollout (without raw inputs)
        Z_sus_pred = []
        curr_z = Z_ctx_pred[:, -1]  # z_hat_{C+1}
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
        """Compute context tracking loss + multi-step autonomous rollout loss + VICReg."""
        Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred = self.forward(ctx, tgt)

        # 1. Context 1-step tracking loss (aligns predicted next with true next)
        loss_track = torch.mean(torch.sum((Z_ctx[:, 1:] - Z_ctx_pred[:, :-1]) ** 2, dim=-1))

        # 2. Autonomous Rollout loss (multi-step latent prediction vs target encoder)
        loss_rollout = torch.mean(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1))

        # 3. Representation Non-Collapse (VICReg across all context latent vectors)
        z_flat = Z_ctx.contiguous().view(-1, self.latent_dim)  # (B*C, D)
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
            "mean_rollout_dist": float(torch.mean(torch.norm(Z_tgt - Z_sus_pred, dim=-1)).item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Dense per-timestep residual scoring across autonomous latent rollout."""
        self.eval()
        horizon = observed_target_windows.shape[1]
        _, _, Z_sus_pred = self.forward_latent_trajectory(context_windows, horizon=horizon)
        Z_tgt = self.target_encoder(observed_target_windows)  # (B, H, D)

        # Dense point-by-point residual in latent space: (B, H)
        step_residuals = torch.sqrt(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1) + 1e-8)

        # Composite score: Average rollout deviation + Max point spike
        mean_res = torch.mean(step_residuals, dim=-1)  # (B,)
        max_res = torch.max(step_residuals, dim=-1)[0]  # (B,)
        total_score = mean_res + 0.5 * max_res
        return total_score

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            horizon = tgt_b.shape[1]
            _, _, Z_sus_pred = self.forward_latent_trajectory(ctx_b, horizon=horizon)
            Z_tgt = self.target_encoder(tgt_b)
            # Mean residual vector across rollout horizon (B, D)
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


LatentWorldJEPA = LatentWorldJEPAModel

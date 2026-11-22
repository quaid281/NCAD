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
    """Recurrent transition core operating strictly within latent space R^D with adaptive observation gate."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.cell = nn.GRUCell(latent_dim, hidden_dim)
        from src.models.geometric_layers import CayleyOrthogonalGate
        self.cayley_gate = CayleyOrthogonalGate(latent_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Adaptive Kalman-style observation gating network
        self.obs_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.obs_proj = nn.Linear(latent_dim, hidden_dim)

    def step(
        self,
        z_t: torch.Tensor,
        h_prev: torch.Tensor,
        z_obs: Optional[torch.Tensor] = None,
        return_innovation: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Perform a single 1-step transition in latent space with optional observation update.
        
        Args:
            z_t: Current latent vector, shape (B, D)
            h_prev: Previous recurrent hidden state, shape (B, H)
            z_obs: Optional target observation latent vector for closed-loop belief update, shape (B, D)
            return_innovation: If True, also returns the innovation residual (B, D)
            
        Returns:
            z_hat_next: Predicted next latent state, shape (B, D)
            h_next: Updated recurrent hidden state, shape (B, H)
            innovation: (Optional) Observation innovation residual, shape (B, D)
        """
        h_next = self.cell(z_t, h_prev)
        delta_z = self.readout(h_next)
        z_hat_next = self.cayley_gate(z_t) + 0.1 * delta_z

        innovation = torch.zeros_like(z_hat_next)
        if z_obs is not None:
            # Observation innovation residual: y = z_obs - z_hat
            innovation = z_obs - z_hat_next
            # Compute adaptive Kalman-style gating
            gate = self.obs_gate(torch.cat([z_hat_next, z_obs], dim=-1))
            # Belief state correction via gated innovation
            h_next = h_next + self.obs_proj(gate * innovation)

        if return_innovation:
            return z_hat_next, h_next, innovation
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
        target_observations: Optional[torch.Tensor] = None,
        return_innovations: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Track context in latent space, then roll forward across horizon."""
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

        # Phase 2: Suspect Rollout with optional adaptive observation gating
        Z_sus_pred = []
        innovations = []
        curr_z = Z_ctx_pred[:, -1]  # z_hat_{C+1}
        for tau in range(horizon):
            Z_sus_pred.append(curr_z)
            z_obs_tau = (
                target_observations[:, tau]
                if target_observations is not None and tau < target_observations.size(1)
                else None
            )
            curr_z, h, innov = self.latent_core.step(curr_z, h, z_obs=z_obs_tau, return_innovation=True)
            innovations.append(innov)

        Z_sus_pred = torch.stack(Z_sus_pred, dim=1)  # (B, H, D)
        if return_innovations:
            innovations_t = torch.stack(innovations, dim=1)  # (B, H, D)
            return Z_ctx, Z_ctx_pred, Z_sus_pred, innovations_t
        return Z_ctx, Z_ctx_pred, Z_sus_pred

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        horizon = target_windows.shape[1] if target_windows is not None else 64
        Z_tgt = None
        if target_windows is not None:
            self.target_encoder.eval()
            with torch.no_grad():
                Z_tgt = self.target_encoder(target_windows)  # (B, H, D)

        Z_ctx, Z_ctx_pred, Z_sus_pred = self.forward_latent_trajectory(
            context_windows,
            horizon=horizon,
            target_observations=Z_tgt,
        )

        if target_windows is None:
            return Z_ctx.mean(dim=1), None, Z_sus_pred.mean(dim=1), Z_sus_pred

        return Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        track_weight: float = 0.0,
        rollout_weight: float = 1.0,
        cov_weight: float = 0.0,
        var_weight: float = 0.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute autonomous multi-step trajectory rollout MSE."""
        Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred = self.forward(ctx, tgt)

        # 1. Autonomous Rollout loss (multi-step latent prediction vs target encoder)
        loss_rollout = torch.mean(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1))
        total_loss = rollout_weight * loss_rollout

        # Diagnostic 1-step tracking
        loss_track = torch.mean(torch.sum((Z_ctx[:, 1:] - Z_ctx_pred[:, :-1]) ** 2, dim=-1))
        if track_weight > 0:
            total_loss = total_loss + track_weight * loss_track

        if var_weight > 0 or cov_weight > 0:
            z_flat = Z_ctx.contiguous().view(-1, self.latent_dim)  # (B*C, D)
            std_c = torch.sqrt(torch.var(z_flat, dim=0, unbiased=False) + eps)
            var_c = torch.mean(F.relu(gamma - std_c))
            cov_c = von_neumann_operator_entropy_loss(z_flat, eps=eps)
            total_loss = total_loss + var_weight * var_c + cov_weight * cov_c

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_rollout": float(loss_rollout.item()),
            "loss_track": float(loss_track.item()),
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
        """Dense per-timestep residual scoring with Kalman innovation tracking."""
        self.eval()
        horizon = observed_target_windows.shape[1]
        Z_tgt = self.target_encoder(observed_target_windows)  # (B, H, D)
        _, _, Z_sus_pred, innovations = self.forward_latent_trajectory(
            context_windows,
            horizon=horizon,
            target_observations=Z_tgt,
            return_innovations=True,
        )

        # Dense point-by-point residual in latent space: (B, H)
        step_residuals = torch.sqrt(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1) + 1e-8)
        step_innovations = torch.sqrt(torch.sum(innovations ** 2, dim=-1) + 1e-8)

        # Composite score: Average rollout deviation + Max point spike + Innovation energy
        mean_res = torch.mean(step_residuals, dim=-1)  # (B,)
        max_res = torch.max(step_residuals, dim=-1)[0]  # (B,)
        innov_score = torch.mean(step_innovations, dim=-1)  # (B,)
        total_score = mean_res + 0.5 * max_res + 0.5 * innov_score
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

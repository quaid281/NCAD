"""Kinematic-Adaptive Latent-Space World Model JEPA (KinematicAdaptiveLatentJEPA).

Synthesized from CMU AHEAD (2026, arXiv:2606.02486) predictive world models:
1. Phase-Space Latent Representation:
       z_t in R^D
       v_t = dot{z}_t = z_t - z_{t-1} (instantaneous latent velocity)
       a_t = ddot{z}_t = v_t - v_{t-1} = z_t - 2*z_{t-1} + z_{t-2} (latent acceleration)
       s_t = [z_t, v_t, a_t] in R^{3D}
2. Kinematic Physics Prior + Neural Correction:
       z_hat_{t+1} = z_t + v_t + 0.5 * (a_t + Delta a(h_t))
       sigma_{t+1}^2 = softplus(Readout_sigma(h_t)) + eps (epistemic dispersion)
3. Autonomous Phase-Space Suspect Rollout:
       Rolls out step-by-step purely in latent kinematic phase space.
4. Adaptive Horizon Uncertainty-Weighted Anomaly Scoring:
       e_tau = ||z_tgt,tau - z_hat_tau||^2 / (sigma_tau^2 + eps)
       Detects breakdown of physical momentum and trajectory predictability.
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
        """Encodes (B, T, C) -> (B, T, D) with strictly zero temporal pooling."""
        B, T, C = x.shape
        x_flat = x.contiguous().view(B * T, C)
        z_flat = self.net(x_flat)
        return z_flat.view(B, T, self.latent_dim)


def compute_discrete_kinematics(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute discrete velocity v_t and acceleration a_t from latent trajectory z.
    
    Args:
        z: Latent tensor of shape (B, T, D)
        
    Returns:
        v: Latent velocity of shape (B, T, D)
        a: Latent acceleration of shape (B, T, D)
    """
    B, T, D = z.shape
    v = torch.zeros_like(z)
    if T > 1:
        v[:, 1:] = z[:, 1:] - z[:, :-1]
        v[:, 0] = v[:, 1]  # edge padding

    a = torch.zeros_like(z)
    if T > 2:
        a[:, 2:] = v[:, 2:] - v[:, 1:-1]
        a[:, 1] = a[:, 2]
        a[:, 0] = a[:, 2]
    elif T == 2:
        a[:, 1] = v[:, 1] - v[:, 0]
        a[:, 0] = a[:, 1]
    return v, a


class KinematicTransitionCore(nn.Module):
    """Kinematic transition core operating on phase-space state s_t = [z_t, v_t, a_t]."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input is 3*latent_dim (position + velocity + acceleration)
        self.cell = nn.GRUCell(latent_dim * 3, hidden_dim)

        # Neural acceleration correction network
        self.accel_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Epistemic dispersion head (variance log-scale)
        self.dispersion_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def step(
        self,
        z_t: torch.Tensor,
        v_t: torch.Tensor,
        a_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one kinematic step in latent space.
        
        Returns:
            z_next: Predicted next latent position (B, D)
            v_next: Predicted next latent velocity (B, D)
            a_next: Predicted next latent acceleration (B, D)
            sigma2: Predicted uncertainty dispersion (B, D)
            h_next: Updated GRU hidden state (B, H)
        """
        s_t = torch.cat([z_t, v_t, a_t], dim=-1)  # (B, 3*D)
        h_next = self.cell(s_t, h_prev)

        delta_a = self.accel_net(h_next)  # (B, D)
        total_a = a_t + delta_a
        v_next = v_t + total_a
        z_next = z_t + v_next

        log_var = self.dispersion_net(h_next)
        sigma2 = F.softplus(log_var) + 1e-4

        return z_next, v_next, total_a, sigma2, h_next


class KinematicAdaptiveLatentJEPAModel(JEPABase):
    """Kinematic Adaptive Latent World Model JEPA (CMU AHEAD 2026 insight)."""

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

        self.latent_core = KinematicTransitionCore(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.register_mahalanobis_buffers(latent_dim)

    def forward_latent_trajectory(
        self,
        context_windows: torch.Tensor,
        horizon: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tracks context in kinematic phase space, then rolls forward autonomously."""
        B, C_len, _ = context_windows.shape
        Z_ctx = self.context_encoder(context_windows)  # (B, C, D)
        V_ctx, A_ctx = compute_discrete_kinematics(Z_ctx)  # (B, C, D) each

        h = torch.zeros(B, self.hidden_dim, device=context_windows.device)
        Z_ctx_pred = []
        Sig_ctx_pred = []

        # Phase 1: Context Tracking
        for t in range(C_len):
            z_t = Z_ctx[:, t]
            v_t = V_ctx[:, t]
            a_t = A_ctx[:, t]
            z_next, v_next, a_next, sig2, h = self.latent_core.step(z_t, v_t, a_t, h)
            Z_ctx_pred.append(z_next)
            Sig_ctx_pred.append(sig2)

        Z_ctx_pred = torch.stack(Z_ctx_pred, dim=1)  # (B, C, D)
        Sig_ctx_pred = torch.stack(Sig_ctx_pred, dim=1)

        # Phase 2: Autonomous Suspect Rollout (Closed-loop kinematic unrolling)
        curr_z = Z_ctx_pred[:, -1]
        curr_v = curr_z - Z_ctx[:, -1]
        curr_a = curr_v - V_ctx[:, -1]

        Z_sus_pred = []
        Sig_sus_pred = []

        for tau in range(horizon):
            Z_sus_pred.append(curr_z)
            curr_z, curr_v, curr_a, sig2, h = self.latent_core.step(curr_z, curr_v, curr_a, h)
            Sig_sus_pred.append(sig2)

        Z_sus_pred = torch.stack(Z_sus_pred, dim=1)  # (B, H, D)
        Sig_sus_pred = torch.stack(Sig_sus_pred, dim=1)  # (B, H, D)

        return Z_ctx, Z_ctx_pred, Z_sus_pred, Sig_sus_pred

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        horizon = target_windows.shape[1] if target_windows is not None else 64
        Z_ctx, Z_ctx_pred, Z_sus_pred, Sig_sus_pred = self.forward_latent_trajectory(
            context_windows, horizon=horizon
        )

        if target_windows is None:
            return Z_ctx.mean(dim=1), None, Z_sus_pred.mean(dim=1), Z_sus_pred, Sig_sus_pred

        self.target_encoder.eval()
        with torch.no_grad():
            Z_tgt = self.target_encoder(target_windows)  # (B, H, D)

        return Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred, Sig_sus_pred

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
        """Compute Kinematic Heteroscedastic NLL loss + VICReg representation non-collapse."""
        Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred, Sig_sus_pred = self.forward(ctx, tgt)

        # 1. 1-step tracking loss along context
        loss_track = torch.mean(torch.sum((Z_ctx[:, 1:] - Z_ctx_pred[:, :-1]) ** 2, dim=-1))

        # 2. Autonomous Rollout Gaussian NLL loss (balances prediction accuracy with uncertainty)
        sq_err = (Z_tgt - Z_sus_pred) ** 2  # (B, H, D)
        nll = 0.5 * (sq_err / (Sig_sus_pred + eps) + torch.log(Sig_sus_pred + eps))
        loss_rollout = torch.mean(torch.sum(nll, dim=-1))

        # 3. VICReg non-collapse
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
            "mean_dispersion": float(torch.mean(Sig_sus_pred).item()),
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
        """Adaptive horizon uncertainty-weighted scoring."""
        self.eval()
        horizon = observed_target_windows.shape[1]
        _, _, Z_sus_pred, Sig_sus_pred = self.forward_latent_trajectory(context_windows, horizon=horizon)
        Z_tgt = self.target_encoder(observed_target_windows)  # (B, H, D)

        # Uncertainty-scaled Mahalanobis-like step residuals: (B, H)
        sq_err = torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1)
        mean_sig = torch.mean(Sig_sus_pred, dim=-1) + 1e-4
        scaled_step_res = torch.sqrt(sq_err / mean_sig + 1e-8)

        # Composite score: mean rollout anomaly + max spike
        mean_res = torch.mean(scaled_step_res, dim=-1)  # (B,)
        max_res = torch.max(scaled_step_res, dim=-1)[0]  # (B,)
        total_score = mean_res + 0.5 * max_res
        return total_score

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            horizon = tgt_b.shape[1]
            _, _, Z_sus_pred, _ = self.forward_latent_trajectory(ctx_b, horizon=horizon)
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


KinematicAdaptiveLatentJEPA = KinematicAdaptiveLatentJEPAModel
KinematicLatentWorldJEPA = KinematicAdaptiveLatentJEPAModel

"""Spectral-Preserving Latent Physics World Model JEPA (SpectralPhysicsLatentJEPA).

Synthesized from Polymathic AI / Flatiron / Princeton (NeurIPS 2025:
"Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics Emulation"):
1. The Physics Problem:
       Standard latent unrolling models suffer severe power spectral decay,
       over-smoothing high-frequency modes (turbulence, vibration, chatter) over time.
2. Dual-Stream Latent Decoupling:
       z_t = [z_t^macro, z_t^micro] in R^D
       - Macro Stream (D/2): Low-frequency continuous macroscopic drift (latent GRU).
       - Micro Stream (D/2): High-frequency oscillatory dynamics (harmonic latent oscillators).
3. Latent Power Spectral Density (PSD) Invariance Loss:
       PSD(Z) = |rfft(Z, dim=temporal)|^2
       L_PSD = ||log(PSD(Z_hat_sus) + eps) - log(PSD(Z_tgt) + eps)||_1
       Guarantees preservation of high-frequency physical energy across the rollout horizon.
4. Spectral-Temporal Anomaly Scoring:
       S_time    = ||Z_tgt - Z_hat_sus||_2 (time-domain trajectory error)
       S_spectral = ||log PSD(Z_tgt) - log PSD(Z_hat_sus)||_1 (frequency dissipation / modal breakdown)
       S_total   = S_time + lambda_spec * S_spectral
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
        """Encodes (B, T, C) -> (B, T, D) instantaneously."""
        B, T, C = x.shape
        x_flat = x.contiguous().view(B * T, C)
        z_flat = self.net(x_flat)
        return z_flat.view(B, T, self.latent_dim)


class DualStreamSpectralCore(nn.Module):
    """Dual-stream latent core separating macroscopic drift from harmonic oscillations."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be divisible by 2 for dual stream"
        self.latent_dim = latent_dim
        self.d_macro = latent_dim // 2
        self.d_micro = latent_dim // 2
        self.hidden_dim = hidden_dim

        # Macro stream: Recurrent drift transition
        self.macro_cell = nn.GRUCell(self.d_macro, hidden_dim)
        self.macro_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.d_macro),
        )

        # Micro stream: Harmonic oscillatory operator with learnable frequencies and damping
        # State per oscillator is (position, velocity)
        self.log_freq = nn.Parameter(torch.randn(self.d_micro) * 0.5)
        self.log_damping = nn.Parameter(torch.zeros(self.d_micro) - 1.0)
        self.micro_coupling = nn.Linear(self.d_macro, self.d_micro)

    def step(
        self,
        z_t: torch.Tensor,
        h_prev: torch.Tensor,
        micro_vel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance both macroscopic and microscopic streams one step.
        
        Args:
            z_t: Current latent vector [macro, micro], shape (B, D)
            h_prev: Macro hidden state, shape (B, H)
            micro_vel: Microscopic oscillator velocity, shape (B, D/2)
            
        Returns:
            z_next: Predicted next latent state [macro_next, micro_next], shape (B, D)
            h_next: Updated macro hidden state, shape (B, H)
            micro_vel_next: Updated microscopic velocity, shape (B, D/2)
        """
        macro_t = z_t[:, : self.d_macro]
        micro_t = z_t[:, self.d_macro :]

        # 1. Macro update
        h_next = self.macro_cell(macro_t, h_prev)
        delta_macro = self.macro_readout(h_next)
        macro_next = macro_t + delta_macro

        # 2. Micro update (Damped harmonic oscillator driven by macro stream)
        omega = torch.exp(torch.clamp(self.log_freq, -3.0, 3.0))  # (D/2,)
        gamma = torch.sigmoid(self.log_damping) * 0.1  # small damping factor
        macro_drive = torch.tanh(self.micro_coupling(macro_next))

        accel = - (omega ** 2) * micro_t - 2 * gamma * micro_vel + 0.1 * macro_drive
        dt = 0.1
        micro_vel_next = micro_vel + dt * accel
        micro_next = micro_t + dt * micro_vel_next

        z_next = torch.cat([macro_next, micro_next], dim=-1)
        return z_next, h_next, micro_vel_next


def compute_latent_psd(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute real Power Spectral Density along temporal dimension.

    Uses real-valued orthogonal discrete trigonometric projection to guarantee
    zero NVRTC runtime JIT compilation and complete driver compatibility on CUDA/WSL.

    Args:
        z: Latent tensor of shape (B, T, D)

    Returns:
        psd: Power spectral density of shape (B, F, D)
    """
    B, T, D = z.shape
    device = z.device
    dtype = z.dtype

    # Orthogonal trigonometric basis for discrete frequencies: k in [0, T//2]
    num_freqs = T // 2 + 1
    t = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)
    k = torch.arange(num_freqs, device=device, dtype=dtype).unsqueeze(0)  # (1, F)
    angles = (2.0 * torch.pi / T) * (t * k)  # (T, F)

    norm_factor = 1.0 / (T ** 0.5)
    w_cos = torch.cos(angles) * norm_factor  # (T, F)
    w_sin = torch.sin(angles) * norm_factor  # (T, F)

    # (B, T, D) projected along T dimension -> (B, F, D) via standard real GEMM
    z_perm = z.transpose(1, 2)  # (B, D, T)
    x_cos = torch.matmul(z_perm, w_cos).transpose(1, 2)  # (B, F, D)
    x_sin = torch.matmul(z_perm, w_sin).transpose(1, 2)  # (B, F, D)

    psd = x_cos.pow(2) + x_sin.pow(2) + eps
    return psd


class SpectralPhysicsLatentJEPAModel(JEPABase):
    """Spectral-Preserving Latent Physics World Model JEPA."""

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

        self.latent_core = DualStreamSpectralCore(
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
        """Tracks context, then rolls forward autonomously with dual-stream physics."""
        B, C_len, _ = context_windows.shape
        Z_ctx = self.context_encoder(context_windows)  # (B, C, D)

        h = torch.zeros(B, self.hidden_dim, device=context_windows.device)
        d_micro = self.latent_dim // 2
        micro_vel = torch.zeros(B, d_micro, device=context_windows.device)
        Z_ctx_pred = []

        # Phase 1: Context Tracking
        for t in range(C_len):
            z_t = Z_ctx[:, t]
            z_next, h, micro_vel = self.latent_core.step(z_t, h, micro_vel)
            Z_ctx_pred.append(z_next)

        Z_ctx_pred = torch.stack(Z_ctx_pred, dim=1)  # (B, C, D)

        # Phase 2: Autonomous Suspect Rollout
        curr_z = Z_ctx_pred[:, -1]
        Z_sus_pred = []
        for tau in range(horizon):
            Z_sus_pred.append(curr_z)
            curr_z, h, micro_vel = self.latent_core.step(curr_z, h, micro_vel)

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
        spectral_weight: float = 0.5,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute time-domain rollout loss + Latent PSD Invariance loss + VICReg."""
        Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred = self.forward(ctx, tgt)

        # 1. 1-step tracking loss along context
        loss_track = torch.mean(torch.sum((Z_ctx[:, 1:] - Z_ctx_pred[:, :-1]) ** 2, dim=-1))

        # 2. Time-domain rollout error
        loss_time = torch.mean(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1))

        # 3. Latent Power Spectral Density (PSD) invariance loss
        psd_pred = compute_latent_psd(Z_sus_pred)
        psd_tgt = compute_latent_psd(Z_tgt)
        loss_spectral = F.l1_loss(torch.log(psd_pred), torch.log(psd_tgt))

        # 4. VICReg non-collapse
        z_flat = Z_ctx.contiguous().view(-1, self.latent_dim)
        std_c = torch.sqrt(torch.var(z_flat, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_flat, eps=eps)

        total_loss = (
            track_weight * loss_track
            + rollout_weight * loss_time
            + spectral_weight * loss_spectral
            + var_weight * var_c
            + cov_weight * cov_c
        )

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_track": float(loss_track.item()),
            "loss_time": float(loss_time.item()),
            "loss_spectral": float(loss_spectral.item()),
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
        """Composite time-domain residual + frequency modal breakdown scoring."""
        self.eval()
        horizon = observed_target_windows.shape[1]
        _, _, Z_sus_pred = self.forward_latent_trajectory(context_windows, horizon=horizon)
        Z_tgt = self.target_encoder(observed_target_windows)  # (B, H, D)

        # 1. Time domain point residuals
        step_residuals = torch.sqrt(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1) + 1e-8)  # (B, H)
        mean_time = torch.mean(step_residuals, dim=-1)  # (B,)
        max_time = torch.max(step_residuals, dim=-1)[0]  # (B,)

        # 2. Spectral power discrepancy (detects frequency dampening / modal shifts)
        psd_pred = compute_latent_psd(Z_sus_pred)
        psd_tgt = compute_latent_psd(Z_tgt)
        spec_diff = torch.mean(torch.abs(torch.log(psd_pred) - torch.log(psd_tgt)), dim=(1, 2))  # (B,)

        total_score = mean_time + 0.5 * max_time + 0.3 * spec_diff
        return total_score

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


SpectralPhysicsLatentJEPA = SpectralPhysicsLatentJEPAModel
SpectralLatentWorldJEPA = SpectralPhysicsLatentJEPAModel

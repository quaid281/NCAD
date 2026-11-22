"""Transfer-Function Spectral Coherence Joint Embedding Predictive Architecture (TransferFunctionJEPA).

Rooted in classical linear systems and control theory:
1. Decomposes latent state z in R^D into D/2 complex harmonic pairs:
       z_k = r_k * exp(i * theta_k), where r_k = sqrt(x_k^2 + y_k^2), theta_k = atan2(y_k, x_k)
2. Transfer Function Predictor estimates:
       Gain Filter:    G(z_ctx) in R^{D/2}_+ (frequency amplitude scaling)
       Phase Shift:    Delta_phi(z_ctx) in [-pi, pi]^{D/2} (phase lead/lag response)
3. Dual Physical Anomaly Scoring:
       Gain Explosion (Point Anomaly):       S_gain  = ||r_tgt - G * r_ctx||_2
       Phase Desynchronization (Context):    S_phase = sum_k (1 - cos(theta_tgt - theta_ctx - Delta_phi))
4. Total Anomaly Score:
       S = S_gain + alpha * S_phase
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss


class SpectralTransferHead(nn.Module):
    """Predicts frequency gain G in R^{D/2} and phase delay Delta_phi in R^{D/2}."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, f"latent_dim ({latent_dim}) must be even for complex pairs"
        self.latent_dim = latent_dim
        self.num_modes = latent_dim // 2

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_modes * 2),  # [log_gain, phase_shift]
        )

    def forward(self, z_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(z_ctx)
        log_gain = out[:, : self.num_modes]
        phase_raw = out[:, self.num_modes :]

        # Gain is positive via softplus / exp clamp
        gain = torch.exp(torch.clamp(log_gain, min=-4.0, max=4.0))  # (B, D/2)
        # Phase delay bounded in [-pi, pi]
        phase_delay = torch.pi * torch.tanh(phase_raw)              # (B, D/2)
        return gain, phase_delay


class TransferFunctionJEPAModel(JEPABase):
    """Transfer-Function Spectral Coherence JEPA."""

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, f"latent_dim ({latent_dim}) must be even"
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.num_modes = latent_dim // 2
        self.ema_decay = ema_decay

        self.target_encoder = self.init_target_encoder(context_encoder)

        self.transfer_head = SpectralTransferHead(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.register_mahalanobis_buffers(latent_dim)

    @staticmethod
    def to_polar(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert latent vector z in R^{B x D} to polar coordinates (r, theta) in R^{B x D/2}."""
        B, D = z.shape
        z_2d = z.view(B, D // 2, 2)
        x = z_2d[:, :, 0]
        y = z_2d[:, :, 1]
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
        theta = torch.atan2(y, x)
        return r, theta

    @staticmethod
    def from_polar(r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Convert polar coordinates (r, theta) in R^{B x D/2} back to Cartesian z in R^{B x D}."""
        B, num_modes = r.shape
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z_2d = torch.stack([x, y], dim=-1)  # (B, D/2, 2)
        return z_2d.view(B, num_modes * 2)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        z_ctx = self.context_encoder(context_windows)
        gain, phase_delay = self.transfer_head(z_ctx)

        if target_windows is None:
            return z_ctx, None, gain, phase_delay

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.target_encoder(target_windows)

        return z_ctx, z_tgt, gain, phase_delay

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        gain_weight: float = 1.0,
        phase_weight: float = 1.0,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute spectral magnitude loss + phase coherence loss + VICReg."""
        z_ctx, z_tgt, gain, phase_delay = self.forward(ctx, tgt)

        r_ctx, theta_ctx = self.to_polar(z_ctx)
        r_tgt, theta_tgt = self.to_polar(z_tgt)

        # 1. Spectral Gain Loss: ||r_tgt - G * r_ctx||^2
        loss_gain = torch.mean(torch.sum((r_tgt - gain * r_ctx) ** 2, dim=-1))

        # 2. Phase Coherence Loss: 1 - cos(theta_tgt - (theta_ctx + phase_delay))
        phase_diff = theta_tgt - (theta_ctx + phase_delay)
        loss_phase = torch.mean(torch.sum(1.0 - torch.cos(phase_diff), dim=-1))

        # 3. Representation Non-Collapse (VICReg)
        std_c = torch.sqrt(torch.var(z_ctx, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)

        total_loss = (
            gain_weight * loss_gain
            + phase_weight * loss_phase
            + var_weight * var_c
            + cov_weight * cov_c
        )

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_gain": float(loss_gain.item()),
            "loss_phase": float(loss_phase.item()),
            "mean_gain": float(torch.mean(gain).item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        alpha_phase: float = 0.50,
        **kwargs,
    ) -> torch.Tensor:
        """Compute decoupled dual anomaly score: Gain (Point) + Phase Coherence (Contextual)."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        gain, phase_delay = self.transfer_head(z_ctx)

        r_ctx, theta_ctx = self.to_polar(z_ctx)
        r_tgt, theta_tgt = self.to_polar(z_tgt)

        # 1. Gain Explosion Score (Point Anomaly)
        score_gain = torch.sqrt(torch.sum((r_tgt - gain * r_ctx) ** 2, dim=-1) + 1e-8)  # (B,)

        # 2. Phase Desynchronization Score (Contextual Anomaly)
        phase_diff = theta_tgt - (theta_ctx + phase_delay)
        score_phase = torch.mean(1.0 - torch.cos(phase_diff), dim=-1)                   # (B,)

        # Decoupled composite score
        total_score = score_gain + alpha_phase * score_phase
        return total_score

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            gain, phase_delay = self.transfer_head(z_ctx)
            r_ctx, theta_ctx = self.to_polar(z_ctx)
            # Reconstructed predicted target in Cartesian space
            r_pred = gain * r_ctx
            theta_pred = theta_ctx + phase_delay
            z_pred = self.from_polar(r_pred, theta_pred)
            return z_tgt - z_pred

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


TransferFunctionJEPA = TransferFunctionJEPAModel

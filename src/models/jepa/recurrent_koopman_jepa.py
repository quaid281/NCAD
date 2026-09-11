"""Recurrent State-Space Koopman Joint Embedding Predictive Architecture (RecurrentKoopmanJEPA).

A mathematically pure, physically unified architecture for time-series anomaly detection:
1. Replaces static window-collapsing encoders with an explicit linear State-Space Model (SSM):
       h_t = A @ h_{t-1} + B(x_t),   where h_t in R^D is the true step-wise latent trajectory.
2. Parameterizes A as block-diagonal 2x2 Harmonic Rotation-Dissipation blocks:
       A_k = exp(-gamma_k) * [[cos(omega_k), -sin(omega_k)], [sin(omega_k), cos(omega_k)]]
       - Exact closed-form eigenvalues: lambda_k = exp(-gamma_k +- i * omega_k),  |lambda_k| <= 1.0
       - Mathematically guarantees stable limit cycles on nominal attractor manifolds.
3. Predicts future trajectories via autonomous unforced dynamical rollout:
       h_hat_{C+tau} = A @ h_hat_{C+tau-1},   for tau = 1, ..., H
4. Dense Per-Timestep Anomaly Scoring:
       e_tau = ||h_hat_{C+tau} - h_{C+tau}^tgt||_2   (exact point-by-point residual)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss


class HarmonicStateSpaceCell(nn.Module):
    """Block-diagonal 2x2 harmonic oscillator state-space transition operator A."""

    def __init__(
        self,
        latent_dim: int = 32,
        min_freq: float = 0.01,
        max_freq: float = 3.14159,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, f"latent_dim must be even, got {latent_dim}"
        self.latent_dim = latent_dim
        self.num_modes = latent_dim // 2

        # Initialize frequencies logarithmically across frequency spectrum
        freq_init = torch.exp(torch.linspace(np.log(min_freq), np.log(max_freq), self.num_modes))
        self.log_omega = nn.Parameter(torch.log(freq_init))

        # Initialize damping gamma >= 0 (initialized small for near-conservative limit cycles)
        self.raw_gamma = nn.Parameter(torch.full((self.num_modes,), -3.0))

    @property
    def omega(self) -> torch.Tensor:
        """Oscillation angular frequencies in [0, pi]."""
        return torch.clamp(torch.exp(self.log_omega), min=1e-4, max=np.pi)

    @property
    def gamma(self) -> torch.Tensor:
        """Damping dissipation rates >= 0."""
        return F.softplus(self.raw_gamma)

    @property
    def eigenvalues(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-form complex eigenvalues: lambda_k = exp(-gamma_k +- i * omega_k).
        
        Returns:
            magnitudes: |lambda_k| = exp(-gamma_k) in (0, 1], shape (num_modes,)
            phases: omega_k in [0, pi], shape (num_modes,)
        """
        magnitudes = torch.exp(-self.gamma)
        phases = self.omega
        return magnitudes, phases

    def step(self, h: torch.Tensor) -> torch.Tensor:
        """Apply state transition h_next = A @ h in O(D) elementwise time.
        
        Args:
            h: Latent state tensor, shape (B, D)
            
        Returns:
            h_next: Next latent state tensor, shape (B, D)
        """
        B, D = h.shape
        # Reshape to (B, num_modes, 2)
        h_pairs = h.view(B, self.num_modes, 2)
        h0 = h_pairs[..., 0]  # (B, num_modes)
        h1 = h_pairs[..., 1]  # (B, num_modes)

        cos_w = torch.cos(self.omega)  # (num_modes,)
        sin_w = torch.sin(self.omega)  # (num_modes,)
        decay = torch.exp(-self.gamma) # (num_modes,)

        # 2D rotation and dissipation
        h0_next = decay * (cos_w * h0 - sin_w * h1)
        h1_next = decay * (sin_w * h0 + cos_w * h1)

        h_next = torch.stack([h0_next, h1_next], dim=-1).view(B, D)
        return h_next

    def rollout(self, h_init: torch.Tensor, steps: int) -> torch.Tensor:
        """Roll out autonomous dynamical trajectory for H steps without input forcing.
        
        Args:
            h_init: Initial latent state, shape (B, D)
            steps: Number of forward rollout steps H
            
        Returns:
            trajectory: Predicted latent states, shape (B, steps, D)
        """
        traj = []
        h_curr = h_init
        for _ in range(steps):
            h_curr = self.step(h_curr)
            traj.append(h_curr)
        return torch.stack(traj, dim=1)  # (B, steps, D)


class RecurrentSSMEncoder(nn.Module):
    """Step-wise causal state-space encoder mapping raw input series x_t to latent trajectory h_t."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Input projection B(x_t)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.ssm_cell = HarmonicStateSpaceCell(latent_dim=latent_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        h_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence into a step-by-step latent trajectory.
        
        Args:
            x_seq: Input sequence, shape (B, T, input_dim)
            h_init: Optional initial state, shape (B, latent_dim)
            
        Returns:
            trajectory: Full latent trajectory (h_1, ..., h_T), shape (B, T, latent_dim)
            h_final: Final latent state h_T, shape (B, latent_dim)
        """
        B, T, _ = x_seq.shape
        u_seq = self.input_proj(x_seq)  # (B, T, latent_dim)

        if h_init is None:
            h_curr = torch.zeros(B, self.latent_dim, device=x_seq.device)
        else:
            h_curr = h_init

        traj = []
        for t in range(T):
            u_t = u_seq[:, t, :]
            # SSM Recurrence: h_t = A @ h_{t-1} + u_t
            h_curr = self.ssm_cell.step(h_curr) + u_t
            traj.append(h_curr)

        trajectory = torch.stack(traj, dim=1)  # (B, T, latent_dim)
        h_final = h_curr
        return trajectory, h_final


class RecurrentKoopmanJEPAModel(JEPABase):
    """Recurrent State-Space Koopman JEPA (RecurrentKoopmanJEPA)."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        # Context Encoder (SSM with harmonic oscillator cell)
        self.context_encoder = RecurrentSSMEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Target Encoder (EMA copy of context encoder)
        self.target_encoder = self.init_target_encoder(self.context_encoder)

        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Encode context and autonomously roll out target trajectory.
        
        Args:
            context_windows: Shape (B, C, input_dim)
            target_windows: Optional shape (B, H, input_dim)
            
        Returns:
            h_ctx_traj: Latent context trajectory, shape (B, C, D)
            h_tgt_traj: Latent true target trajectory, shape (B, H, D) or None
            h_pred_traj: Predicted unforced target rollout, shape (B, H, D)
        """
        B, C, _ = context_windows.shape
        H = target_windows.shape[1] if target_windows is not None else 64

        # 1. Encode context window step-by-step
        h_ctx_traj, h_ctx_final = self.context_encoder(context_windows)  # (B, C, D), (B, D)

        # 2. Autonomous Koopman rollout into the future: h_hat_{C+1} ... h_hat_{C+H}
        h_pred_traj = self.context_encoder.ssm_cell.rollout(h_ctx_final, steps=H)  # (B, H, D)

        if target_windows is None:
            return h_ctx_traj, None, h_pred_traj

        # 3. Ground truth target trajectory via EMA target encoder
        self.target_encoder.eval()
        with torch.no_grad():
            h_tgt_traj, _ = self.target_encoder(target_windows, h_init=h_ctx_final)

        return h_ctx_traj, h_tgt_traj, h_pred_traj

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        freq_div_weight: float = 0.05,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute trajectory rollout loss + harmonic diversity + VICReg non-collapse."""
        h_ctx_traj, h_tgt_traj, h_pred_traj = self.forward(ctx, tgt)

        # 1. Dense Point-Wise Rollout Prediction Loss across the full horizon H
        loss_rollout = F.mse_loss(h_pred_traj, h_tgt_traj)

        # 2. Representation Dispersion on context latents (VICReg)
        z_ctx = h_ctx_traj[:, -1, :]  # (B, D)
        std_c = torch.sqrt(torch.var(z_ctx, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)

        # 3. Modal Frequency Diversity Loss (prevents mode collapse among harmonic oscillators)
        omega = self.context_encoder.ssm_cell.omega
        diffs = torch.abs(omega.unsqueeze(0) - omega.unsqueeze(1)) + torch.eye(len(omega), device=omega.device)
        loss_freq_div = torch.mean(1.0 / (diffs + 1e-3))

        total_loss = loss_rollout + var_weight * var_c + cov_weight * cov_c + freq_div_weight * loss_freq_div

        mags, phases = self.context_encoder.ssm_cell.eigenvalues
        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_rollout": float(loss_rollout.item()),
            "mean_eigenvalue_mag": float(torch.mean(mags).item()),
            "min_eigenvalue_mag": float(torch.min(mags).item()),
            "loss_freq_div": float(loss_freq_div.item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        reduction: str = "mean",
        **kwargs,
    ) -> torch.Tensor:
        """Compute dense per-step trajectory anomaly discrepancy."""
        self.eval()
        h_ctx_traj, h_ctx_final = self.context_encoder(context_windows)
        H = observed_target_windows.shape[1]

        # Predicted autonomous trajectory
        h_pred_traj = self.context_encoder.ssm_cell.rollout(h_ctx_final, steps=H)  # (B, H, D)

        # Ground truth target trajectory
        h_tgt_traj, _ = self.target_encoder(observed_target_windows, h_init=h_ctx_final)  # (B, H, D)

        # Step-by-step point-wise residuals in R^{B x H}
        step_residuals = torch.sqrt(torch.sum((h_tgt_traj - h_pred_traj) ** 2, dim=-1) + 1e-8)

        if reduction == "mean":
            window_score = torch.mean(step_residuals, dim=1)  # (B,)
        elif reduction == "max":
            window_score = torch.max(step_residuals, dim=1)[0]
        else:
            window_score = step_residuals

        return window_score

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            _, h_ctx_final = self.context_encoder(ctx_b)
            H = tgt_b.shape[1]
            h_pred_traj = self.context_encoder.ssm_cell.rollout(h_ctx_final, steps=H)
            h_tgt_traj, _ = self.target_encoder(tgt_b, h_init=h_ctx_final)
            # Average residual vector across the rollout horizon
            return torch.mean(h_tgt_traj - h_pred_traj, dim=1)

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


RecurrentKoopmanJEPA = RecurrentKoopmanJEPAModel

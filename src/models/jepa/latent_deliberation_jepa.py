"""Latent-Space Deliberation World Model JEPA (LatentDeliberationJEPA).

Synthesized from DMLR (Liu et al., CVPR 2026) & Monet (Wang et al., CVPR 2026):
1. Instantaneous Spatial Lifting:
       z_t = E_spatial(x_t) in R^D (per-timestep MLP, zero temporal pooling).
2. Context Memory Formulation:
       M_ctx = [z_1, z_2, ..., z_C] in R^{C x D}
3. Candidate Suspect Initialization:
       Z_sus^(0) in R^{H x D} initialized via autonomous latent rollout.
4. Continuous Latent Deliberation ("Thinking Tokens"):
       Performs K continuous test-time / train-time mental thought iterations:
       For k in 0..K-1:
           Attn_k = CrossAttention(query=Z_sus^(k), key=M_ctx, value=M_ctx)
           Z_sus^(k+1) = Z_sus^(k) + alpha * MLP(LayerNorm(Z_sus^(k) + Attn_k))
       Cognitive turbulence delta_k = ||Z_sus^(k+1) - Z_sus^(k)||_2
5. Dual Discrepancy Scoring:
       S_pred = ||Z_tgt - Z_sus^(K)||_2 (predictive trajectory error)
       S_turbulence = mean_k(delta_k) (cognitive dissonance under physical disruption)
       S_total = S_pred + lambda_turb * S_turbulence
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

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


class LatentDeliberationBlock(nn.Module):
    """Single contraction mapping block for iterative latent thought refinement."""

    def __init__(self, latent_dim: int = 32, num_heads: int = 4, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Learnable step scale alpha, initialized small for stable contraction mapping
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, z_sus: torch.Tensor, m_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one deliberation step.
        
        Args:
            z_sus: Candidate suspect trajectory, shape (B, H, D)
            m_ctx: Context memory embeddings, shape (B, C, D)
            
        Returns:
            z_next: Refined suspect trajectory, shape (B, H, D)
            step_diff: Magnitude of thought modification, shape (B,)
        """
        # Cross-attention into context memory
        q = self.norm1(z_sus)
        k = m_ctx
        v = m_ctx
        attn_out, _ = self.attn(q, k, v)

        # Residual update
        z_mid = z_sus + attn_out
        delta = self.mlp(self.norm2(z_mid))
        z_next = z_sus + self.alpha * delta

        # Step turbulence per sample: mean L2 delta across time and features
        step_diff = torch.mean(torch.norm(z_next - z_sus, dim=-1), dim=-1)  # (B,)
        return z_next, step_diff


class LatentDeliberationJEPAModel(JEPABase):
    """Latent Deliberation World Model JEPA (DMLR & Monet CVPR 2026 insight)."""

    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_thought_steps: int = 4,
        num_heads: int = 4,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_thought_steps = num_thought_steps
        self.ema_decay = ema_decay

        self.context_encoder = SpatialFrameEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.target_encoder = self.init_target_encoder(self.context_encoder)

        # Base recurrent core for context tracking and initial rollout proposal
        self.recurrent_cell = nn.GRUCell(latent_dim, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Deliberation contraction module (shared weights across thought iterations)
        self.deliberation_block = LatentDeliberationBlock(
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tracks context, unrolls candidate trajectory, then performs K deliberation steps."""
        B, C_len, _ = context_windows.shape
        Z_ctx = self.context_encoder(context_windows)  # (B, C, D)

        h = torch.zeros(B, self.hidden_dim, device=context_windows.device)
        Z_ctx_pred = []

        # Phase 1: Context Tracking
        for t in range(C_len):
            z_t = Z_ctx[:, t]
            h = self.recurrent_cell(z_t, h)
            delta_z = self.readout(h)
            Z_ctx_pred.append(z_t + delta_z)

        Z_ctx_pred = torch.stack(Z_ctx_pred, dim=1)  # (B, C, D)

        # Phase 2: Initial Autonomous Candidate Rollout Z_sus^(0)
        curr_z = Z_ctx_pred[:, -1]
        Z_sus_init = []
        for tau in range(horizon):
            Z_sus_init.append(curr_z)
            h = self.recurrent_cell(curr_z, h)
            curr_z = curr_z + self.readout(h)
        Z_sus_curr = torch.stack(Z_sus_init, dim=1)  # (B, H, D)

        # Phase 3: Continuous Latent Deliberation (K mental thought iterations)
        turbulences = []
        for k in range(self.num_thought_steps):
            Z_sus_curr, diff_k = self.deliberation_block(Z_sus_curr, Z_ctx)
            turbulences.append(diff_k)

        # Mean cognitive turbulence across thought steps: (B,)
        mean_turbulence = torch.stack(turbulences, dim=1).mean(dim=1)  # (B,)

        return Z_ctx, Z_ctx_pred, Z_sus_curr, mean_turbulence

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        horizon = target_windows.shape[1] if target_windows is not None else 64
        Z_ctx, Z_ctx_pred, Z_sus_pred, turbulence = self.forward_latent_trajectory(
            context_windows, horizon=horizon
        )

        if target_windows is None:
            return Z_ctx.mean(dim=1), None, Z_sus_pred.mean(dim=1), Z_sus_pred, turbulence

        self.target_encoder.eval()
        with torch.no_grad():
            Z_tgt = self.target_encoder(target_windows)  # (B, H, D)

        return Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred, turbulence

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        track_weight: float = 0.5,
        rollout_weight: float = 1.0,
        turbulence_reg: float = 0.1,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute context tracking + deliberated rollout + turbulence stabilization + VICReg."""
        Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred, turbulence = self.forward(ctx, tgt)

        # 1. Context 1-step tracking loss
        loss_track = torch.mean(torch.sum((Z_ctx[:, 1:] - Z_ctx_pred[:, :-1]) ** 2, dim=-1))

        # 2. Deliberated Rollout loss against true target encoder
        loss_rollout = torch.mean(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1))

        # 3. Turbulence stabilization loss (encourages contraction mapping convergence)
        loss_turb = torch.mean(turbulence)

        # 4. VICReg non-collapse
        z_flat = Z_ctx.contiguous().view(-1, self.latent_dim)
        std_c = torch.sqrt(torch.var(z_flat, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_flat, eps=eps)

        total_loss = (
            track_weight * loss_track
            + rollout_weight * loss_rollout
            + turbulence_reg * loss_turb
            + var_weight * var_c
            + cov_weight * cov_c
        )

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_track": float(loss_track.item()),
            "loss_rollout": float(loss_rollout.item()),
            "loss_turb": float(loss_turb.item()),
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
        """Score based on both deliberated trajectory residual and cognitive turbulence."""
        self.eval()
        horizon = observed_target_windows.shape[1]
        _, _, Z_sus_pred, turbulence = self.forward_latent_trajectory(context_windows, horizon=horizon)
        Z_tgt = self.target_encoder(observed_target_windows)  # (B, H, D)

        # Trajectory error across suspect steps
        step_residuals = torch.sqrt(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1) + 1e-8)  # (B, H)
        mean_res = torch.mean(step_residuals, dim=-1)  # (B,)
        max_res = torch.max(step_residuals, dim=-1)[0]  # (B,)

        # Fusion: trajectory deviation + cognitive turbulence dissonance
        total_score = mean_res + 0.5 * max_res + 0.3 * turbulence
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


LatentDeliberationJEPA = LatentDeliberationJEPAModel
DeliberationLatentWorldJEPA = LatentDeliberationJEPAModel

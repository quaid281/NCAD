"""Minimalist JEPA Architectures for Time-Series Anomaly Detection.

Implements three clean, deceptively simple, and highly performant architectures:
1. CosineJEPA: Hyperspherical directional discrepancy (magnitude-invariant, bounded in [0, 2]).
2. CycleJEPA: Bidirectional causal round-trip consistency (Past -> Future & Future -> Past).
3. CosineCycleJEPA: Hyperspherical directional cycle consistency (combining angle-invariance & time-reversibility, bounded in [0, 4]).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss

# =============================================================================
# 1. Cosine JEPA (Hyperspherical Directional Discrepancy)
# =============================================================================

class CosineJEPAModel(JEPABase):
    """Cosine JEPA: Projects representations to the unit hypersphere S^{D-1}.
    
    Eliminates magnitude noise, amplitude spikes, and sensor drift false alarms.
    Anomaly score is strictly bounded in [0.0, 2.0] as 1.0 - cos(theta).
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        predictor_layers: int = 2,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        self.target_encoder = self.init_target_encoder(context_encoder)

        layers = []
        in_d = latent_dim
        for _ in range(predictor_layers - 1):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, latent_dim))
        self.predictor = nn.Sequential(*layers)

        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass projecting to S^{D-1} and computing directional alignment."""
        z_ctx = self.context_encoder(context_windows)
        z_pred_raw = self.predictor(z_ctx)
        z_pred = F.normalize(z_pred_raw, p=2, dim=-1)

        if target_windows is None:
            return z_pred, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt_raw = self.target_encoder(target_windows)
            z_tgt = F.normalize(z_tgt_raw, p=2, dim=-1)

        # Directional cosine similarity in [-1.0, 1.0]
        cos_sim = torch.sum(z_pred * z_tgt, dim=-1)  # (B,)
        angular_discrepancy = 1.0 - cos_sim          # (B,) in [0.0, 2.0]

        return z_pred, z_tgt, angular_discrepancy

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Directional cosine loss + operator entropy dispersion to prevent representation collapse."""
        z_pred, z_tgt, angular_discrepancy = self.forward(ctx, tgt)

        # 1. Cosine alignment loss: E[1 - cos(theta)]
        loss_cosine = torch.mean(angular_discrepancy)

        # 2. Hyperspherical dispersion regularization (prevents all vectors collapsing to a single point on S^{D-1})
        std_p = torch.sqrt(torch.var(z_pred, dim=0, unbiased=False) + eps)
        var_p = torch.mean(F.relu(gamma - std_p))
        cov_p = von_neumann_operator_entropy_loss(z_pred, eps=eps)

        total_loss = loss_cosine + var_weight * var_p + cov_weight * cov_p

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_cosine": float(loss_cosine.item()),
            "loss_var": float(var_p.item()),
            "loss_cov": float(cov_p.item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute exact angular discrepancy in [0, 2] in O(D) time."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_pred = F.normalize(self.predictor(z_ctx), p=2, dim=-1)
        z_tgt = F.normalize(self.target_encoder(observed_target_windows), p=2, dim=-1)

        cos_sim = torch.sum(z_pred * z_tgt, dim=-1)
        return torch.clamp(1.0 - cos_sim, min=0.0, max=2.0)

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        """Fit compatibility buffers."""
        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_pred = F.normalize(self.predictor(z_ctx), p=2, dim=-1)
            z_tgt = F.normalize(self.target_encoder(tgt_b), p=2, dim=-1)
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


# =============================================================================
# 2. Cycle JEPA (Bidirectional Causal Consistency)
# =============================================================================

class CycleJEPAModel(JEPABase):
    """Cycle JEPA: Enforces bidirectional causal consistency (Past -> Future -> Past).
    
    Verifies that observed future states can causally explain their past origins.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        self.target_encoder = self.init_target_encoder(context_encoder)

        # Forward predictor: Past -> Future
        self.fwd_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Backward predictor: Future -> Past
        self.bwd_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        z_ctx = self.context_encoder(context_windows)
        z_fwd_pred = self.fwd_predictor(z_ctx)

        if target_windows is None:
            return z_fwd_pred, None, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.target_encoder(target_windows)

        # Backward reconstruction from observed future to past
        z_bwd_pred = self.bwd_predictor(z_tgt)

        return z_ctx, z_tgt, z_fwd_pred, z_bwd_pred

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        cycle_weight: float = 0.5,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute bidirectional forward + backward + round-trip cycle loss."""
        z_ctx, z_tgt, z_fwd_pred, z_bwd_pred = self.forward(ctx, tgt)

        # 1. Forward predictive loss: ||F(z_ctx) - z_tgt||^2
        loss_fwd = F.mse_loss(z_fwd_pred, z_tgt)

        # 2. Backward causal reconstruction loss: ||B(z_tgt) - z_ctx||^2
        loss_bwd = F.mse_loss(z_bwd_pred, z_ctx)

        # 3. Round-trip Cycle consistency: ||B(F(z_ctx)) - z_ctx||^2
        z_cycle = self.bwd_predictor(z_fwd_pred)
        loss_cycle = F.mse_loss(z_cycle, z_ctx)

        # 4. Representation dispersion
        std_c = torch.sqrt(torch.var(z_ctx, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)

        total_loss = loss_fwd + loss_bwd + cycle_weight * loss_cycle + var_weight * var_c + cov_weight * cov_c

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_fwd": float(loss_fwd.item()),
            "loss_bwd": float(loss_bwd.item()),
            "loss_cycle": float(loss_cycle.item()),
            "loss_var": float(var_c.item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute two-way causal discrepancy: ||F(z_ctx) - z_tgt|| + ||B(z_tgt) - z_ctx||."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        z_fwd_pred = self.fwd_predictor(z_ctx)
        z_bwd_pred = self.bwd_predictor(z_tgt)

        err_fwd = torch.sqrt(torch.sum((z_fwd_pred - z_tgt) ** 2, dim=-1) + 1e-8)
        err_bwd = torch.sqrt(torch.sum((z_bwd_pred - z_ctx) ** 2, dim=-1) + 1e-8)

        return err_fwd + err_bwd

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            z_fwd_pred = self.fwd_predictor(z_ctx)
            return z_tgt - z_fwd_pred

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


# =============================================================================
# 3. Cosine-Cycle JEPA (Directional Hyperspherical Time-Reversibility)
# =============================================================================

class CosineCycleJEPAModel(JEPABase):
    """Cosine-Cycle JEPA: Combines hyperspherical angular projection with bidirectional cycle consistency.
    
    - Scale-invariant and magnitude-normalized.
    - Two-way causal consistency on S^{D-1}.
    - Anomaly score strictly bounded in [0.0, 4.0].
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        self.target_encoder = self.init_target_encoder(context_encoder)

        self.fwd_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.bwd_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        z_ctx = F.normalize(self.context_encoder(context_windows), p=2, dim=-1)
        z_fwd_pred = F.normalize(self.fwd_predictor(z_ctx), p=2, dim=-1)

        if target_windows is None:
            return z_fwd_pred, None, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = F.normalize(self.target_encoder(target_windows), p=2, dim=-1)

        z_bwd_pred = F.normalize(self.bwd_predictor(z_tgt), p=2, dim=-1)

        return z_ctx, z_tgt, z_fwd_pred, z_bwd_pred

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        cycle_weight: float = 0.5,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        z_ctx, z_tgt, z_fwd_pred, z_bwd_pred = self.forward(ctx, tgt)

        # 1. Forward angular loss: 1 - cos(F(z_ctx), z_tgt)
        cos_fwd = torch.sum(z_fwd_pred * z_tgt, dim=-1)
        loss_fwd = torch.mean(1.0 - cos_fwd)

        # 2. Backward angular loss: 1 - cos(B(z_tgt), z_ctx)
        cos_bwd = torch.sum(z_bwd_pred * z_ctx, dim=-1)
        loss_bwd = torch.mean(1.0 - cos_bwd)

        # 3. Cycle round-trip: 1 - cos(B(F(z_ctx)), z_ctx)
        z_cycle = F.normalize(self.bwd_predictor(z_fwd_pred), p=2, dim=-1)
        cos_cycle = torch.sum(z_cycle * z_ctx, dim=-1)
        loss_cycle = torch.mean(1.0 - cos_cycle)

        # 4. Representation dispersion
        std_c = torch.sqrt(torch.var(z_ctx, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)

        total_loss = loss_fwd + loss_bwd + cycle_weight * loss_cycle + var_weight * var_c + cov_weight * cov_c

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_fwd": float(loss_fwd.item()),
            "loss_bwd": float(loss_bwd.item()),
            "loss_cycle": float(loss_cycle.item()),
            "loss_var": float(var_c.item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute two-way directional angular discrepancy in [0.0, 4.0]."""
        self.eval()
        z_ctx = F.normalize(self.context_encoder(context_windows), p=2, dim=-1)
        z_tgt = F.normalize(self.target_encoder(observed_target_windows), p=2, dim=-1)

        z_fwd_pred = F.normalize(self.fwd_predictor(z_ctx), p=2, dim=-1)
        z_bwd_pred = F.normalize(self.bwd_predictor(z_tgt), p=2, dim=-1)

        cos_fwd = torch.sum(z_fwd_pred * z_tgt, dim=-1)
        cos_bwd = torch.sum(z_bwd_pred * z_ctx, dim=-1)

        disc_fwd = torch.clamp(1.0 - cos_fwd, min=0.0, max=2.0)
        disc_bwd = torch.clamp(1.0 - cos_bwd, min=0.0, max=2.0)

        return disc_fwd + disc_bwd

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            z_ctx = F.normalize(self.context_encoder(ctx_b), p=2, dim=-1)
            z_tgt = F.normalize(self.target_encoder(tgt_b), p=2, dim=-1)
            z_fwd_pred = F.normalize(self.fwd_predictor(z_ctx), p=2, dim=-1)
            return z_tgt - z_fwd_pred

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


CosineJEPA = CosineJEPAModel
CycleJEPA = CycleJEPAModel
CosineCycleJEPA = CosineCycleJEPAModel

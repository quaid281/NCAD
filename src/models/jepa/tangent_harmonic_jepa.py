"""Tangent-Harmonic Subspace Projection JEPA (Chapter 2 of Ten Advances).

Projects latent representations to the unit hypersphere S^{D-1} and associates
each point with a moving orthogonal projection P_z = I - z z^T onto the tangent
harmonic space H_1(z^perp) and higher-order traceless harmonic tensors H_2(z^perp).

The anomaly discrepancy is measured via the normalized subspace projection overlap:
    S(z_pred, z_tgt) = (1 - s) + alpha * (1 - s^2) / (D - 1) + beta * (1 - C_2(s))
where s = <z_pred, z_tgt> and C_2(s) is the degree-2 Gegenbauer polynomial.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched, to_device_tensor
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss


class TangentHarmonicProjector(nn.Module):
    """Computes moving tangent-harmonic projection overlaps on S^{D-1}.
    
    Implements the moving projection kernel tr(P_x P_y) from Chapter 2, Section 5.2.
    """

    def __init__(self, latent_dim: int, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta

    def forward(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute tangent-harmonic discrepancy between unit vectors z_pred and z_tgt.
        
        Args:
            z_pred: (B, D) unit vectors on S^{D-1}
            z_tgt: (B, D) unit vectors on S^{D-1}
            
        Returns:
            discrepancy: (B,) scalar discrepancy in [0, 4]
            metrics: dict containing individual harmonic components
        """
        D = float(self.latent_dim)
        # 1. Degree-0 scalar inner product: s in [-1, 1]
        s = torch.sum(z_pred * z_tgt, dim=-1)
        s = torch.clamp(s, -1.0, 1.0)
        d0 = 1.0 - s  # in [0, 2]

        # 2. Degree-1 tangent projector overlap: tr(P_x P_y) = D - 2 + s^2
        # Normalized tangent deficit: 1 - tr(P_x P_y) / (D - 1) = (1 - s^2) / (D - 1)
        d1 = (1.0 - torch.square(s)) / max(D - 1.0, 1.0)

        # 3. Degree-2 Gegenbauer harmonic polynomial:
        # C_2^{(D-2)/2}(s) normalized to 1 at s=1: (D * s^2 - 1) / (D - 1)
        c2_norm = (D * torch.square(s) - 1.0) / max(D - 1.0, 1.0)
        d2 = 1.0 - c2_norm  # in [0, 2]

        # Total combined harmonic discrepancy
        total_discrepancy = d0 + self.alpha * d1 + self.beta * d2

        metrics = {
            "d0_angular": d0.mean(),
            "d1_tangent_overlap": d1.mean(),
            "d2_gegenbauer": d2.mean(),
            "mean_cosine": s.mean(),
        }
        return total_discrepancy, metrics


class TangentHarmonicJEPAModel(JEPABase):
    """Tangent-Harmonic JEPA Architecture.
    
    Combines self-supervised predictive representation learning with moving
    tangent-harmonic projection operators on the unit hypersphere S^{D-1}.
    """


    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        predictor_layers: int = 2,
        ema_decay: float = 0.996,
        alpha: float = 1.0,
        beta: float = 0.5,
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

        self.harmonic_projector = TangentHarmonicProjector(
            latent_dim=latent_dim,
            alpha=alpha,
            beta=beta,
        )

        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass projecting to S^{D-1} and computing tangent harmonic alignment."""
        z_ctx = self.context_encoder(context_windows)
        z_pred_raw = self.predictor(z_ctx)
        z_pred = F.normalize(z_pred_raw, p=2, dim=-1)

        if target_windows is None:
            return z_pred, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt_raw = self.target_encoder(target_windows)
            z_tgt = F.normalize(z_tgt_raw, p=2, dim=-1)

        discrepancy, _ = self.harmonic_projector(z_pred, z_tgt)
        return z_pred, z_tgt, discrepancy

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        cov_weight: float = 0.0,
        var_weight: float = 0.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute tangent-harmonic predictive loss on S^{D-1}."""
        z_pred, z_tgt, discrepancy = self.forward(ctx, tgt)
        harmonic_loss = discrepancy.mean()
        total_loss = harmonic_loss

        # Optional backward-compatible Euclidean regularization if explicitly requested
        std_loss = torch.tensor(0.0, device=ctx.device)
        cov_loss = torch.tensor(0.0, device=ctx.device)
        std_pred = torch.sqrt(z_pred.var(dim=0) + eps)

        if var_weight > 0:
            std_loss = torch.mean(F.relu(gamma - std_pred))
            total_loss = total_loss + var_weight * std_loss

        if cov_weight > 0:
            B = z_pred.size(0)
            z_pred_cent = z_pred - z_pred.mean(dim=0)
            cov_pred = (z_pred_cent.T @ z_pred_cent) / max(B - 1, 1)
            off_diag = cov_pred - torch.diag(torch.diag(cov_pred))
            cov_loss = torch.sum(torch.square(off_diag)) / max(self.latent_dim, 1)
            total_loss = total_loss + cov_weight * cov_loss

        metrics = {
            "loss": total_loss.item(),
            "harmonic_loss": harmonic_loss.item(),
            "std_loss": std_loss.item(),
            "cov_loss": cov_loss.item(),
            "mean_pred_std": std_pred.mean().item(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        """Compute tangent-harmonic predictive discrepancy for anomaly scoring.
        
        Args:
            context_windows: (B, L_ctx, C)
            observed_target_windows: (B, L_tgt, C)
            use_mahalanobis: Whether to whiten discrepancy via fitted resolvent precision.
            
        Returns:
            scores: (B,) anomaly discrepancy scores.
        """
        self.eval()
        z_pred, z_tgt, harmonic_disc = self.forward(context_windows, observed_target_windows)

        if not use_mahalanobis or not bool(self.precision_fitted.item()):
            return harmonic_disc

        # Whitened tangent residual: delta = z_tgt - z_pred
        diff = z_tgt - z_pred
        diff_cent = diff - self.residual_mean
        whitened = torch.sum((diff_cent @ self.precision_matrix) * diff_cent, dim=-1)
        whitened_norm = torch.sqrt(torch.clamp(whitened, min=0.0))

        # Blend directional harmonic score with whitened coordinate residual
        return harmonic_disc * (1.0 + 0.5 * whitened_norm)

    def fit_covariance(
        self,
        context_windows: Union[np.ndarray, torch.Tensor],
        target_windows: Union[np.ndarray, torch.Tensor],
        batch_size: int = 512,
        reg: float = 1e-3,
        method: str = "resolvent",
    ) -> None:
        """Fit empirical residual precision matrix using resolvent spectral filtering."""
        def residual_fn(ctx, tgt):
            z_pred, z_tgt, _ = self.forward(ctx, tgt)
            return z_tgt - z_pred

        fit_covariance_batched(
            self,
            context_windows,
            target_windows,
            residual_fn=residual_fn,
            dim=self.latent_dim,
            batch_size=batch_size,
            reg=reg,
            method=method,
            precision_buffer=self.precision_matrix,
            residual_mean_buffer=self.residual_mean,
            fitted_buffer=self.precision_fitted,
        )

    fit_mahalanobis_covariance = fit_covariance



TangentHarmonicJEPA = TangentHarmonicJEPAModel


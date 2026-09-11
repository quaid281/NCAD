"""Tangent-Normal Subspace Decomposition Joint Embedding Predictive Architecture (TangentNormalJEPA).

Decouples latent space discrepancies into mutually orthogonal components:
1. Normal Component (Point Anomaly):
       e_perp = (I - U @ U.T) @ (z_tgt - z_ctx)
       Measures displacement orthogonal to the nominal attractor manifold (unphysical void state).
2. Tangent Component (Contextual Anomaly):
       e_parallel = U @ U.T @ (z_tgt - z_ctx)
       Measures deviation along the manifold from the contextually predicted velocity v_tangent.
3. Total Anomaly Score:
       S = ||e_perp||_2 + alpha * ||U.T @ e - v_tangent||_2
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss


class TangentSubspaceHead(nn.Module):
    """Predicts local orthonormal tangent frame U in R^{D x d} and tangent velocity v in R^d."""

    def __init__(
        self,
        latent_dim: int = 32,
        subspace_dim: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        assert subspace_dim < latent_dim, f"subspace_dim ({subspace_dim}) must be < latent_dim ({latent_dim})"
        self.latent_dim = latent_dim
        self.subspace_dim = subspace_dim

        # Predicts D x d matrix basis
        self.basis_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * subspace_dim),
        )

        # Predicts expected velocity along tangent subspace
        self.velocity_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, subspace_dim),
        )

    def forward(self, z_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute orthonormal basis U and tangent velocity v.
        
        Args:
            z_ctx: Context latent tensor, shape (B, D)
            
        Returns:
            U: Orthonormal basis tensor, shape (B, D, d), where U.T @ U = I_d
            v_tangent: Expected tangent velocity, shape (B, d)
        """
        B, D = z_ctx.shape
        d = self.subspace_dim

        # Predict basis matrix M in R^{B x D x d}
        M = self.basis_net(z_ctx).view(B, D, d)

        # Compute orthonormal basis U via QR decomposition: Q in R^{B x D x d}
        # Note: QR decomposition ensures U.T @ U = I_d
        Q, _ = torch.linalg.qr(M)
        U = Q  # (B, D, d)

        v_tangent = self.velocity_net(z_ctx)  # (B, d)
        return U, v_tangent


class TangentNormalJEPAModel(JEPABase):
    """Tangent-Normal Subspace Decomposition JEPA (TangentNormalJEPA)."""

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        subspace_dim: int = 4,
        hidden_dim: int = 64,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.subspace_dim = subspace_dim
        self.ema_decay = ema_decay

        self.target_encoder = self.init_target_encoder(context_encoder)

        self.tangent_head = TangentSubspaceHead(
            latent_dim=latent_dim,
            subspace_dim=subspace_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        z_ctx = self.context_encoder(context_windows)
        U, v_tangent = self.tangent_head(z_ctx)

        if target_windows is None:
            return z_ctx, None, U, v_tangent

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.target_encoder(target_windows)

        return z_ctx, z_tgt, U, v_tangent

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        normal_weight: float = 1.0,
        tangent_weight: float = 1.0,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute normal manifold containment loss + tangent velocity loss + VICReg."""
        z_ctx, z_tgt, U, v_tangent = self.forward(ctx, tgt)

        # Displacement vector e = z_tgt - z_ctx in R^{B x D}
        e = z_tgt - z_ctx

        # Tangent coordinates: c_parallel = U^T @ e in R^{B x d}
        c_parallel = torch.bmm(U.transpose(1, 2), e.unsqueeze(-1)).squeeze(-1)  # (B, d)

        # Tangent reconstructed vector: e_parallel = U @ c_parallel in R^{B x D}
        e_parallel = torch.bmm(U, c_parallel.unsqueeze(-1)).squeeze(-1)  # (B, D)

        # Normal vector (orthogonal): e_perp = e - e_parallel in R^{B x D}
        e_perp = e - e_parallel  # (B, D)

        # 1. Normal Manifold Containment Loss: ||e_perp||^2 (nominal dynamics stay on manifold)
        loss_normal = torch.mean(torch.sum(e_perp ** 2, dim=-1))

        # 2. Tangent Velocity Prediction Loss: ||c_parallel - v_tangent||^2 (predicts trajectory along manifold)
        loss_tangent = torch.mean(torch.sum((c_parallel - v_tangent) ** 2, dim=-1))

        # 3. Representation Non-Collapse (VICReg)
        std_c = torch.sqrt(torch.var(z_ctx, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)

        total_loss = (
            normal_weight * loss_normal
            + tangent_weight * loss_tangent
            + var_weight * var_c
            + cov_weight * cov_c
        )

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_normal": float(loss_normal.item()),
            "loss_tangent": float(loss_tangent.item()),
            "mean_normal_norm": float(torch.mean(torch.norm(e_perp, dim=-1)).item()),
            "mean_tangent_norm": float(torch.mean(torch.norm(c_parallel, dim=-1)).item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        alpha_context: float = 0.50,
        **kwargs,
    ) -> torch.Tensor:
        """Compute decoupled dual anomaly score: Point (Normal) + Contextual (Tangent)."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        U, v_tangent = self.tangent_head(z_ctx)
        e = z_tgt - z_ctx

        # Tangent coordinates: c_parallel = U^T @ e in R^{B x d}
        c_parallel = torch.bmm(U.transpose(1, 2), e.unsqueeze(-1)).squeeze(-1)  # (B, d)
        e_parallel = torch.bmm(U, c_parallel.unsqueeze(-1)).squeeze(-1)          # (B, D)
        e_perp = e - e_parallel                                                  # (B, D)

        # 1. Point Anomaly Score (Normal displacement off the manifold)
        score_point = torch.sqrt(torch.sum(e_perp ** 2, dim=-1) + 1e-8)  # (B,)

        # 2. Contextual Anomaly Score (Tangent velocity discrepancy along the manifold)
        score_context = torch.sqrt(torch.sum((c_parallel - v_tangent) ** 2, dim=-1) + 1e-8)  # (B,)

        # Decoupled composite score
        total_score = score_point + alpha_context * score_context
        return total_score

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            U, v_tangent = self.tangent_head(z_ctx)
            e = z_tgt - z_ctx
            c_parallel = torch.bmm(U.transpose(1, 2), e.unsqueeze(-1)).squeeze(-1)
            e_parallel = torch.bmm(U, c_parallel.unsqueeze(-1)).squeeze(-1)
            e_perp = e - e_parallel
            # Composite residual vector: normal displacement + tangent deviation projected back
            tangent_res_D = torch.bmm(U, (c_parallel - v_tangent).unsqueeze(-1)).squeeze(-1)
            return e_perp + 0.5 * tangent_res_D

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


TangentNormalJEPA = TangentNormalJEPAModel

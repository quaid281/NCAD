"""Koopman-Spectral & Lyapunov Joint Embedding Predictive Architecture (KoopmanJEPA).

A physically grounded, mathematically elegant architecture for time-series anomaly detection:
1. Absorbs non-linearities in the encoder E_theta; enforces exact linearity in latent space:
       z_tgt_pred = K(z_ctx) @ z_ctx
2. Evaluates the Eigenvalue Spectrum of K:
       lambda_i in C,  |lambda_i| <= 1.0  (Marginal/Dissipative stability on nominal attractors)
3. Detects anomalies via dual physical signals:
       Signal 1 (Residual): ||z_tgt - K z_ctx||_2
       Signal 2 (Spectral/Lyapunov Instability): max_i ReLU(|lambda_i| - 1.0)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss


class KoopmanOperatorPredictor(nn.Module):
    """Predicts a structured, mathematically stable Koopman transition matrix K(z_ctx) in R^{D x D}.

    Uses the Resolvent / Cayley Transform parameterization (Chapter 6 of Ten Advances in Mathematics):
        K = (I + S + D)^{-1} (I - S - D)
    where S = 1/2(W - W^T) is skew-symmetric and D >= 0 is dissipative damping.
    This guarantees that every eigenvalue satisfies |lambda_i| <= 1.0 algebraically.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Base nominal skew-symmetric matrix generator W_0
        self.W_0 = nn.Parameter(0.01 * torch.randn(latent_dim, latent_dim))
        self.d_0 = nn.Parameter(torch.zeros(latent_dim))

        # Context-adaptive perturbation: Delta W(z_ctx) and Delta d(z_ctx)
        self.adaptive_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * latent_dim + latent_dim),
        )

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """Compute context-conditioned Koopman matrix K in R^{B x D x D}.
        Guarantees ||K||_2 <= 1.0 and |lambda_i(K)| <= 1.0.
        """
        B, D = z_ctx.shape
        device = z_ctx.device
        dtype = z_ctx.dtype

        out = self.adaptive_net(z_ctx)
        delta_W = out[:, : D * D].view(B, D, D) * 0.1
        delta_d = out[:, D * D :] * 0.1

        W = self.W_0.unsqueeze(0) + delta_W
        S = 0.5 * (W - W.transpose(-1, -2))  # Strictly skew-symmetric

        d = F.softplus(self.d_0.unsqueeze(0) + delta_d)  # Strictly non-negative damping
        D_mat = torch.diag_embed(d)

        I = torch.eye(D, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        A_plus = I + S + D_mat
        A_minus = I - S - D_mat

        # Cayley transform: K = (I + S + D)^{-1} (I - S - D)
        K = torch.linalg.solve(A_plus, A_minus)
        return K


class KoopmanJEPAModel(JEPABase):
    """Koopman-Spectral Joint Embedding Predictive Architecture (KoopmanJEPA)."""

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

        self.koopman_predictor = KoopmanOperatorPredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.register_mahalanobis_buffers(latent_dim)

    def compute_spectral_properties(self, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spectral radius and local Lyapunov exponent for Koopman matrix K via fast power iteration.
        
        Args:
            K: Koopman matrices, shape (B, D, D)
            
        Returns:
            spectral_instability: Max eigenvalue expansion beyond unit circle max(|lambda_i| - 1.0)_+, shape (B,)
            lyapunov_exp: Local maximum Lyapunov expansion log ||K||_2, shape (B,)
        """
        B, D, _ = K.shape
        # Fast 2-step power iteration for spectral norm sigma_max(K) in O(D^2)
        # Avoids calling O(D^3) full LAPACK SVD/eigvals on millions of mini-batches
        u = torch.ones(B, D, 1, device=K.device) / np.sqrt(D)
        v = torch.bmm(K, u)
        v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)
        u_next = torch.bmm(K.transpose(1, 2), v)
        sigma_max = torch.norm(u_next, dim=1).squeeze(-1)  # (B,)

        # By spectral radius theorem, max |lambda_i| <= sigma_max(K)
        spectral_instability = F.relu(sigma_max - 1.0)  # (B,)
        lyapunov_exp = torch.log(torch.clamp(sigma_max, min=1e-6))  # (B,)

        return spectral_instability, lyapunov_exp



    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        z_ctx = self.context_encoder(context_windows)
        K = self.koopman_predictor(z_ctx)  # (B, D, D)

        # Linear Koopman prediction: z_pred = K @ z_ctx
        z_pred = torch.bmm(K, z_ctx.unsqueeze(-1)).squeeze(-1)  # (B, D)

        if target_windows is None:
            return z_ctx, None, z_pred, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.target_encoder(target_windows)

        spectral_instability, lyapunov_exp = self.compute_spectral_properties(K)

        return z_ctx, z_tgt, z_pred, spectral_instability, lyapunov_exp

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        stability_weight: float = 0.0,
        cov_weight: float = 0.0,
        var_weight: float = 0.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute Koopman linear prediction loss.
        Spectral stability |lambda_i| <= 1.0 is guaranteed algebraically by Cayley transform.
        """
        z_ctx, z_tgt, z_pred, spectral_instability, lyapunov_exp = self.forward(ctx, tgt)

        # 1. Koopman Linear Prediction Loss: ||z_tgt - K z_ctx||^2
        loss_pred = F.mse_loss(z_pred, z_tgt)
        total_loss = loss_pred

        if stability_weight > 0:
            loss_stability = torch.mean(spectral_instability ** 2)
            total_loss = total_loss + stability_weight * loss_stability
        else:
            loss_stability = torch.tensor(0.0, device=ctx.device)

        if var_weight > 0 or cov_weight > 0:
            std_c = torch.sqrt(torch.var(z_ctx, dim=0, unbiased=False) + eps)
            var_c = torch.mean(F.relu(gamma - std_c))
            cov_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)
            total_loss = total_loss + var_weight * var_c + cov_weight * cov_c

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_pred": float(loss_pred.item()),
            "loss_stability": float(loss_stability.item()),
            "mean_spectral_instability": float(torch.mean(spectral_instability).item()),
            "mean_lyapunov_exp": float(torch.mean(lyapunov_exp).item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        include_spectral: bool = True,
        spectral_weight: float = 0.10,
        **kwargs,
    ) -> torch.Tensor:
        """Compute dual Koopman anomaly score: Residual + Spectral Instability."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        K = self.koopman_predictor(z_ctx)
        z_pred = torch.bmm(K, z_ctx.unsqueeze(-1)).squeeze(-1)

        # 1. Base prediction residual distance
        residual_dist = torch.sqrt(torch.sum((z_tgt - z_pred) ** 2, dim=-1) + 1e-8)

        if include_spectral:
            # 2. Spectral Instability: Fast spectral norm expansion beyond unit circle
            B, D, _ = K.shape
            u = torch.ones(B, D, 1, device=K.device) / np.sqrt(D)
            v = torch.bmm(K, u)
            v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)
            u_next = torch.bmm(K.transpose(1, 2), v)
            sigma_max = torch.norm(u_next, dim=1).squeeze(-1)
            spectral_instability = F.relu(sigma_max - 1.0)
            score = residual_dist + spectral_weight * spectral_instability
        else:
            score = residual_dist

        return score



    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            K = self.koopman_predictor(z_ctx)
            z_pred = torch.bmm(K, z_ctx.unsqueeze(-1)).squeeze(-1)
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


KoopmanJEPA = KoopmanJEPAModel

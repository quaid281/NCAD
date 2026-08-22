"""Time-Series Joint Embedding Predictive Architecture (TS-JEPA).

Eliminates the dependency on artificial synthetic anomaly injections by training
the encoder to predict future latent states directly in representation space,
regularized with non-contrastive VICReg (Variance-Invariance-Covariance) loss.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictor(nn.Module):
    """Predicts target latent embedding from context latent embedding.
    
    A multi-layer bottleneck network with residual connections, LayerNorm, and GELU activations.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        layers = []
        in_d = latent_dim
        for i in range(num_layers):
            out_d = hidden_dim if i < num_layers - 1 else latent_dim
            layers.extend(
                [
                    nn.Linear(in_d, out_d),
                    nn.GELU(),
                    nn.LayerNorm(out_d),
                    nn.Dropout(dropout),
                ]
            )
            in_d = out_d
        self.net = nn.Sequential(*layers)
        self.residual_proj = nn.Identity() if in_d == latent_dim else nn.Linear(latent_dim, latent_dim)

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        """Predict future latent state: (batch, latent_dim) -> (batch, latent_dim)."""
        return self.net(z_context) + self.residual_proj(z_context)


class TSJEPAModel(nn.Module):
    """Time-Series Joint Embedding Predictive Architecture.
    
    Wraps:
    - Context Encoder E_theta (trained with gradients)
    - Target Encoder E_phi (updated via Exponential Moving Average)
    - Latent Predictor P_psi (trained to map E_theta(x_ctx) -> E_phi(x_target))
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int,
        predictor_hidden_dim: int = 64,
        predictor_layers: int = 2,
        ema_decay: float = 0.995,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        # Target encoder is an EMA copy of the context encoder
        self.target_encoder = copy.deepcopy(context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )

        self.register_buffer("precision_matrix", torch.eye(latent_dim))
        self.precision_fitted = False

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward pass.
        
        Args:
            context_windows: Tensor of shape (batch, context_len, features)
            target_windows: Optional tensor of shape (batch, target_len, features)
            
        Returns:
            Tuple of (z_context, z_target_true, z_target_pred)
        """
        # Encode context with active gradients
        z_context = self.context_encoder(context_windows)
        # Predict future target in latent space
        z_target_pred = self.predictor(z_context)

        z_target_true = None
        if target_windows is not None:
            # Encode target future with target encoder (no grad)
            with torch.no_grad():
                z_target_true = self.target_encoder(target_windows)

        return z_context, z_target_true, z_target_pred

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        """Update target encoder weights via Exponential Moving Average (EMA)."""
        m = self.ema_decay if decay is None else decay
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.data)

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical residual covariance matrix for Mahalanobis discrepancy."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_pred = self.predictor(z_ctx)
        z_obs = self.target_encoder(target_windows)
        residuals = z_obs - z_pred

        residuals_centered = residuals - residuals.mean(dim=0, keepdim=True)
        cov = (residuals_centered.T @ residuals_centered) / max(len(residuals) - 1, 1)
        cov_reg = cov + reg * torch.eye(self.latent_dim, device=cov.device)
        self.precision_matrix = torch.linalg.pinv(cov_reg)
        self.precision_fitted = True

    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        """Compute the JEPA prediction residual ||E_target(target) - P(E_ctx(ctx))||.
        
        A pure, unbiased physical dynamics anomaly score without synthetic injections.
        """
        self.eval()
        with torch.no_grad():
            z_ctx = self.context_encoder(context_windows)
            z_pred = self.predictor(z_ctx)
            z_obs = self.target_encoder(observed_target_windows)
            diff = z_obs - z_pred
            if use_mahalanobis and self.precision_fitted:
                mahal = torch.sum((diff @ self.precision_matrix) * diff, dim=-1)
                return torch.sqrt(torch.clamp(mahal, min=1e-8))
            else:
                return torch.linalg.norm(diff, dim=-1)


def jepa_vicreg_loss(
    z_target_pred: torch.Tensor,
    z_target_true: torch.Tensor,
    z_context: Optional[torch.Tensor] = None,
    sim_weight: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 0.05,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Non-contrastive Variance-Invariance-Covariance (VICReg) JEPA Loss.
    
    1. Invariance / Prediction Loss: MSE between predicted target and EMA target representation.
    2. Variance Regularization: Forces standard deviation of each latent dimension >= gamma.
    3. Covariance Decorrelation: Penalizes off-diagonal covariance to prevent representational collapse.
    """
    # 1. Prediction / Invariance Loss
    sim_loss = F.mse_loss(z_target_pred, z_target_true)

    # 2. Variance Loss on representations
    z_to_reg = z_target_pred if z_context is None else torch.cat([z_target_pred, z_context], dim=0)
    std_z = torch.sqrt(torch.var(z_to_reg, dim=0, unbiased=False) + eps)
    var_loss = torch.mean(F.relu(gamma - std_z))

    # 3. Covariance Decorrelation Loss
    batch_size, latent_dim = z_to_reg.shape
    if batch_size > 1:
        z_centered = z_to_reg - torch.mean(z_to_reg, dim=0, keepdim=True)
        cov_mat = (z_centered.T @ z_centered) / (batch_size - 1)
        off_diag = cov_mat - torch.diag(torch.diag(cov_mat))
        cov_loss = torch.sum(off_diag.pow(2)) / max(latent_dim, 1)
    else:
        cov_loss = torch.tensor(0.0, device=z_target_pred.device)

    total_loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    return total_loss

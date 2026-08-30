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
    - Target Encoder E_phi (updated via Exponential Moving Average, strictly deterministic in eval mode)
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
        self.target_encoder.eval()

        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )

        self.register_buffer("precision_matrix", torch.eye(latent_dim))
        self.register_buffer("residual_mean", torch.zeros(latent_dim))
        self.register_buffer("precision_fitted", torch.tensor(False, dtype=torch.bool))

    def train(self, mode: bool = True) -> TSJEPAModel:
        """Override train to ensure the target encoder strictly remains in eval mode."""
        super().train(mode)
        self.target_encoder.eval()
        return self

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
            # Target encoder is always strictly deterministic (eval mode, no grad)
            self.target_encoder.eval()
            with torch.no_grad():
                z_target_true = self.target_encoder(target_windows)

        return z_context, z_target_true, z_target_pred

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        """Update target encoder weights and buffers via Exponential Moving Average (EMA)."""
        m = self.ema_decay if decay is None else decay
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.data)
        for buf_q, buf_k in zip(self.context_encoder.buffers(), self.target_encoder.buffers()):
            buf_k.data.copy_(buf_q.data)

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical residual covariance matrix for Mahalanobis discrepancy using batched accumulation."""
        self.eval()
        n_samples = len(context_windows)
        residuals_list = []

        for i in range(0, n_samples, batch_size):
            ctx_b = context_windows[i : i + batch_size]
            tgt_b = target_windows[i : i + batch_size]
            z_ctx = self.context_encoder(ctx_b)
            z_pred = self.predictor(z_ctx)
            z_obs = self.target_encoder(tgt_b)
            residuals_list.append(z_obs - z_pred)

        residuals = torch.cat(residuals_list, dim=0)
        mean_res = residuals.mean(dim=0, keepdim=True)
        self.residual_mean.copy_(mean_res.squeeze(0))
        residuals_centered = residuals - mean_res
        cov = (residuals_centered.T @ residuals_centered) / max(len(residuals) - 1, 1)
        cov_reg = cov + reg * torch.eye(self.latent_dim, device=cov.device)
        self.precision_matrix.copy_(torch.linalg.pinv(cov_reg))
        self.precision_fitted.copy_(torch.tensor(True, dtype=torch.bool))

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
            if use_mahalanobis:
                if not bool(self.precision_fitted.item()):
                    raise RuntimeError(
                        "Mahalanobis discrepancy requested (use_mahalanobis=True), "
                        "but the precision matrix has not been fitted! "
                        "Call fit_mahalanobis_covariance() before inference."
                    )
                diff_centered = diff - self.residual_mean
                mahal = torch.sum((diff_centered @ self.precision_matrix) * diff_centered, dim=-1)
                return torch.sqrt(torch.clamp(mahal, min=1e-8))
            else:
                return torch.linalg.norm(diff, dim=-1)


def _vicreg_branch_loss(
    z: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute variance and covariance penalties for a single representation branch."""
    batch_size, latent_dim = z.shape
    std_z = torch.sqrt(torch.var(z, dim=0, unbiased=False) + eps)
    var_loss = torch.mean(F.relu(gamma - std_z))

    if batch_size > 1:
        z_centered = z - torch.mean(z, dim=0, keepdim=True)
        cov_mat = (z_centered.T @ z_centered) / (batch_size - 1)
        off_diag = cov_mat - torch.diag(torch.diag(cov_mat))
        cov_loss = torch.sum(off_diag.pow(2)) / max(latent_dim, 1)
    else:
        cov_loss = torch.tensor(0.0, device=z.device)

    return var_loss, cov_loss


def jepa_vicreg_loss(
    z_target_pred: torch.Tensor,
    z_target_true: torch.Tensor,
    z_context: Optional[torch.Tensor] = None,
    sim_weight: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 0.5,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Non-contrastive Branch-Wise Variance-Invariance-Covariance (VICReg) JEPA Loss.
    
    1. Invariance / Prediction Loss: MSE between predicted target and EMA target representation.
    2. Variance Regularization: Enforces std(z) >= gamma on each representation branch independently.
    3. Covariance Decorrelation: Penalizes off-diagonal covariance on each branch independently.
    """
    # 1. Prediction / Invariance Loss
    sim_loss = F.mse_loss(z_target_pred, z_target_true)

    # 2. Branch-wise Variance and Covariance Loss (Independent calculation avoids mean-shift loophole)
    var_pred, cov_pred = _vicreg_branch_loss(z_target_pred, gamma=gamma, eps=eps)

    if z_context is not None:
        var_ctx, cov_ctx = _vicreg_branch_loss(z_context, gamma=gamma, eps=eps)
        var_loss = 0.5 * (var_pred + var_ctx)
        cov_loss = 0.5 * (cov_pred + cov_ctx)
    else:
        var_loss = var_pred
        cov_loss = cov_pred

    total_loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    return total_loss

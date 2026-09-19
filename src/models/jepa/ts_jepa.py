"""Time-Series Joint Embedding Predictive Architecture (TS-JEPA).

Eliminates the dependency on artificial synthetic anomaly injections by training
the encoder to predict future latent states directly in representation space,
regularized with non-contrastive VICReg (Variance-Invariance-Covariance) loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase


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


class TSJEPAModel(JEPABase):
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
        self.target_encoder = self.init_target_encoder(context_encoder)

        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )

        self.register_mahalanobis_buffers(latent_dim)

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

    def compute_objective(self, ctx, tgt, config, **kwargs):
        """Endpoint JEPA VICReg loss."""
        z_ctx, z_tgt_true, z_tgt_pred = self.forward(ctx, tgt)
        loss = jepa_vicreg_loss(
            z_target_pred=z_tgt_pred,
            z_target_true=z_tgt_true,
            z_context=z_ctx,
            sim_weight=config.vicreg_sim_weight,
            var_weight=config.vicreg_var_weight,
            cov_weight=config.vicreg_cov_weight,
        )
        return loss, {"loss": float(loss.item())}

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows,
        target_windows,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical residual covariance matrix for Mahalanobis discrepancy using batched accumulation.

        Accepts numpy arrays or tensors; data is transferred to the model device
        in batches to avoid peak GPU memory spikes.
        """
        from src.models._jepa_utils import fit_covariance_batched

        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_pred = self.predictor(z_ctx)
            z_obs = self.target_encoder(tgt_b)
            return z_obs - z_pred

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


def log_det_whitening_loss(
    z: torch.Tensor,
    beta: float = 0.05,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Log-Determinant Volume Regularization (Streamlined VICReg Option A).

    Maximizing ln det(Sigma + eps*I) simultaneously:
    1. Enforces non-zero variance on all channels (analytic log-barrier against collapse).
    2. Enforces mutual decorrelation (Hadamard's inequality: det(Sigma) <= prod Sigma_jj,
       achieving equality iff all off-diagonal covariances are zero).

    Args:
        z: Latent batch representations (B, D).
        beta: Regularization weight (default: 0.05).
        eps: Diagonal regularizer for numerical stability (default: 1e-4).

    Returns:
        Scalar penalty: - beta / D * ln det(Sigma + eps*I).
    """
    B, D = z.shape
    if B <= 1:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    # Centered batch representation
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1)  # (D, D)

    # Regularized covariance: Sigma + eps * I
    eye = torch.eye(D, device=z.device, dtype=z.dtype)
    cov_reg = cov + eps * eye

    # Stable log-determinant via Cholesky decomposition:
    # ln det(A) = 2 * sum(ln diag(L)) where A = L L^T
    try:
        L = torch.linalg.cholesky(cov_reg)
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
    except RuntimeError:
        # Fallback to slogdet if Cholesky encounters near-singular jitter
        sign, log_det = torch.linalg.slogdet(cov_reg)
        if sign <= 0:
            log_det = torch.tensor(0.0, device=z.device, dtype=z.dtype)

    # Normalize by dimension D: penalty = - (beta / D) * log_det
    return - (beta / float(D)) * log_det


def jepa_vicreg_loss(
    z_target_pred: torch.Tensor,
    z_target_true: torch.Tensor,
    z_context: Optional[torch.Tensor] = None,
    sim_weight: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 0.5,
    gamma: float = 1.0,
    eps: float = 1e-4,
    use_log_det: bool = True,
) -> torch.Tensor:
    """Streamlined Single-Objective JEPA Loss with Log-Det Whitening (Option A).
    
    1. Prediction / Invariance: Exact MSE ||z_pred - z_tgt||^2.
    2. Representation Geometry: Unified Log-Det volume maximization on z_context & z_target_pred.
    """
    # 1. Prediction / Invariance Loss
    sim_loss = F.mse_loss(z_target_pred, z_target_true)

    if not use_log_det:
        # Classical 3-penalty VICReg fallback
        var_pred, cov_pred = _vicreg_branch_loss(z_target_pred, gamma=gamma, eps=eps)
        if z_context is not None:
            var_ctx, cov_ctx = _vicreg_branch_loss(z_context, gamma=gamma, eps=eps)
            var_loss = 0.5 * (var_pred + var_ctx)
            cov_loss = 0.5 * (cov_pred + cov_ctx)
        else:
            var_loss = var_pred
            cov_loss = cov_pred
        return sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss

    # Option A: Single unified log-det regularization (replaces variance hinge + covariance Frobenius sum)
    log_det_pred = log_det_whitening_loss(z_target_pred, beta=0.05, eps=eps)
    log_det_ctx = (
        log_det_whitening_loss(z_context, beta=0.05, eps=eps)
        if z_context is not None
        else torch.tensor(0.0, device=z_target_pred.device)
    )

    total_loss = sim_loss + 0.5 * (log_det_pred + log_det_ctx)
    return total_loss

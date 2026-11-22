"""Minimum Description Length (MDL) Neural Entropy Compressor JEPA (MDLCompressorJEPA).

Rooted in Algorithmic Information Theory and Kolmogorov Complexity:
1. Context Encoder maps x_ctx to a Gaussian Conditional Prior N(mu_ctx, diag(sigma_ctx^2)).
2. Target Encoder maps x_tgt to latent embedding z_tgt.
3. Description Length in Bits:
       L_bits(z_tgt | z_ctx) = 1/(2 * ln 2) * sum_d [ ln(2*pi*sigma_d^2) + (z_tgt,d - mu_d)^2 / sigma_d^2 ]
4. Anomaly Scoring:
       Point Innovation (Point Anomaly):     S_innov = sqrt(sum_d (z_tgt,d - mu_d)^2 / sigma_d^2)
       Total Description Length in Bits:     S_bits  = L_bits(z_tgt | z_ctx)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss


class EntropyPriorHead(nn.Module):
    """Predicts conditional Gaussian prior mean mu and log_var in R^D."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # [mu, log_var]
        )

    def forward(self, z_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(z_ctx)
        mu = out[:, : self.latent_dim]
        log_var = torch.clamp(out[:, self.latent_dim :], min=-6.0, max=6.0)
        return mu, log_var


class MDLCompressorJEPAModel(JEPABase):
    """Minimum Description Length Compressor JEPA."""

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

        self.prior_head = EntropyPriorHead(
            latent_dim=latent_dim,
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
        mu, log_var = self.prior_head(z_ctx)

        if target_windows is None:
            return z_ctx, None, mu, log_var

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.target_encoder(target_windows)

        return z_ctx, z_tgt, mu, log_var

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        nll_weight: float = 1.0,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute Negative Log-Likelihood (MDL Bitrate) + VICReg."""
        z_ctx, z_tgt, mu, log_var = self.forward(ctx, tgt)

        var = torch.exp(log_var)  # (B, D)

        # Negative Log-Likelihood / Conditional Description Length
        # NLL = 0.5 * sum [ log(2*pi*var) + (z_tgt - mu)^2 / var ]
        nll = 0.5 * torch.sum(np.log(2.0 * np.pi) + log_var + ((z_tgt - mu) ** 2) / (var + 1e-6), dim=-1)
        loss_nll = torch.mean(nll)

        # Bitrate in bits (divided by ln 2)
        bitrate = loss_nll / np.log(2.0)

        # Representation Non-Collapse (VICReg)
        std_c = torch.sqrt(torch.var(z_ctx, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)

        total_loss = (
            nll_weight * loss_nll
            + var_weight * var_c
            + cov_weight * cov_c
        )

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_nll": float(loss_nll.item()),
            "bitrate_bits": float(bitrate.item()),
            "mean_var": float(torch.mean(var).item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute Description Length in Bits as exact anomaly score."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        mu, log_var = self.prior_head(z_ctx)
        var = torch.exp(log_var)

        # Exact Description Length in Bits: L_bits(z_tgt | z_ctx)
        nll_bits = 0.5 * torch.sum(
            np.log(2.0 * np.pi) + log_var + ((z_tgt - mu) ** 2) / (var + 1e-6),
            dim=-1,
        ) / np.log(2.0)

        return nll_bits

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            mu, log_var = self.prior_head(z_ctx)
            # Normalized innovation
            std = torch.exp(0.5 * log_var)
            return (z_tgt - mu) / (std + 1e-6)

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


MDLCompressorJEPA = MDLCompressorJEPAModel

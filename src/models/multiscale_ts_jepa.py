"""Multi-Horizon Hierarchical Joint Embedding Predictive Architecture (MultiScale-TS-JEPA).

Captures dynamics at multiple temporal granularities simultaneously by forecasting
future latent representations across multiple prediction horizons (e.g. 16, 64, 128 steps).
Combines sensitivity to abrupt transient spikes with long-range drift detection.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tcn_encoder import HybridTCNEncoder
from src.models.ts_jepa import LatentPredictor, jepa_vicreg_loss


class MultiScaleTSJEPA(nn.Module):
    """Multi-Horizon Hierarchical TS-JEPA."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        horizons: Tuple[int, ...] = (16, 64),
        filters: int = 48,
        tcn_layers: int = 6,
        kernel_size: int = 5,
        dropout: float = 0.20,
        predictor_hidden_dim: int = 64,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.horizons = horizons
        self.ema_decay = ema_decay

        # Shared Context Encoder
        self.context_encoder = HybridTCNEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            filters=filters,
            tcn_layers=tcn_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # EMA Target Encoder
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Multi-Head Latent Predictors for each horizon
        self.predictors = nn.ModuleDict(
            {
                str(h): LatentPredictor(
                    latent_dim=latent_dim,
                    hidden_dim=predictor_hidden_dim,
                    num_layers=2,
                    dropout=dropout,
                )
                for h in horizons
            }
        )

        for h in horizons:
            self.register_buffer(f"precision_matrix_{h}", torch.eye(latent_dim))
            self.register_buffer(f"residual_mean_{h}", torch.zeros(latent_dim))
        self.precision_fitted = False

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Optional[torch.Tensor]], Dict[int, torch.Tensor]]:
        """Forward pass.
        
        Returns:
            z_ctx: (B, D)
            z_tgt_true_dict: {horizon: (B, D) or None}
            z_tgt_pred_dict: {horizon: (B, D)}
        """
        z_ctx = self.context_encoder(context_windows)
        z_tgt_pred_dict = {}
        for h in self.horizons:
            z_tgt_pred_dict[h] = self.predictors[str(h)](z_ctx)

        z_tgt_true_dict = {}
        if target_windows is not None:
            with torch.no_grad():
                for h in self.horizons:
                    # Slice target up to horizon h
                    h_len = min(h, target_windows.size(1))
                    tgt_slice = target_windows[:, :h_len, :]
                    z_tgt_true_dict[h] = self.target_encoder(tgt_slice)
        else:
            for h in self.horizons:
                z_tgt_true_dict[h] = None

        return z_ctx, z_tgt_true_dict, z_tgt_pred_dict

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        """EMA update of target encoder."""
        m = self.ema_decay if decay is None else decay
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.data)

    def compute_multiscale_loss(
        self,
        z_tgt_pred_dict: Dict[int, torch.Tensor],
        z_tgt_true_dict: Dict[int, torch.Tensor],
        z_ctx: Optional[torch.Tensor] = None,
        cov_weight: float = 0.5,
    ) -> torch.Tensor:
        """Sum of VICReg losses across all horizons."""
        total_loss = 0.0
        for h in self.horizons:
            loss_h = jepa_vicreg_loss(
                z_tgt_pred_dict[h],
                z_tgt_true_dict[h],
                z_context=z_ctx,
                cov_weight=cov_weight,
            )
            total_loss = total_loss + loss_h
        return total_loss / len(self.horizons)

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical residual covariance matrices for each horizon."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        for h in self.horizons:
            h_len = min(h, target_windows.size(1))
            tgt_slice = target_windows[:, :h_len, :]
            z_obs = self.target_encoder(tgt_slice)
            z_pred = self.predictors[str(h)](z_ctx)
            residuals = z_obs - z_pred

            mean_res = residuals.mean(dim=0, keepdim=True)
            getattr(self, f"residual_mean_{h}").copy_(mean_res.squeeze(0))
            residuals_centered = residuals - mean_res
            cov = (residuals_centered.T @ residuals_centered) / max(len(residuals) - 1, 1)
            cov_reg = cov + reg * torch.eye(self.latent_dim, device=cov.device)
            getattr(self, f"precision_matrix_{h}").copy_(torch.linalg.pinv(cov_reg))
        self.precision_fitted = True

    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        """Compute fused multi-horizon discrepancy.
        
        Returns:
            Tensor of shape (B,) representing the max standardized multi-horizon discrepancy.
        """
        self.eval()
        scores_list = []
        with torch.no_grad():
            z_ctx = self.context_encoder(context_windows)
            for h in self.horizons:
                h_len = min(h, observed_target_windows.size(1))
                tgt_slice = observed_target_windows[:, :h_len, :]
                z_obs = self.target_encoder(tgt_slice)
                z_pred = self.predictors[str(h)](z_ctx)
                diff = z_obs - z_pred
                if use_mahalanobis and self.precision_fitted:
                    mean_res = getattr(self, f"residual_mean_{h}")
                    prec_mat = getattr(self, f"precision_matrix_{h}")
                    diff_centered = diff - mean_res
                    mahal = torch.sum((diff_centered @ prec_mat) * diff_centered, dim=-1)
                    score_h = torch.sqrt(torch.clamp(mahal, min=1e-8))
                else:
                    score_h = torch.linalg.norm(diff, dim=-1)
                scores_list.append(score_h)

        # Fused max across horizons
        stacked = torch.stack(scores_list, dim=-1)  # (B, num_horizons)
        return torch.max(stacked, dim=-1).values

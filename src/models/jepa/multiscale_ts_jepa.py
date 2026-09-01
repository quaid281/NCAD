"""Multi-Horizon Hierarchical Joint Embedding Predictive Architecture (MultiScale-TS-JEPA).

Captures dynamics at multiple temporal granularities simultaneously by forecasting
future latent representations across multiple prediction horizons (e.g. 16, 64, 128 steps).
Combines sensitivity to abrupt transient spikes with long-range drift detection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.ts_jepa import LatentPredictor, jepa_vicreg_loss


class MultiScaleTSJEPA(JEPABase):
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
        if not horizons:
            raise ValueError("horizons must be a non-empty sequence of positive integers.")
        for h in horizons:
            if not isinstance(h, int) or h <= 0:
                raise ValueError(f"Each horizon must be a positive integer, got {h}.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.horizons = tuple(horizons)
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

        # EMA Target Encoder (strictly deterministic eval mode)
        self.target_encoder = self.init_target_encoder(self.context_encoder)

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
        self.register_buffer("precision_fitted", torch.tensor(False, dtype=torch.bool))

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
            self.target_encoder.eval()
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
        context_windows,
        target_windows,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical residual covariance matrices for each horizon.

        Accepts numpy arrays or tensors; data is transferred in batches.
        """
        from src.models._jepa_utils import _model_device, to_device_tensor

        self.eval()
        device = _model_device(self)
        n_samples = len(context_windows)

        # Accumulate per-horizon residual sums and outer-product sums incrementally
        for h in self.horizons:
            h_len = min(h, target_windows.shape[1])
            res_sum = torch.zeros(self.latent_dim, device=device, dtype=torch.float32)
            outer_sum = torch.zeros((self.latent_dim, self.latent_dim), device=device, dtype=torch.float32)
            count = 0

            for i in range(0, n_samples, batch_size):
                ctx_b = to_device_tensor(context_windows[i : i + batch_size], self)
                tgt_b = to_device_tensor(target_windows[i : i + batch_size], self)
                z_ctx = self.context_encoder(ctx_b)
                tgt_slice = tgt_b[:, :h_len, :]
                z_obs = self.target_encoder(tgt_slice)
                z_pred = self.predictors[str(h)](z_ctx)
                residuals = (z_obs - z_pred).reshape(-1, self.latent_dim).to(torch.float32)
                res_sum += residuals.sum(dim=0)
                outer_sum += residuals.T @ residuals
                count += residuals.shape[0]

            if count == 0:
                continue
            mean_res = res_sum / count
            getattr(self, f"residual_mean_{h}").copy_(mean_res)
            cov = (outer_sum - count * torch.outer(mean_res, mean_res)) / max(count - 1, 1)
            cov_reg = cov + reg * torch.eye(self.latent_dim, device=device)
            getattr(self, f"precision_matrix_{h}").copy_(torch.linalg.pinv(cov_reg))
        self.precision_fitted.fill_(True)

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
        if use_mahalanobis and not bool(self.precision_fitted.item()):
            raise RuntimeError("Mahalanobis scoring requested, but covariance has not been fitted.")

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
                if use_mahalanobis:
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

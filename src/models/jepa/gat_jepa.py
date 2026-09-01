"""Relational GAT-JEPA: Spatial-Temporal Relational Graph Attention Network within TS-JEPA.

Combines:
1. Dynamic inter-sensor graph attention layers to model pairwise multi-channel topological dependencies.
2. Causal temporal convolutions for multi-frequency temporal feature extraction.
3. Latent predictive dynamics forecasting the future multi-sensor graph representation state.
4. Non-contrastive VICReg (Variance-Invariance-Covariance) self-supervised training.
5. Optional Covariance-Whitened (Mahalanobis) latent discrepancy scoring.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase
from src.models.encoders.relational_gat_encoder import RelationalGATEncoder
from src.models.jepa.ts_jepa import LatentPredictor, jepa_vicreg_loss


class RelationalGAT_JEPAModel(JEPABase):
    """Spatial-Temporal Relational Graph Attention Network wrapped in TS-JEPA."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        filters: int = 48,
        tcn_layers: int = 3,
        gat_layers: int = 2,
        gat_heads: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.20,
        predictor_hidden_dim: int = 64,
        predictor_layers: int = 2,
        ema_decay: float = 0.995,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        # 1. Active GAT Context Encoder
        self.context_encoder = RelationalGATEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            filters=filters,
            tcn_layers=tcn_layers,
            gat_layers=gat_layers,
            gat_heads=gat_heads,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # 2. EMA Target GAT Encoder
        self.target_encoder = self.init_target_encoder(self.context_encoder)

        # 3. Latent Predictor
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )

        # Optional empirical precision matrix (inverse covariance) for Mahalanobis scoring
        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward pass.
        
        Args:
            context_windows: (batch, context_len, num_sensors)
            target_windows: Optional (batch, target_len, num_sensors)
            
        Returns:
            Tuple of (z_context, z_target_true, z_target_pred)
        """
        z_context = self.context_encoder(context_windows)
        z_target_pred = self.predictor(z_context)

        z_target_true = None
        if target_windows is not None:
            self.target_encoder.eval()
            with torch.no_grad():
                z_target_true = self.target_encoder(target_windows)

        return z_context, z_target_true, z_target_pred


    def compute_loss(
        self,
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        sim_weight: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.5,
    ) -> torch.Tensor:
        """Compute VICReg loss."""
        z_context, z_target_true, z_target_pred = self.forward(context_windows, target_windows)
        return jepa_vicreg_loss(
            z_target_pred=z_target_pred,
            z_target_true=z_target_true,
            z_context=z_context,
            sim_weight=sim_weight,
            var_weight=var_weight,
            cov_weight=cov_weight,
        )

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows,
        target_windows,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical residual covariance matrix for Mahalanobis discrepancy using batched accumulation.

        Accepts numpy arrays or tensors; data is transferred in batches.
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

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        """Compute latent physical prediction discrepancy."""
        self.eval()
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

    @torch.no_grad()
    def encode_context(self, context_windows: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.context_encoder(context_windows)

    @torch.no_grad()
    def encode_target(self, target_windows: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.target_encoder(target_windows)

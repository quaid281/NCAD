"""NCAD-JEPA: Fused Neural Contextual Anomaly Detection with Time-Series Joint-Embedding Predictive Architecture.

This model combines:
1. Dilated Causal TCN Encoder (E_theta) for temporal representation learning.
2. Exponential Moving Average (EMA) Target Encoder (E_phi) for stable latent targets.
3. Latent Predictor (P_psi) mapping context embeddings to expected target embeddings.
4. Unified multi-task objective: Non-contrastive VICReg (invariance, variance, covariance)
   + Contextual Anomaly Injection Contrastive Margin Loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase
from src.models.tcn_encoder import HybridTCNEncoder, contrastive_loss
from src.models.ts_jepa import LatentPredictor, jepa_vicreg_loss


class NCADJEPAModel(JEPABase):
    """Unified NCAD-TCN + TS-JEPA Model."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        filters: int = 48,
        tcn_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.20,
        predictor_hidden_dim: int = 64,
        predictor_layers: int = 2,
        ema_decay: float = 0.995,
        injection_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.injection_loss_weight = injection_loss_weight

        # 1. Base Causal TCN Encoder (Active Gradients)
        self.context_encoder = HybridTCNEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            filters=filters,
            tcn_layers=tcn_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # 2. Target Encoder (EMA updated, no grads, strictly deterministic eval)
        self.target_encoder = self.init_target_encoder(self.context_encoder)

        # 3. Latent Space Dynamics Predictor
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward pass through context encoder and latent predictor.
        
        Args:
            context_windows: (batch, context_len, features)
            target_windows: Optional (batch, target_len, features)
            
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


    def compute_joint_loss(
        self,
        clean_context: torch.Tensor,
        clean_target: torch.Tensor,
        injected_context: torch.Tensor,
        injected_label: torch.Tensor,
        sim_weight: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.05,
        margin: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the combined VICReg + Synthetic Anomaly Contrastive loss.
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # 1. JEPA Predictive Pass
        z_ctx, z_tgt_true, z_tgt_pred = self.forward(clean_context, clean_target)
        loss_jepa = jepa_vicreg_loss(
            z_target_pred=z_tgt_pred,
            z_target_true=z_tgt_true,
            z_context=z_ctx,
            sim_weight=sim_weight,
            var_weight=var_weight,
            cov_weight=cov_weight,
        )

        # 2. NCAD Contrastive Injection Pass
        z_injected = self.context_encoder(injected_context)
        loss_contrastive = contrastive_loss(
            z_injected,
            z_ctx,
            injected_label,
            margin=margin,
        )

        # Total combined loss
        total_loss = loss_jepa + self.injection_loss_weight * loss_contrastive

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "loss_jepa": float(loss_jepa.item()),
            "loss_contrastive": float(loss_contrastive.item()),
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
    ) -> torch.Tensor:
        """Compute latent physical dynamics prediction error ||E(target) - P(E(ctx))||_2."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_pred = self.predictor(z_ctx)
        z_obs = self.target_encoder(observed_target_windows)
        discrepancy = torch.linalg.norm(z_obs - z_pred, dim=-1)
        return discrepancy

    @torch.no_grad()
    def encode_context(self, context_windows: torch.Tensor) -> torch.Tensor:
        """Encode context windows into latent space."""
        self.eval()
        return self.context_encoder(context_windows)

    @torch.no_grad()
    def encode_target(self, target_windows: torch.Tensor) -> torch.Tensor:
        """Encode target windows using target encoder."""
        self.eval()
        return self.target_encoder(target_windows)

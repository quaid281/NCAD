"""NCAD-Flow-JEPA: Fused Neural Contextual Anomaly Detection with Flow Matching JEPA.

This model combines:
1. Dilated Causal TCN Encoder (E_theta) for temporal representation learning.
2. Exponential Moving Average (EMA) Target Encoder (E_phi) for stable latent targets.
3. Continuous Flow Velocity Predictor (v_psi) trained via Optimal Transport Flow Matching.
4. Unified multi-task objective: Flow Matching VICReg (velocity MSE + variance/covariance
   regularization) + Contextual Anomaly Injection Contrastive Margin Loss.

The flow matching objective learns a continuous velocity field v(z_t, t, z_ctx) that
transports the prior distribution to the target latent distribution. The contrastive
injection objective teaches the encoder to be sensitive to contextual anomalies,
mirroring the legacy NCAD approach.

This is the flow-matching complement to NCADJEPAModel (which uses endpoint prediction).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models._jepa_utils import JEPABase
from src.models.encoders.tcn_encoder import HybridTCNEncoder, contrastive_loss
from src.models.jepa.flow_ts_jepa import (
    FlowLatentPredictor,
    _get_chebyshev_collocation_nodes,
    flow_matching_vicreg_loss,
)


class NCADFlowJEPAModel(JEPABase):
    """Unified NCAD-TCN + Flow Matching TS-JEPA Model."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        filters: int = 48,
        tcn_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.20,
        predictor_hidden_dim: int = 64,
        predictor_layers: int = 3,
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

        # 3. Continuous Flow Velocity Predictor
        self.flow_predictor = FlowLatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )

        # 4. Mahalanobis-whitened scoring buffers
        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        z_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for flow matching training.

        Args:
            context_windows: (B, L_ctx, C)
            target_windows: Optional (B, L_tgt, C)
            t: Optional continuous time in [0, 1], shape (B,). If None, sampled U(0, 1).
            z_noise: Optional Gaussian prior at t=0, shape (B, D). If None, sampled N(0, I).

        Returns:
            Tuple of (z_ctx, z_tgt_true, v_pred, v_target)
        """
        B = context_windows.size(0)
        device = context_windows.device

        z_ctx = self.context_encoder(context_windows)

        if target_windows is None:
            return z_ctx, None, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt_true = self.target_encoder(target_windows)

        if t is None:
            t = torch.rand(B, device=device, dtype=context_windows.dtype)

        if z_noise is None:
            z_noise = torch.randn(B, self.latent_dim, device=device, dtype=context_windows.dtype)

        t_expand = t.view(B, 1)
        z_t = (1.0 - t_expand) * z_noise + t_expand * z_tgt_true
        v_target = z_tgt_true - z_noise
        v_pred = self.flow_predictor(z_t, t, z_ctx)

        return z_ctx, z_tgt_true, v_pred, v_target

    def compute_objective(self, ctx, tgt, config, *, injector=None, full_batch=None, **kwargs):
        """Joint flow matching + contrastive injection loss.

        Requires ``injector`` and ``full_batch`` to produce injected windows.
        In eval mode, uses deterministic t=0.5 and zero prior z_0=0 for stable
        validation, and a deterministic validation injector.
        """
        if injector is None or full_batch is None:
            raise ValueError(
                "NCADFlowJEPAModel.compute_objective requires 'injector' and 'full_batch' "
                "keyword arguments for contextual anomaly injection."
            )

        # 1. Flow Matching pass (principal objective)
        if not self.training:
            B = ctx.size(0)
            device = ctx.device
            dtype = ctx.dtype
            t_val = torch.full((B,), 0.5, device=device, dtype=dtype)
            z_zero = torch.zeros(B, self.latent_dim, device=device, dtype=dtype)
            z_ctx, z_tgt_true, v_pred, v_target = self.forward(ctx, tgt, t=t_val, z_noise=z_zero)
        else:
            z_ctx, z_tgt_true, v_pred, v_target = self.forward(ctx, tgt)

        loss_flow, flow_metrics = flow_matching_vicreg_loss(
            v_pred=v_pred, v_target=v_target, z_ctx=z_ctx, z_tgt_true=z_tgt_true,
            flow_weight=config.vicreg_sim_weight,
            var_weight=config.vicreg_var_weight,
            cov_weight=config.vicreg_cov_weight,
        )

        # 2. NCAD Contrastive Injection pass (auxiliary objective)
        injected_full, labels = injector.inject_batch(full_batch, config.context_size)
        injected_tensor = torch.from_numpy(injected_full).float().to(ctx.device)
        label_tensor = torch.from_numpy(labels).float().to(ctx.device)

        z_injected = self.context_encoder(injected_tensor)
        loss_contrastive = contrastive_loss(z_injected, z_ctx, label_tensor)

        total_loss = loss_flow + self.injection_loss_weight * loss_contrastive

        metrics = dict(flow_metrics)
        metrics["total_loss"] = float(total_loss.item())
        metrics["loss_flow"] = float(loss_flow.item())
        metrics["loss_contrastive"] = float(loss_contrastive.item())
        return total_loss, metrics

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows,
        target_windows,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical vector field residual covariance for Mahalanobis-whitened scoring."""
        from src.models._jepa_utils import fit_covariance_batched

        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            B_b = ctx_b.size(0)
            t_mid = torch.full((B_b,), 0.5, device=ctx_b.device, dtype=ctx_b.dtype)
            z_mid = 0.5 * z_tgt
            v_pred = self.flow_predictor(z_mid, t_mid, z_ctx)
            return v_pred - z_tgt

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
        collocation: str = "midpoint",
    ) -> torch.Tensor:
        """Compute deterministic OT flow discrepancy along Chebyshev collocation nodes.

        Evaluates the velocity field consistency along the straight OT-CFM trajectory
        from z_0=0 to z_1=E_phi(x_tgt), at deterministic quadrature nodes.

        Returns:
            Discrepancy scores of shape (B,).
        """
        if use_mahalanobis and not bool(self.precision_fitted.item()):
            raise RuntimeError(
                "Mahalanobis discrepancy requested (use_mahalanobis=True), "
                "but the precision matrix has not been fitted! "
                "Call fit_mahalanobis_covariance() before inference."
            )

        self.eval()
        B = context_windows.size(0)
        device = context_windows.device
        dtype = context_windows.dtype

        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        nodes, weights = _get_chebyshev_collocation_nodes(collocation, device=device, dtype=dtype)
        total_score = torch.zeros(B, device=device, dtype=dtype)

        for t_val, w_val in zip(nodes, weights):
            t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)
            z_node = t_val * z_tgt
            v_pred = self.flow_predictor(z_node, t_tensor, z_ctx)
            diff = v_pred - z_tgt

            if use_mahalanobis:
                diff_c = diff - self.residual_mean
                m_dist = torch.sum((diff_c @ self.precision_matrix) * diff_c, dim=-1)
                node_score = torch.sqrt(torch.clamp(m_dist, min=1e-8))
            else:
                node_score = torch.linalg.norm(diff, dim=-1)

            total_score = total_score + w_val * node_score

        return total_score

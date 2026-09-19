"""Prototype-Graph Transition Joint Embedding Predictive Architecture (PrototypeGraphJEPA).

A clean, interpretable architecture that completely decouples Point and Contextual anomalies:
1. Partitions latent space into K learnable Anchor Prototypes C = [c_1, ..., c_K] in R^{K x D}
   and a Markov Transition Matrix T in R^{K x K} where T_ij = P(regime_j | regime_i).
2. Point Anomaly Score (Anchor Distance):
       S_point = min_k ||z_tgt - c_k||_2
       Directly measures whether z_tgt is in an unphysical void far from all known operational regimes.
3. Contextual Anomaly Score (Transition Surprise):
       p_pred = w(z_ctx) @ T,    S_context = -sum_k w_k(z_tgt) * log(p_pred_k)
       Directly measures whether the transition from the current context to the target regime is illegal.
4. Total Anomaly Score:
       S = S_point + beta * S_context
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss
from src.models.jepa.hankel_moments import HankelMomentTracker


class PrototypeGraphModule(nn.Module):
    """Maintains K anchor prototypes and a learnable Markov transition matrix."""

    def __init__(
        self,
        num_prototypes: int = 16,
        latent_dim: int = 32,
        temperature: float = 0.5,
        use_hankel: bool = True,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.use_hankel = use_hankel

        # Learnable anchor prototypes C in R^{K x D}
        # Initialized with unit sphere uniform distribution
        init_c = torch.randn(num_prototypes, latent_dim)
        init_c = F.normalize(init_c, p=2, dim=-1)
        self.prototypes = nn.Parameter(init_c)

        # Learnable transition logits T_logits in R^{K x K} (initialized near identity + uniform)
        init_T = torch.eye(num_prototypes) * 2.0 + 0.1 * torch.randn(num_prototypes, num_prototypes)
        self.transition_logits = nn.Parameter(init_T)

        # Algebraic Hankel moment tracker for regime non-collapse (Chapter 7)
        hankel_k = min(num_prototypes, 4)
        self.hankel_tracker = HankelMomentTracker(n_regimes=hankel_k) if use_hankel else None


    @property
    def transition_matrix(self) -> torch.Tensor:
        """Row-normalized Markov transition matrix T in [0, 1]^{K x K}."""
        return F.softmax(self.transition_logits, dim=-1)

    def compute_soft_assignments(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute soft prototype assignment probabilities and squared distances.
        
        Args:
            z: Latent tensor, shape (B, D)
            
        Returns:
            w: Soft assignment distribution in Delta^{K-1}, shape (B, K)
            sq_dists: Squared Euclidean distances to all prototypes, shape (B, K)
        """
        # z: (B, 1, D), prototypes: (1, K, D)
        diffs = z.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (B, K, D)
        sq_dists = torch.sum(diffs ** 2, dim=-1)               # (B, K)

        # Soft assignments via softmax over negative scaled distance
        w = F.softmax(-sq_dists / self.temperature, dim=-1)    # (B, K)
        return w, sq_dists


class PrototypeGraphJEPAModel(JEPABase):
    """Prototype-Graph Transition JEPA (PrototypeGraphJEPA)."""

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        num_prototypes: int = 16,
        temperature: float = 0.5,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.num_prototypes = num_prototypes
        self.ema_decay = ema_decay

        self.target_encoder = self.init_target_encoder(context_encoder)

        self.prototype_graph = PrototypeGraphModule(
            num_prototypes=num_prototypes,
            latent_dim=latent_dim,
            temperature=temperature,
        )

        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        z_ctx = self.context_encoder(context_windows)
        w_ctx, sq_dists_ctx = self.prototype_graph.compute_soft_assignments(z_ctx)

        if target_windows is None:
            return z_ctx, None, w_ctx, sq_dists_ctx

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.target_encoder(target_windows)

        return z_ctx, z_tgt, w_ctx, sq_dists_ctx

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        trans_weight: float = 1.0,
        anchor_weight: float = 0.0,
        diversity_weight: float = 0.0,
        cov_weight: float = 0.0,
        var_weight: float = 0.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute prototype Markov transition prediction loss.
        Prototypes lie on the hypersphere, eliminating the need for inverse-distance diversity penalties.
        """
        z_ctx, z_tgt, w_ctx, sq_dists_ctx = self.forward(ctx, tgt)
        w_tgt, sq_dists_tgt = self.prototype_graph.compute_soft_assignments(z_tgt)

        # 1. Markov Transition Prediction Loss: cross-entropy between predicted regime and true regime
        T = self.prototype_graph.transition_matrix  # (K, K)
        p_pred = torch.matmul(w_ctx, T)            # (B, K)
        loss_trans = -torch.mean(torch.sum(w_tgt * torch.log(p_pred + 1e-8), dim=-1))
        total_loss = trans_weight * loss_trans

        # Diagnostic anchor distance
        loss_anchor = torch.mean(torch.min(sq_dists_ctx, dim=-1)[0] + torch.min(sq_dists_tgt, dim=-1)[0])
        if anchor_weight > 0:
            total_loss = total_loss + anchor_weight * loss_anchor

        # Diagnostic diversity & Hankel separation
        prototypes = self.prototype_graph.prototypes  # (K, D)
        proto_diffs = prototypes.unsqueeze(0) - prototypes.unsqueeze(1)  # (K, K, D)
        proto_dists = torch.norm(proto_diffs, dim=-1) + torch.eye(self.num_prototypes, device=ctx.device)
        loss_diversity = torch.mean(1.0 / (proto_dists + 1e-3))

        if self.prototype_graph.hankel_tracker is not None:
            hankel_k = self.prototype_graph.hankel_tracker.K
            loss_hankel = self.prototype_graph.hankel_tracker.compute_regime_separation_loss(prototypes[:hankel_k])
        else:
            loss_hankel = torch.tensor(0.0, device=ctx.device)

        if diversity_weight > 0:
            total_loss = total_loss + diversity_weight * (loss_diversity + 0.1 * loss_hankel)

        if var_weight > 0 or cov_weight > 0:
            std_c = torch.sqrt(torch.var(z_ctx, dim=0, unbiased=False) + eps)
            var_c = torch.mean(F.relu(gamma - std_c))
            cov_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)
            total_loss = total_loss + var_weight * var_c + cov_weight * cov_c

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_trans": float(loss_trans.item()),
            "loss_anchor": float(loss_anchor.item()),
            "loss_diversity": float(loss_diversity.item()),
            "loss_hankel": float(loss_hankel.item()),
            "mean_min_proto_dist": float(torch.mean(torch.sqrt(torch.min(sq_dists_ctx, dim=-1)[0])).item()),
        }

        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        beta_context: float = 0.50,
        **kwargs,
    ) -> torch.Tensor:
        """Compute decoupled dual anomaly score: Point (Anchor Distance) + Contextual (Transition Surprise)."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        w_ctx, _ = self.prototype_graph.compute_soft_assignments(z_ctx)
        w_tgt, sq_dists_tgt = self.prototype_graph.compute_soft_assignments(z_tgt)

        # 1. Point Anomaly Score: Minimum Euclidean distance to nearest prototype well
        score_point = torch.sqrt(torch.min(sq_dists_tgt, dim=-1)[0] + 1e-8)  # (B,)

        # 2. Contextual Anomaly Score: Markov transition surprise from context to target
        T = self.prototype_graph.transition_matrix  # (K, K)
        p_pred = torch.matmul(w_ctx, T)            # (B, K)
        score_context = -torch.sum(w_tgt * torch.log(p_pred + 1e-8), dim=-1)  # (B,)

        # Decoupled composite score
        total_score = score_point + beta_context * score_context
        return total_score

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            # Reconstruct expected target from predicted prototype distribution
            w_ctx, _ = self.prototype_graph.compute_soft_assignments(z_ctx)
            T = self.prototype_graph.transition_matrix
            p_pred = torch.matmul(w_ctx, T)  # (B, K)
            z_pred_expected = torch.matmul(p_pred, self.prototype_graph.prototypes)  # (B, D)
            return z_tgt - z_pred_expected

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


PrototypeGraphJEPA = PrototypeGraphJEPAModel

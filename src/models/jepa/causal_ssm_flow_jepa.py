"""Causal State-Space Flow-Matching Joint-Embedding Predictive Architecture (CausalSSMFlowJEPA).

Combines:
1. Bidirectional Selective State-Space sequence modeling (SSM) for O(L) long-range temporal context.
2. Spatial-Temporal Relational Graph Attention (GAT) for learning dynamic causal inter-sensor topologies.
3. Continuous Optimal Transport Conditional Flow Matching (OT-CFM) for velocity field trajectory modeling.
4. Non-contrastive VICReg manifold regularization (variance hinge, covariance decorrelation).
5. Zero-shot counterfactual intervention for root-cause channel attribution.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase
from src.models.encoders.relational_gat_encoder import RelationalGraphAttentionLayer
from src.models.encoders.selective_ssm_encoder import GRUStateSpaceBlock
from src.models.jepa.flow_ts_jepa import FlowLatentPredictor, TimestepEmbedding
from src.models.jepa.ts_jepa import _vicreg_branch_loss


class CausalSSMContextEncoder(nn.Module):
    """Spatial-Temporal Causal Encoder combining Selective SSM temporal modeling with Relational GAT.

    Input shape:  (batch, seq_len, in_channels)
    Output:
        - z_ctx: (batch, latent_dim) global context representation
        - z_nodes: (batch, in_channels, node_dim) per-variable node representations
        - attention_weights: (batch, in_channels, in_channels) dynamic causal adjacency matrix
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        node_dim: int = 32,
        ssm_layers: int = 2,
        gat_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim

        # Per-channel temporal projection: maps each channel's time-series scalar to hidden_dim
        self.channel_proj = nn.Linear(1, hidden_dim)

        # Temporal sequence modeling via stacked Selective SSM / Gated Recurrence blocks
        self.ssm_blocks = nn.ModuleList(
            [GRUStateSpaceBlock(hidden_dim, dropout=dropout) for _ in range(ssm_layers)]
        )

        # Temporal summary pooling head -> node features
        self.temporal_pool = nn.Sequential(
            nn.Linear(hidden_dim * 3, node_dim),
            nn.GELU(),
            nn.LayerNorm(node_dim),
        )

        # Spatial Relational Graph Attention layers across sensor nodes
        self.gat_blocks = nn.ModuleList(
            [
                RelationalGraphAttentionLayer(
                    node_dim=node_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_node_identity=True,
                    max_nodes=max(in_channels, 64),
                )
                for _ in range(gat_layers)
            ]
        )

        # Global context projection from all node features
        self.global_proj = nn.Sequential(
            nn.Linear(in_channels * node_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self, inputs: torch.Tensor, return_graph: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through Causal SSM Encoder.

        Args:
            inputs: Tensor of shape (batch, seq_len, in_channels).
            return_graph: If True, returns (z_ctx, z_nodes, attn_weights).

        Returns:
            z_ctx of shape (batch, latent_dim) if return_graph is False,
            or (z_ctx, z_nodes, attn_weights) if return_graph is True.
        """
        if inputs.ndim != 3:
            raise ValueError(f"Expected 3D inputs (batch, seq_len, channels), got shape {inputs.shape}")

        B, L, C = inputs.shape

        # Step 1: Reshape to treat each channel independently across batch: (B*C, L, 1)
        # inputs: (B, L, C) -> transpose to (B, C, L) -> reshape (B*C, L, 1)
        x_reshaped = inputs.transpose(1, 2).reshape(B * C, L, 1)
        h = self.channel_proj(x_reshaped)  # (B*C, L, hidden_dim)

        for ssm in self.ssm_blocks:
            h = ssm(h)  # (B*C, L, hidden_dim)

        last_f = h[:, -1]
        mean_f = torch.mean(h, dim=1)
        max_f = torch.max(h, dim=1).values
        node_feats = self.temporal_pool(torch.cat([last_f, mean_f, max_f], dim=-1))  # (B*C, node_dim)
        node_feats = node_feats.view(B, C, self.node_dim)  # (B, C, node_dim)

        # Step 2: Spatial Relational GAT
        curr_nodes = node_feats
        last_attn = None
        for gat in self.gat_blocks:
            curr_nodes, last_attn = gat(curr_nodes, return_attention=True)

        # Step 3: Global context projection
        flat_nodes = curr_nodes.reshape(B, C * self.node_dim)
        z_ctx = self.global_proj(flat_nodes)

        if return_graph:
            # Average attention over heads: (B, num_heads, C, C) -> (B, C, C)
            attn_matrix = torch.mean(last_attn, dim=1) if last_attn is not None else None
            return z_ctx, curr_nodes, attn_matrix
        return z_ctx


class CausalSSMFlowJEPA(JEPABase):
    """Causal State-Space Flow-Matching Joint Embedding Predictive Architecture.

    Unifies Causal Relational Graph Attention, Selective SSM sequence context modeling,
    Optimal Transport Conditional Flow Matching (OT-CFM), and Conformal Calibration.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        node_dim: int = 32,
        ssm_layers: int = 2,
        gat_layers: int = 2,
        num_heads: int = 4,
        flow_layers: int = 3,
        dropout: float = 0.10,
        ema_decay: float = 0.996,
        vicreg_weight: float = 0.10,
        graph_sparsity_weight: float = 1e-4,
        n_eval_times: int = 5,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.vicreg_weight = vicreg_weight
        self.graph_sparsity_weight = graph_sparsity_weight
        self.n_eval_times = max(1, n_eval_times)

        # Context Encoder (Online, Trainable)
        self.context_encoder = CausalSSMContextEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            ssm_layers=ssm_layers,
            gat_layers=gat_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Target Encoder (Momentum, Frozen) via JEPABase helper
        self.target_encoder = self.init_target_encoder(self.context_encoder)

        # Continuous Flow Predictor v_psi(z_t, t, z_ctx)
        self.flow_predictor = FlowLatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=flow_layers,
            dropout=dropout,
        )

    def update_target_encoder(self, momentum: Optional[float] = None):
        """Update target encoder via Exponential Moving Average (EMA)."""
        m = self.ema_decay if momentum is None else momentum
        with torch.no_grad():
            for param_online, param_target in zip(
                self.context_encoder.parameters(), self.target_encoder.parameters()
            ):
                param_target.data.mul_(m).add_(param_online.data, alpha=1.0 - m)

    def forward(
        self,
        x_ctx: torch.Tensor,
        x_tgt: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute training loss under Optimal Transport Conditional Flow Matching (OT-CFM).

        Args:
            x_ctx: Context window of shape (batch, seq_len_ctx, in_channels).
            x_tgt: Target window of shape (batch, seq_len_tgt, in_channels).
            return_diagnostics: If True, returns (loss, metrics_dict).

        Returns:
            Scalar loss tensor, optionally with dictionary of diagnostic metrics.
        """
        B = x_ctx.shape[0]
        device = x_ctx.device

        # Step 1: Online context encoding with graph outputs
        z_ctx, z_nodes, attn_matrix = self.context_encoder(x_ctx, return_graph=True)

        # Step 2: Target momentum encoding (z_1)
        with torch.no_grad():
            z_1 = self.target_encoder(x_tgt, return_graph=False)

        # Step 3: Optimal Transport flow interpolation
        # Sample prior noise z_0 ~ N(0, I) and time t ~ U(0, 1)
        z_0 = torch.randn_like(z_1)
        t = torch.rand(B, device=device)

        # Straight OT path: z_t = (1 - t) * z_0 + t * z_1, target velocity = z_1 - z_0
        t_expand = t[:, None]
        z_t = (1.0 - t_expand) * z_0 + t_expand * z_1
        target_velocity = z_1 - z_0

        # Predict velocity field
        pred_velocity = self.flow_predictor(z_t, t, z_ctx)
        cfm_loss = F.mse_loss(pred_velocity, target_velocity)

        # Step 4: Non-contrastive VICReg regularization on representations
        var_ctx, cov_ctx = _vicreg_branch_loss(z_ctx)
        var_tgt, cov_tgt = _vicreg_branch_loss(z_1)
        vic_loss = 0.5 * (var_ctx + var_tgt) + 0.25 * (cov_ctx + cov_tgt)

        # Step 5: Causal graph sparsity penalty
        graph_loss = torch.mean(torch.abs(attn_matrix)) if attn_matrix is not None else torch.tensor(0.0, device=device)

        total_loss = cfm_loss + self.vicreg_weight * vic_loss + self.graph_sparsity_weight * graph_loss

        if return_diagnostics:
            diagnostics = {
                "cfm_loss": float(cfm_loss.detach().item()),
                "vic_loss": float(vic_loss.detach().item()),
                "graph_loss": float(graph_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
            }
            return total_loss, diagnostics

        return total_loss

    def compute_anomaly_score(
        self,
        x_ctx: torch.Tensor,
        x_tgt: torch.Tensor,
        eval_time: float = 0.5,
    ) -> torch.Tensor:
        """Compute Deterministic Velocity Discrepancy score at inference.

        Evaluates velocity prediction error at deterministic midpoint t=0.5 with zero prior variance.

        Args:
            x_ctx: Context window (batch, seq_len_ctx, in_channels).
            x_tgt: Target window (batch, seq_len_tgt, in_channels).
            eval_time: Continuous flow time evaluation point (default 0.5).

        Returns:
            1D anomaly score tensor of shape (batch,).
        """
        self.eval()
        with torch.no_grad():
            B = x_ctx.shape[0]
            device = x_ctx.device

            z_ctx = self.context_encoder(x_ctx, return_graph=False)
            z_1 = self.target_encoder(x_tgt, return_graph=False)

            # Deterministic midpoint evaluation: z_0 = 0
            z_0 = torch.zeros_like(z_1)
            t = torch.full((B,), fill_value=eval_time, device=device)
            t_expand = t[:, None]
            z_t = (1.0 - t_expand) * z_0 + t_expand * z_1
            target_velocity = z_1 - z_0

            pred_velocity = self.flow_predictor(z_t, t, z_ctx)
            discrepancy = torch.norm(pred_velocity - target_velocity, p=2, dim=-1)
            return discrepancy

    def counterfactual_root_cause_attribution(
        self,
        x_ctx: torch.Tensor,
        x_tgt: torch.Tensor,
        eval_time: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform zero-shot counterfactual intervention to rank root-cause anomaly channels.

        For each channel i in {0, ..., C-1}, evaluates discrepancy drop under do(X_{-i}):
        Delta D_i = Score(X) - Score(X with channel i masked)

        Args:
            x_ctx: Context window (batch, seq_len_ctx, in_channels).
            x_tgt: Target window (batch, seq_len_tgt, in_channels).
            eval_time: Time evaluation point.

        Returns:
            - channel_scores: (batch, in_channels) channel attribution scores Delta D_i.
            - top_root_causes: (batch, in_channels) ranked channel indices from highest to lowest contribution.
        """
        self.eval()
        with torch.no_grad():
            B, L_ctx, C = x_ctx.shape
            base_score = self.compute_anomaly_score(x_ctx, x_tgt, eval_time=eval_time)  # (B,)

            channel_contributions = []
            for c in range(C):
                # Counterfactual intervention: mask channel c to mean baseline
                x_ctx_cf = x_ctx.clone()
                x_tgt_cf = x_tgt.clone()
                x_ctx_cf[:, :, c] = 0.0
                x_tgt_cf[:, :, c] = 0.0

                cf_score = self.compute_anomaly_score(x_ctx_cf, x_tgt_cf, eval_time=eval_time)  # (B,)
                # Discrepancy drop caused by removing channel c
                delta_d = base_score - cf_score
                channel_contributions.append(delta_d)

            channel_scores = torch.stack(channel_contributions, dim=1)  # (B, C)
            top_root_causes = torch.argsort(channel_scores, dim=1, descending=True)  # (B, C)
            return channel_scores, top_root_causes

    def get_causal_graph(self, x_ctx: torch.Tensor) -> torch.Tensor:
        """Extract learned causal dependency matrix A in R^{B x C x C} across sensor channels."""
        self.eval()
        with torch.no_grad():
            _, _, attn_matrix = self.context_encoder(x_ctx, return_graph=True)
            return attn_matrix

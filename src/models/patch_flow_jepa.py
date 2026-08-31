"""Patch-Level Sequence Flow Matching Joint Embedding Predictive Architecture (PatchFlowJEPA).

Combines:
1. Spatio-temporal patch tokenization with Transformer sequence encoders.
2. Exponential Moving Average (EMA) patch target representation encoding.
3. Patch Flow Transformer Predictor (DiT-style cross-attention) learning continuous
   velocity fields v_psi(Z_{t, tgt}, t, H_ctx) across future target patch tokens.
4. Optimal Transport Conditional Flow Matching (OT-CFM) with token-level VICReg regularization.
5. High-resolution patch-level spatio-temporal anomaly localization.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase
from src.models.flow_ts_jepa import TimestepEmbedding, _get_chebyshev_collocation_nodes
from src.models.patch_ts_jepa import PatchSequenceEncoder, PositionalEncoding
from src.models.ts_jepa import _vicreg_branch_loss


class PatchFlowPredictor(nn.Module):
    """Flow Transformer Predictor for patch token sequences.
    
    Predicts continuous velocity fields v_psi(Z_{t, tgt}, t, H_ctx) across target patch tokens
    conditioned on time t in [0, 1] and context patch representations H_ctx.
    """

    def __init__(
        self,
        d_model: int = 48,
        n_target_patches: int = 4,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 96,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_target_patches = n_target_patches

        self.pos_encoder = PositionalEncoding(d_model)
        self.time_embed = TimestepEmbedding(embed_dim=d_model)

        # Transformer Decoder Layers (Cross-attending from target tokens to context memory)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output velocity projection
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        z_t_tgt: torch.Tensor,
        t: torch.Tensor,
        h_context: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocities for all target patch tokens.
        
        Args:
            z_t_tgt: Noisy target tokens at time t, shape (B, N_tgt, d_model)
            t: Continuous flow time in [0, 1], shape (B,) or scalar
            h_context: Context memory sequence, shape (B, N_ctx, d_model)
            
        Returns:
            v_pred: Predicted velocities dz_t/dt, shape (B, N_tgt, d_model)
        """
        B, N_tgt, D = z_t_tgt.shape
        if t.ndim == 0 or (t.ndim == 1 and t.size(0) == 1 and B > 1):
            t = t.expand(B)

        # Timestep modulation embedding added to target tokens
        t_emb = self.time_embed(t).unsqueeze(1)  # (B, 1, d_model)
        tokens = z_t_tgt + t_emb
        tokens = self.pos_encoder(tokens)

        # Cross-attention decoding
        out = self.decoder(tgt=tokens, memory=h_context)
        out = self.out_norm(out)
        return self.out_proj(out)


class PatchFlowJEPA(JEPABase):
    """Patch-Level Sequence Flow Matching Joint Embedding Predictive Architecture."""

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 96,
        n_target_patches: int = 4,
        predictor_layers: int = 2,
        ema_decay: float = 0.996,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_target_patches = n_target_patches
        self.ema_decay = ema_decay

        # Active Context Encoder
        self.context_encoder = PatchSequenceEncoder(
            input_dim=input_dim,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # EMA Target Encoder
        self.target_encoder = self.init_target_encoder(self.context_encoder)

        # Flow Transformer Predictor
        self.flow_predictor = PatchFlowPredictor(
            d_model=d_model,
            n_target_patches=n_target_patches,
            n_heads=n_heads,
            n_layers=predictor_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Buffers for Mahalanobis-whitened flow scoring
        self.register_mahalanobis_buffers(d_model)

    @property
    def latent_dim(self) -> int:
        """Alias for ``d_model`` so patch flow models expose the same interface as other JEPA variants."""
        return self.d_model

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        z_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for training.
        
        Args:
            context_windows: (B, L_ctx, C)
            target_windows: Optional (B, L_tgt, C)
            t: Optional continuous flow time in [0, 1], shape (B,)
            z_noise: Optional prior noise tokens, shape (B, N_tgt, d_model)
            
        Returns:
            Tuple of (h_ctx, z_tgt_true, v_pred, v_target)
        """
        B = context_windows.size(0)
        device = context_windows.device

        # Encode context tokens
        h_ctx = self.context_encoder(context_windows)  # (B, N_ctx, d_model)

        if target_windows is None:
            return h_ctx, None, None, None

        expected_len = self.n_target_patches * self.patch_size
        if target_windows.size(1) != expected_len:
            raise ValueError(
                f"target_windows sequence length ({target_windows.size(1)}) must match "
                f"n_target_patches * patch_size ({self.n_target_patches} * {self.patch_size} = {expected_len})."
            )

        # Encode target tokens with EMA target encoder
        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt_true = self.target_encoder(target_windows)  # (B, N_tgt, d_model)

        if t is None:
            t = torch.rand(B, device=device, dtype=context_windows.dtype)

        if z_noise is None:
            z_noise = torch.randn(B, self.n_target_patches, self.d_model, device=device, dtype=context_windows.dtype)

        # OT Flow Linear Interpolation: Z_t = (1 - t) * Z_0 + t * Z_1
        t_expand = t.view(B, 1, 1)
        z_t = (1.0 - t_expand) * z_noise + t_expand * z_tgt_true

        # Ground truth target velocity
        v_target = z_tgt_true - z_noise

        # Predicted velocity field
        v_pred = self.flow_predictor(z_t, t, h_ctx)

        return h_ctx, z_tgt_true, v_pred, v_target

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows,
        target_windows,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical residual covariance across patch tokens for Mahalanobis scoring.

        Accepts numpy arrays or tensors; data is transferred in batches.
        """
        from src.models._jepa_utils import fit_covariance_batched

        def residual_fn(ctx_b, tgt_b):
            h_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            B_b = ctx_b.size(0)
            t_mid = torch.full((B_b,), 0.5, device=ctx_b.device, dtype=ctx_b.dtype)
            z_mid = 0.5 * z_tgt
            v_pred = self.flow_predictor(z_mid, t_mid, h_ctx)
            return (v_pred - z_tgt).reshape(-1, self.d_model)

        fit_covariance_batched(
            self,
            context_windows,
            target_windows,
            residual_fn=residual_fn,
            dim=self.d_model,
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
        collocation: Literal["midpoint", "chebyshev_3", "chebyshev_5"] = "midpoint",
    ) -> torch.Tensor:
        """Compute deterministic patch flow discrepancy averaged across target horizon.
        
        Evaluates the vector field along deterministic Chebyshev-Lobatto quadrature nodes
        along the straight path from z_0=0 to z_1=z_tgt:
        || v_psi(t * z_tgt, t, h_ctx) - z_tgt ||.
        
        Completely eliminates Monte Carlo sampling noise, allowing EVT threshold calibration
        to match the true GPD tail.
        
        Returns:
            Tensor of shape (B,)
        """
        if use_mahalanobis and not bool(self.precision_fitted.item()):
            raise RuntimeError("Mahalanobis scoring requested, but covariance has not been fitted.")

        self.eval()
        B = context_windows.size(0)
        device = context_windows.device
        dtype = context_windows.dtype

        h_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        nodes, weights = _get_chebyshev_collocation_nodes(collocation, device=device, dtype=dtype)
        total_window_score = torch.zeros(B, device=device, dtype=dtype)

        for t_val, w_val in zip(nodes, weights):
            t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)
            z_node = t_val * z_tgt
            v_pred = self.flow_predictor(z_node, t_tensor, h_ctx)
            diff = v_pred - z_tgt  # (B, N_tgt, d_model)

            if use_mahalanobis:
                diff_c = diff - self.residual_mean
                mahal = torch.sum((diff_c @ self.precision_matrix) * diff_c, dim=-1)
                patch_scores = torch.sqrt(torch.clamp(mahal, min=1e-8))  # (B, N_tgt)
            else:
                patch_scores = torch.linalg.norm(diff, dim=-1)  # (B, N_tgt)

            node_window_score = patch_scores.mean(dim=-1)  # (B,)
            total_window_score = total_window_score + w_val * node_window_score

        return total_window_score

    @torch.no_grad()
    def sample_target_patches(
        self,
        context_windows: torch.Tensor,
        n_steps: int = 4,
        solver: Literal["euler", "midpoint"] = "midpoint",
        z_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate ODE across target patch tokens from t=0 to t=1.
        
        Args:
            context_windows: (B, L_ctx, C)
            n_steps: Number of integration steps
            solver: Numerical ODE solver ('euler', 'midpoint')
            z_init: Optional initial noise tokens (B, N_tgt, d_model)
            
        Returns:
            Generated target tokens (B, N_tgt, d_model)
        """
        self.eval()
        B = context_windows.size(0)
        device = context_windows.device
        dtype = context_windows.dtype
        h_ctx = self.context_encoder(context_windows)

        if z_init is None:
            z_t = torch.randn(B, self.n_target_patches, self.d_model, device=device, dtype=dtype)
        else:
            z_t = z_init.clone()

        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = step * dt
            t_curr = torch.full((B,), t_val, device=device, dtype=dtype)

            if solver == "euler":
                v = self.flow_predictor(z_t, t_curr, h_ctx)
                z_t = z_t + dt * v
            elif solver == "midpoint":
                v_curr = self.flow_predictor(z_t, t_curr, h_ctx)
                z_mid = z_t + 0.5 * dt * v_curr
                t_mid = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=dtype)
                v_mid = self.flow_predictor(z_mid, t_mid, h_ctx)
                z_t = z_t + dt * v_mid
            else:
                raise ValueError(f"Unknown solver: {solver}")

        return z_t

    @torch.no_grad()
    def compute_patch_instantaneous_discrepancy(
        self,
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        n_eval_times: int = 1,
        collocation: Literal["midpoint", "chebyshev_3", "chebyshev_5"] = "midpoint",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute patch-level and window-level deterministic flow discrepancy.

        Evaluates the vector field along the straight OT path from prior mean z_0=0 to target tokens z_tgt.
        Supports Chebyshev-Lobatto multi-collocation quadrature nodes.

        Returns:
            patch_scores: (B, N_tgt) - Anomaly score localized per patch token
            window_scores: (B,) - Mean anomaly score across target window
        """
        self.eval()
        B = context_windows.size(0)
        device = context_windows.device
        dtype = context_windows.dtype

        h_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(target_windows)

        nodes, weights = _get_chebyshev_collocation_nodes(collocation, device=device, dtype=dtype)
        total_patch_scores = torch.zeros(B, self.n_target_patches, device=device, dtype=dtype)

        for t_val, w_val in zip(nodes, weights):
            t_curr = torch.full((B,), t_val, device=device, dtype=dtype)
            z_t = t_val * z_tgt
            v_pred = self.flow_predictor(z_t, t_curr, h_ctx)
            node_patch_scores = torch.linalg.norm(v_pred - z_tgt, dim=-1)  # (B, N_tgt)
            total_patch_scores = total_patch_scores + w_val * node_patch_scores

        window_scores = total_patch_scores.mean(dim=-1)  # (B,)
        return total_patch_scores, window_scores


    @torch.no_grad()
    def compute_patch_trajectory_discrepancy(
        self,
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        n_steps: int = 4,
        solver: Literal["euler", "midpoint"] = "midpoint",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute patch-level and window-level trajectory discrepancy via ODE sampling.
        
        Returns:
            patch_scores: (B, N_tgt)
            window_scores: (B,)
        """
        self.eval()
        z_tgt_obs = self.target_encoder(target_windows)
        z_gen = self.sample_target_patches(context_windows, n_steps=n_steps, solver=solver)

        patch_scores = torch.linalg.norm(z_tgt_obs - z_gen, dim=-1)  # (B, N_tgt)
        window_scores = patch_scores.mean(dim=-1)  # (B,)
        return patch_scores, window_scores

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

import copy
import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.flow_ts_jepa import TimestepEmbedding
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


class PatchFlowJEPA(nn.Module):
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
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

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
        self.register_buffer("precision_matrix", torch.eye(d_model))
        self.register_buffer("residual_mean", torch.zeros(d_model))
        self.register_buffer("precision_fitted", torch.tensor(False, dtype=torch.bool))

    def train(self, mode: bool = True) -> PatchFlowJEPA:
        """Keep target encoder strictly in evaluation mode."""
        super().train(mode)
        self.target_encoder.eval()
        return self

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

        # Encode target tokens with EMA target encoder
        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt_true = self.target_encoder(target_windows)  # (B, N_tgt, d_model)

        if t is None:
            t = torch.rand(B, device=device)

        if z_noise is None:
            z_noise = torch.randn(B, self.n_target_patches, self.d_model, device=device)

        # OT Flow Linear Interpolation: Z_t = (1 - t) * Z_0 + t * Z_1
        t_expand = t.view(B, 1, 1)
        z_t = (1.0 - t_expand) * z_noise + t_expand * z_tgt_true

        # Ground truth target velocity
        v_target = z_tgt_true - z_noise

        # Predicted velocity field
        v_pred = self.flow_predictor(z_t, t, h_ctx)

        return h_ctx, z_tgt_true, v_pred, v_target

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        """Update target encoder via Exponential Moving Average (EMA)."""
        m = self.ema_decay if decay is None else decay
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.data)
        for buf_q, buf_k in zip(self.context_encoder.buffers(), self.target_encoder.buffers()):
            buf_k.data.copy_(buf_q.data)

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical residual covariance across patch tokens for Mahalanobis scoring."""
        self.eval()
        n_samples = len(context_windows)
        residuals_list = []

        for i in range(0, n_samples, batch_size):
            ctx_b = context_windows[i : i + batch_size]
            tgt_b = target_windows[i : i + batch_size]
            h_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            B_b = len(ctx_b)
            t_mid = torch.full((B_b,), 0.5, device=ctx_b.device, dtype=torch.float32)
            z_mid = 0.5 * z_tgt
            v_pred = self.flow_predictor(z_mid, t_mid, h_ctx)
            residuals_list.append((v_pred - z_tgt).reshape(-1, self.d_model))

        residuals = torch.cat(residuals_list, dim=0)
        mean_res = residuals.mean(dim=0, keepdim=True)
        self.residual_mean.copy_(mean_res.squeeze(0))
        residuals_centered = residuals - mean_res
        cov = (residuals_centered.T @ residuals_centered) / max(len(residuals) - 1, 1)
        cov_reg = cov + reg * torch.eye(self.d_model, device=cov.device)
        self.precision_matrix.copy_(torch.linalg.pinv(cov_reg))
        self.precision_fitted.copy_(torch.tensor(True, dtype=torch.bool))

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        """Compute deterministic patch flow discrepancy averaged across target horizon.
        
        Evaluates the vector field at t=0.5 along the straight path from z_0=0 to z_1=z_tgt:
        || v_psi(0.5 * z_tgt, t=0.5, h_ctx) - z_tgt ||.
        
        Completely eliminates Monte Carlo sampling noise, allowing EVT threshold calibration
        to match the true GPD tail.
        
        Returns:
            Tensor of shape (B,)
        """
        self.eval()
        B = context_windows.size(0)
        device = context_windows.device

        h_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        t_mid = torch.full((B,), 0.5, device=device, dtype=torch.float32)
        z_mid = 0.5 * z_tgt
        v_pred = self.flow_predictor(z_mid, t_mid, h_ctx)
        diff = v_pred - z_tgt  # (B, N_tgt, d_model)

        if use_mahalanobis and bool(self.precision_fitted.item()):
            diff_c = diff - self.residual_mean
            mahal = torch.sum((diff_c @ self.precision_matrix) * diff_c, dim=-1)
            patch_scores = torch.sqrt(torch.clamp(mahal, min=1e-8))  # (B, N_tgt)
        else:
            patch_scores = torch.linalg.norm(diff, dim=-1)  # (B, N_tgt)

        return torch.mean(patch_scores, dim=-1)

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
        h_ctx = self.context_encoder(context_windows)

        if z_init is None:
            z_t = torch.randn(B, self.n_target_patches, self.d_model, device=device)
        else:
            z_t = z_init.clone()

        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = step * dt
            t_curr = torch.full((B,), t_val, device=device, dtype=torch.float32)

            if solver == "euler":
                v = self.flow_predictor(z_t, t_curr, h_ctx)
                z_t = z_t + dt * v
            elif solver == "midpoint":
                v_curr = self.flow_predictor(z_t, t_curr, h_ctx)
                z_mid = z_t + 0.5 * dt * v_curr
                t_mid = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=torch.float32)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute patch-level and window-level deterministic flow discrepancy.
        
        Returns:
            patch_scores: (B, N_tgt) - Anomaly score localized per patch token
            window_scores: (B,) - Mean anomaly score across target window
        """
        self.eval()
        B = context_windows.size(0)
        device = context_windows.device

        h_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(target_windows)

        t_mid = torch.full((B,), 0.5, device=device, dtype=torch.float32)
        z_mid = 0.5 * z_tgt
        v_pred = self.flow_predictor(z_mid, t_mid, h_ctx)

        patch_scores = torch.linalg.norm(v_pred - z_tgt, dim=-1)  # (B, N_tgt)
        window_scores = patch_scores.mean(dim=-1)  # (B,)
        return patch_scores, window_scores

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

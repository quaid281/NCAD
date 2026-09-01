"""Patch-Level Sequence Joint Embedding Predictive Architecture (Patch-TS-JEPA).

Eliminates temporal dilution caused by global pooling by tokenizing time-series
into non-overlapping patches, encoding token dynamics with Transformers, and
predicting future patch token embeddings directly in latent representation space.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase
from src.models.jepa.ts_jepa import jepa_vicreg_loss


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding: (B, N, D) + (1, N, D)."""
        return x + self.pe[:, : x.size(1)]


class PatchTokenizer(nn.Module):
    """Splits a multivariate sequence into non-overlapping temporal patches."""

    def __init__(self, input_dim: int, patch_size: int, d_model: int):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.d_model = d_model
        self.proj = nn.Linear(input_dim * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, L, C) -> (B, N_patches, d_model)."""
        B, L, C = x.shape
        if L % self.patch_size != 0:
            raise ValueError(
                f"Sequence length {L} must be divisible by patch_size {self.patch_size}"
            )
        num_patches = L // self.patch_size
        # Reshape to (B, num_patches, patch_size * C)
        x_patches = x.view(B, num_patches, self.patch_size * C)
        return self.norm(self.proj(x_patches))


class PatchSequenceEncoder(nn.Module):
    """Transformer Encoder operating over patch tokens."""

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 96,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tokenizer = PatchTokenizer(input_dim, patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, L, C) -> (B, N_patches, d_model)."""
        tokens = self.tokenizer(x)
        tokens = self.pos_encoder(tokens)
        out = self.transformer(tokens)
        return self.norm(out)


class PatchSequencePredictor(nn.Module):
    """Predicts future patch tokens using cross-attention over context patch representations."""

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
        self.n_target_patches = n_target_patches
        self.d_model = d_model
        # Learnable target future query tokens
        self.target_queries = nn.Parameter(torch.randn(1, n_target_patches, d_model) * 0.02)
        self.pos_encoder = PositionalEncoding(d_model)

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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_context: torch.Tensor) -> torch.Tensor:
        """Predict future tokens: (B, N_ctx, d_model) -> (B, N_tgt, d_model)."""
        B = h_context.size(0)
        queries = self.target_queries.repeat(B, 1, 1)
        queries = self.pos_encoder(queries)
        out = self.decoder(tgt=queries, memory=h_context)
        return self.norm(out)


class PatchTSJEPA(JEPABase):
    """Patch-Level Sequence Joint Embedding Predictive Architecture."""

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 96,
        n_target_patches: int = 4,
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

        # Sequence Predictor
        self.predictor = PatchSequencePredictor(
            d_model=d_model,
            n_target_patches=n_target_patches,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.register_mahalanobis_buffers(d_model)

    @property
    def latent_dim(self) -> int:
        """Alias for ``d_model`` so patch models expose the same interface as other JEPA variants."""
        return self.d_model

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward pass.
        
        Returns:
            Tuple of (h_ctx: (B, N_ctx, D), h_tgt_true: (B, N_tgt, D), h_tgt_pred: (B, N_tgt, D))
        """
        h_ctx = self.context_encoder(context_windows)
        h_tgt_pred = self.predictor(h_ctx)

        h_tgt_true = None
        if target_windows is not None:
            expected_len = self.n_target_patches * self.patch_size
            if target_windows.size(1) != expected_len:
                raise ValueError(
                    f"target_windows sequence length ({target_windows.size(1)}) must match "
                    f"n_target_patches * patch_size ({self.n_target_patches} * {self.patch_size} = {expected_len})."
                )
            self.target_encoder.eval()
            with torch.no_grad():
                h_tgt_true = self.target_encoder(target_windows)

        return h_ctx, h_tgt_true, h_tgt_pred

    def compute_objective(self, ctx, tgt, config, **kwargs):
        """Patch-level JEPA VICReg loss."""
        h_ctx, h_tgt_true, h_tgt_pred = self.forward(ctx, tgt)
        loss = self.compute_patch_loss(
            h_tgt_pred, h_tgt_true, h_ctx=h_ctx,
            sim_weight=config.vicreg_sim_weight,
            var_weight=config.vicreg_var_weight,
            cov_weight=config.vicreg_cov_weight,
        )
        return loss, {"loss": float(loss.item())}

    def compute_patch_loss(
        self,
        h_tgt_pred: torch.Tensor,
        h_tgt_true: torch.Tensor,
        h_ctx: Optional[torch.Tensor] = None,
        sim_weight: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.5,
    ) -> torch.Tensor:
        """Token-level VICReg loss averaged across future patch positions."""
        if h_tgt_pred.shape != h_tgt_true.shape:
            raise ValueError(
                f"Shape mismatch between h_tgt_pred {tuple(h_tgt_pred.shape)} and h_tgt_true {tuple(h_tgt_true.shape)}."
            )
        B, N_tgt, D = h_tgt_pred.shape
        loss_total = 0.0
        for i in range(N_tgt):
            z_pred_i = h_tgt_pred[:, i, :]
            z_true_i = h_tgt_true[:, i, :]
            z_ctx_i = h_ctx[:, i % h_ctx.size(1), :] if h_ctx is not None else None
            loss_i = jepa_vicreg_loss(
                z_pred_i, z_true_i, z_context=z_ctx_i,
                sim_weight=sim_weight, var_weight=var_weight, cov_weight=cov_weight,
            )
            loss_total = loss_total + loss_i
        return loss_total / N_tgt

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows,
        target_windows,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical residual covariance across patch tokens using batched accumulation.

        Accepts numpy arrays or tensors; data is transferred in batches.
        """
        from src.models._jepa_utils import fit_covariance_batched

        def residual_fn(ctx_b, tgt_b):
            h_ctx = self.context_encoder(ctx_b)
            h_pred = self.predictor(h_ctx)
            h_obs = self.target_encoder(tgt_b)
            return (h_obs - h_pred).reshape(-1, self.d_model)

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

    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        """Compute patch-level prediction discrepancy.
        
        Returns:
            Tensor of shape (B,) representing the mean patch prediction error.
        """
        self.eval()
        with torch.no_grad():
            h_ctx = self.context_encoder(context_windows)
            h_pred = self.predictor(h_ctx)
            h_obs = self.target_encoder(observed_target_windows)
            diff = h_obs - h_pred  # (B, N_tgt, D)
            if use_mahalanobis:
                if not bool(self.precision_fitted.item()):
                    raise RuntimeError(
                        "Mahalanobis discrepancy requested (use_mahalanobis=True), "
                        "but the precision matrix has not been fitted! "
                        "Call fit_mahalanobis_covariance() before inference."
                    )
                diff_centered = diff - self.residual_mean
                # (B, N_tgt, D) @ (D, D) -> (B, N_tgt, D)
                mahal = torch.sum((diff_centered @ self.precision_matrix) * diff_centered, dim=-1)
                patch_scores = torch.sqrt(torch.clamp(mahal, min=1e-8))  # (B, N_tgt)
            else:
                patch_scores = torch.linalg.norm(diff, dim=-1)  # (B, N_tgt)
            # Mean patch score across the target horizon (well-calibrated EVT distribution)
            return torch.mean(patch_scores, dim=-1)

"""Laminar-JEPA-Lite: Streamlined Multi-Horizon Spherical JEPA Architecture.

Consolidates the load-bearing Stage 1--3 components identified by the flagship
layer-to-layer activation autopsy while pruning redundant depth, duplicate
normalizations, dormant submodules, and split-predictor discontinuities:
  1. Adaptive 4-block causal TCN encoder (`dilations = {1, 2, 4, 8}`) with `K > 1`
     depthwise temporal stem, `K >= 25` gated 4-shell rFFT dyadic front-end, zero
     unused QR trajectory projections, and zero dead `Dropout(0.0)` leaves.
  2. Multi-depth functional hierarchy tapping (`MultiDepthTapHead`) + synchronized
     EMA online barycenter subtraction (`running_centroid`).
  3. Single parallel `LiteLatentPredictor` with one internal `LayerNorm(H)` and
     one terminal `F.layer_norm(D)` across `M = 3` dyadic target prefixes
     (`{S/4, S/2, S}` = `{16, 32, 64}` at `S = 64`).
  4. Training-fixed `(M*D) x (M*D)` (`96 x 96`) resolvent precision whitening.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched, whiten_residuals
from src.models.encoders.tcn_encoder import TimeFirstLayerNorm
from src.models.jepa.alien_jepas import (
    _encode_with_multidepth_taps,
    _init_jepa_encoders,
    _init_tap_heads,
    _update_tap_ema,
    _whitening_defect_loss,
)


class LaminarLiteTCNBlock(nn.Module):
    """Horizon-adaptive two-layer causal dilated TCN block without dead dropout leaves."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = TimeFirstLayerNorm(channels)
        self.norm2 = TimeFirstLayerNorm(channels)

    def effective_dilation_and_padding(self, seq_len: int) -> tuple[int, int]:
        """Cap receptive span `2 * (k - 1) * d_eff` at `seq_len` for short target prefixes."""
        if self.kernel_size <= 1:
            return 1, 0
        max_dilation = max(1, seq_len // (2 * (self.kernel_size - 1)))
        eff_dilation = min(self.dilation, max_dilation)
        return eff_dilation, (self.kernel_size - 1) * eff_dilation

    @staticmethod
    def _run_conv(conv: nn.Conv1d, x: torch.Tensor, eff_dilation: int) -> torch.Tensor:
        orig_dilation = conv.dilation
        if orig_dilation == (eff_dilation,):
            return conv(x)
        try:
            conv.dilation = (eff_dilation,)
            return conv(x)
        finally:
            conv.dilation = orig_dilation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len = inputs.shape[-1]
        eff_dilation, eff_padding = self.effective_dilation_and_padding(seq_len)

        x = F.pad(inputs, (eff_padding, 0))
        x = self.norm1(F.gelu(self._run_conv(self.conv1, x, eff_dilation)))

        x = F.pad(x, (eff_padding, 0))
        x = self.norm2(F.gelu(self._run_conv(self.conv2, x, eff_dilation)))
        return F.gelu(x + inputs)


class LaminarLiteEncoder(nn.Module):
    """Lean causal multiscale TCN encoder with `K >= 25` dyadic gating and zero QR overhead."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        filters: int = 48,
        tcn_layers: int = 4,
        kernel_size: int = 5,
        dyadic_min_channels: int = 25,
        num_dyadic_shells: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.filters = filters
        self.tcn_layers = tcn_layers
        self.dyadic_min_channels = dyadic_min_channels

        self.stem_kernel = 3
        if input_dim > 1:
            self.channel_stem: Optional[nn.Conv1d] = nn.Conv1d(
                input_dim, input_dim, kernel_size=self.stem_kernel, groups=input_dim
            )
            self.stem_gate: Optional[nn.Parameter] = nn.Parameter(torch.tensor(0.05))
        else:
            self.channel_stem = None
            self.stem_gate = None

        self.input_projection = nn.Conv1d(input_dim, filters, kernel_size=1)

        if input_dim >= dyadic_min_channels:
            from src.models.geometric_layers import LittlewoodPaleyDyadicBlock

            self.dyadic_block: Optional[nn.Module] = LittlewoodPaleyDyadicBlock(
                channels=filters, num_shells=num_dyadic_shells, dropout=0.0
            )
        else:
            self.dyadic_block = None

        self.blocks = nn.ModuleList(
            [
                LaminarLiteTCNBlock(filters, kernel_size=kernel_size, dilation=2**layer)
                for layer in range(tcn_layers)
            ]
        )

        proj_dim = max(32, 2 * latent_dim)
        self.temporal_dense = nn.Sequential(
            nn.Linear(filters * 3, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.stat_dense = nn.Sequential(
            nn.Linear(filters * 3, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(2 * proj_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    @staticmethod
    def _statistical_descriptors(x: torch.Tensor) -> torch.Tensor:
        std = torch.std(x, dim=2, unbiased=False)
        log_energy = torch.log1p(torch.mean(x.pow(2), dim=2))
        if x.shape[2] > 1:
            tv1 = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]), dim=2)
        else:
            tv1 = torch.zeros_like(std)
        return torch.nan_to_num(torch.cat([std, log_energy, tv1], dim=1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, channels), got {tuple(inputs.shape)}.")

        x = inputs.transpose(1, 2)
        if self.channel_stem is not None and self.stem_gate is not None and x.shape[-1] >= self.stem_kernel:
            stem_out = F.gelu(self.channel_stem(F.pad(x, (self.stem_kernel - 1, 0))))
            stem_rms = torch.sqrt(torch.mean(stem_out.pow(2), dim=-1, keepdim=True) + 1.0)
            x = x + torch.tanh(self.stem_gate) * (stem_out / stem_rms)

        x = self.input_projection(x)
        if self.dyadic_block is not None:
            x = self.dyadic_block(x)

        for block in self.blocks:
            x = block(x)

        temporal_cat = torch.cat([x[:, :, -1], torch.max(x, dim=2).values, torch.mean(x, dim=2)], dim=1)
        fused = torch.cat([self.temporal_dense(temporal_cat), self.stat_dense(self._statistical_descriptors(x))], dim=1)
        return torch.nan_to_num(self.bottleneck(fused))


class LiteLatentPredictor(nn.Module):
    """Single-LayerNorm skip-connected MLP predictor supporting batched `[B, M, D]` horizons."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        delta = self.fc2(self.norm1(F.gelu(self.fc1(z_in))))
        return F.layer_norm(z_in + delta, (self.latent_dim,))


class LaminarJEPALiteModel(JEPABase):
    """Streamlined 3-Horizon Unified-Predictor Spherical JEPA with Resolvent Whitening."""

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        predictor_hidden_dim: int = 64,
        ema_decay: float = 0.996,
        num_horizons: int = 3,
    ):
        super().__init__()
        _init_jepa_encoders(self, context_encoder, latent_dim, ema_decay)
        self.num_horizons = num_horizons
        self.num_subblocks = num_horizons  # Alias for multi-horizon helper compatibility

        _init_tap_heads(self, context_encoder, latent_dim)

        self.predictor = LiteLatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim)
        self.horizon_embeddings = nn.Parameter(0.02 * torch.randn(num_horizons, latent_dim))
        self.register_mahalanobis_buffers(num_horizons * latent_dim)

    @staticmethod
    def prefix_lengths(suspect_len: int, num_horizons: int = 3) -> list[int]:
        """Return dyadic prefix lengths `{max(2, S//4), max(4, S//2), S}` for `M = 3`."""
        if num_horizons == 3:
            return [max(2, suspect_len // 4), max(4, suspect_len // 2), suspect_len]
        return [
            suspect_len if m == num_horizons - 1 else max(2, ((m + 1) * suspect_len) // num_horizons)
            for m in range(num_horizons)
        ]

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        super().update_target_encoder(momentum)
        _update_tap_ema(self, momentum)

    def predict_horizons(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """Predict all `M` target horizon summaries in one parallel `[B, M, D]` pass."""
        return self.predictor(z_ctx.unsqueeze(1) + self.horizon_embeddings.unsqueeze(0))

    @torch.no_grad()
    def encode_target_prefixes(self, tgt_win: torch.Tensor) -> torch.Tensor:
        """Encode the `M` nested target prefixes cleanly under `torch.no_grad()`."""
        self.target_encoder.eval()
        self.target_tap_head.eval()
        lengths = self.prefix_lengths(tgt_win.shape[1], self.num_horizons)
        z_list = [
            _encode_with_multidepth_taps(self.target_encoder, tgt_win[:, :s_len, :], self.target_tap_head).detach()
            for s_len in lengths
        ]
        return torch.stack(z_list, dim=1)

    def forward(
        self, ctx_win: torch.Tensor, tgt_win: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        z_ctx = _encode_with_multidepth_taps(self.context_encoder, ctx_win, self.context_tap_head)
        z_sub_pred = self.predict_horizons(z_ctx)
        z_pred = z_sub_pred[:, -1, :]

        z_tgt: Optional[torch.Tensor] = None
        z_sub_tgt: Optional[torch.Tensor] = None
        if tgt_win is not None:
            z_sub_tgt = self.encode_target_prefixes(tgt_win)
            z_tgt = z_sub_tgt[:, -1, :]
        return z_ctx, z_tgt, z_pred, z_sub_pred, z_sub_tgt

    def _subblock_residuals_flat(self, ctx_win: torch.Tensor, tgt_win: torch.Tensor) -> torch.Tensor:
        _, _, _, z_sub_pred, z_sub_tgt = self.forward(ctx_win, tgt_win)
        assert z_sub_tgt is not None
        return (z_sub_tgt - z_sub_pred).reshape(ctx_win.shape[0], -1)

    def compute_objective(self, ctx: torch.Tensor, tgt: torch.Tensor, config=None, **kwargs):
        _, _, _, z_sub_pred, z_sub_tgt = self.forward(ctx, tgt)
        assert z_sub_tgt is not None
        m_cnt = self.num_horizons
        weights = [0.50 / max(m_cnt - 1, 1)] * (m_cnt - 1) + [0.50]
        loss = sum(
            w_m * (F.mse_loss(z_sub_pred[:, m, :], z_sub_tgt[:, m, :]) + _whitening_defect_loss(z_sub_pred[:, m, :], gamma=0.75))
            for m, w_m in enumerate(weights)
        )
        return loss, {"loss": float(loss.detach().item())}

    @torch.no_grad()
    def fit_resolvent_whitening(self, train_windows: np.ndarray, context_len: int = 256, batch_size: int = 256) -> None:
        """Fit training-only `(M*D) x (M*D)` resolvent precision matrix and residual mean."""
        self.eval()
        ctx_all, tgt_all = train_windows[:, :context_len], train_windows[:, context_len:]
        fit_covariance_batched(
            self,
            ctx_all,
            tgt_all,
            residual_fn=self._subblock_residuals_flat,
            dim=self.num_horizons * self.latent_dim,
            batch_size=batch_size,
            reg=1e-3,
            precision_buffer=self.precision_matrix,
            residual_mean_buffer=self.residual_mean,
            fitted_buffer=self.precision_fitted,
        )

    @torch.no_grad()
    def fit_alien_bank(self, train_windows: np.ndarray, context_len: int = 256, batch_size: int = 256) -> None:
        """Compatibility hook for multi-horizon benchmark pipelines (fits resolvent whitening only)."""
        self.fit_resolvent_whitening(train_windows, context_len=context_len, batch_size=batch_size)

    @torch.no_grad()
    def extract_raw_and_whitened_residuals(
        self, windows: np.ndarray, context_len: int = 256, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract `(N_win, M, D)` raw and resolvent-whitened multi-horizon residuals in one pass."""
        self.eval()
        device = next(self.parameters()).device
        m_cnt, dim = self.num_horizons, self.latent_dim
        raw_chunks: list[np.ndarray] = []
        white_chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for i in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[i : i + batch_size]).float().to(device)
                _, _, _, z_sub_pred, z_sub_tgt = self.forward(batch[:, :context_len], batch[:, context_len:])
                assert z_sub_tgt is not None
                raw = z_sub_tgt - z_sub_pred
                if bool(self.precision_fitted.item()):
                    flat = raw.reshape(len(batch), m_cnt * dim) - self.residual_mean
                    white = whiten_residuals(flat, self.precision_matrix).reshape(len(batch), m_cnt, dim)
                else:
                    white = raw
                raw_chunks.append(raw.detach().cpu().numpy().astype(np.float32))
                white_chunks.append(white.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(raw_chunks, axis=0), np.concatenate(white_chunks, axis=0)

    @torch.no_grad()
    def compute_predictive_discrepancy(self, ctx: torch.Tensor, tgt: torch.Tensor, **kwargs) -> torch.Tensor:
        """1D whitened Euclidean discrepancy fallback for scalar comparator benchmarks."""
        self.eval()
        _, _, _, z_sub_pred, z_sub_tgt = self.forward(ctx, tgt)
        assert z_sub_tgt is not None
        bsz = ctx.shape[0]
        raw_flat = (z_sub_tgt - z_sub_pred).reshape(bsz, self.num_horizons * self.latent_dim)
        if bool(self.precision_fitted.item()):
            white_flat = whiten_residuals(raw_flat - self.residual_mean, self.precision_matrix)
            return torch.linalg.norm(white_flat, dim=-1)
        return torch.linalg.norm(raw_flat, dim=-1)


__all__ = [
    "LaminarJEPALiteModel",
    "LaminarLiteEncoder",
    "LaminarLiteTCNBlock",
    "LiteLatentPredictor",
]

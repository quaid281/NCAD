"""TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis (Wu et al., ICLR 2023).

Transforms 1D time series into 2D temporal variations using Fast Fourier Transform (FFT)
to identify dominant periods, followed by 2D Inception blocks to model intra-period
and inter-period dependencies.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6, init_weight: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,
                    padding=i,
                )
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = [k(x) for k in self.kernels]
        res = torch.stack(res_list, dim=-1).mean(dim=-1)
        return res


def FFT_for_Period(x: torch.Tensor, k: int = 2) -> Tuple[List[int], torch.Tensor]:
    """Find top-k periods and their amplitudes using 1D FFT without NVRTC complex JIT kernels."""
    # x shape: (B, L, d_model)
    xf = torch.fft.rfft(x, dim=1)
    # Compute magnitude using native real/imag operations to avoid libnvrtc complex abs compilation errors
    mag = torch.sqrt(xf.real.pow(2) + xf.imag.pow(2) + 1e-12)
    frequency_list = mag.mean(dim=0).mean(dim=-1)  # (L/2 + 1,)
    frequency_list[0] = 0.0  # Ignore DC / zero-frequency component
    _, top_list = torch.topk(frequency_list, k=min(k, len(frequency_list)))
    top_list_np = top_list.detach().cpu().numpy()

    seq_len = x.shape[1]
    period = []
    for top in top_list_np:
        p = math.ceil(seq_len / (top + 1e-5))
        period.append(max(p, 1))

    period_weight = F.softmax(frequency_list[top_list], dim=-1)
    return period, period_weight.to(x.device)


class TimesBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        top_k: int = 3,
        num_kernels: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        res = []
        for i in range(len(period_list)):
            period = period_list[i]
            # Pad length if not divisible
            if (T % period) != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([B, (length - T), N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x

            # Reshape 1D -> 2D: (B, N, length/period, period)
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D Inception
            out = self.conv(out)
            # Reshape 2D -> 1D: (B, length, N)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B, length, N)
            out = out[:, :T, :]
            res.append(out)

        res = torch.stack(res, dim=-1)
        # Multi-period frequency weighted aggregation
        period_weight = period_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N, len(period_list))
        res = torch.sum(res * period_weight, dim=-1)
        # Residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """TimesNet 2D Temporal Variation Model for Time-Series Anomaly Detection (ICLR 2023)."""

    def __init__(
        self,
        c_in: int,
        d_model: int = 64,
        d_ff: int = 64,
        e_layers: int = 2,
        top_k: int = 3,
        num_kernels: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.embedding = nn.Linear(c_in, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(d_model=d_model, d_ff=d_ff, top_k=top_k, num_kernels=num_kernels)
                for _ in range(e_layers)
            ]
        )
        self.projection = nn.Linear(d_model, c_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels)
            
        Returns:
            reconstruction: (batch, seq_len, channels)
        """
        # Linear embedding + norm
        enc_out = self.embedding(x)
        enc_out = self.layer_norm(enc_out)
        enc_out = self.dropout(enc_out)

        for block in self.blocks:
            enc_out = block(enc_out)

        reconstruction = self.projection(enc_out)
        return reconstruction

    def compute_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute point-level reconstruction anomaly score: ||x - x_rec||_2^2."""
        self.eval()
        with torch.no_grad():
            rec = self.forward(x)
            score = torch.mean((rec - x) ** 2, dim=-1)  # (B, L)
            return score

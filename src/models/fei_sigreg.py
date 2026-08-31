"""Frequency-Masked Embedding Inference (FEI) & Sketched Isotropic Gaussian Regularization (SIGReg).

Eliminates the dependency on artificial time-domain anomaly injections by pre-training
the context encoder using frequency-domain spectral masking (FFT) and non-contrastive
SIGReg variance/covariance regularization.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyMasker:
    """Masks random frequency components of input time-series windows via FFT."""

    def __init__(self, mask_ratio: float = 0.30, seed: Optional[int] = 42):
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}.")
        self.mask_ratio = mask_ratio
        self.seed = seed

    def mask_batch(self, windows: torch.Tensor) -> torch.Tensor:
        """Apply Real FFT, mask random frequency bands, and return reconstructed time-domain tensor."""
        if windows.ndim != 3:
            raise ValueError("Expected 3D tensor of shape (batch, sequence, features).")

        batch_size, seq_len, num_feats = windows.shape
        fft_coefs = torch.fft.rfft(windows, dim=1)
        num_freqs = fft_coefs.shape[1]

        num_mask = min(num_freqs, max(0, int(num_freqs * self.mask_ratio)))
        if num_mask > 0:
            if self.seed is not None:
                gen = torch.Generator(device=windows.device)
                gen.manual_seed(self.seed)
                rand_weights = torch.rand(batch_size, num_freqs, generator=gen, device=windows.device)
            else:
                rand_weights = torch.rand(batch_size, num_freqs, device=windows.device)
            _, mask_indices = torch.topk(rand_weights, k=num_mask, dim=1)
            mask = torch.ones((batch_size, num_freqs, 1), device=windows.device)
            mask.scatter_(1, mask_indices.unsqueeze(-1), 0.0)
            fft_coefs = fft_coefs * mask

        masked_windows = torch.fft.irfft(fft_coefs, n=seq_len, dim=1)
        return masked_windows


def sigreg_loss(
    z_masked: torch.Tensor,
    z_clean: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 0.1,
    eps: float = 1e-4,
    symmetric: bool = True,
) -> torch.Tensor:
    """Non-contrastive SIGReg loss preventing representational collapse.

    Combines Mean Squared Invariance Loss + Variance Regularization + Off-Diagonal Covariance Penalty.
    """
    inv_loss = F.mse_loss(z_masked, z_clean)

    def _var_cov(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        std_z = torch.sqrt(torch.var(z, dim=0, unbiased=False) + eps)
        var_l = torch.mean(F.relu(1.0 - std_z))
        batch_size, latent_dim = z.shape
        if batch_size > 1:
            z_centered = z - torch.mean(z, dim=0, keepdim=True)
            cov_mat = (z_centered.T @ z_centered) / (batch_size - 1)
            off_diag = cov_mat - torch.diag(torch.diag(cov_mat))
            cov_l = torch.sum(off_diag.pow(2)) / latent_dim
        else:
            cov_l = torch.tensor(0.0, device=z.device)
        return var_l, cov_l

    var_clean, cov_clean = _var_cov(z_clean)
    if symmetric:
        var_masked, cov_masked = _var_cov(z_masked)
        var_loss = 0.5 * (var_clean + var_masked)
        cov_loss = 0.5 * (cov_clean + cov_masked)
    else:
        var_loss = var_clean
        cov_loss = cov_clean

    return inv_loss + var_weight * var_loss + cov_weight * cov_loss

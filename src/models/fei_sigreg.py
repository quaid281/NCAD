"""Frequency-Masked Embedding Inference (FEI) & Sketched Isotropic Gaussian Regularization (SIGReg).

Eliminates the dependency on artificial time-domain anomaly injections by pre-training
the context encoder using frequency-domain spectral masking (FFT) and non-contrastive
SIGReg variance/covariance regularization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyMasker:
    """Masks random frequency components of input time-series windows via FFT."""

    def __init__(self, mask_ratio: float = 0.30, seed: int = 42):
        self.mask_ratio = mask_ratio
        self.seed = seed

    def mask_batch(self, windows: torch.Tensor) -> torch.Tensor:
        """Apply Real FFT, mask random frequency bands, and return reconstructed time-domain tensor."""
        if windows.ndim != 3:
            raise ValueError("Expected 3D tensor of shape (batch, sequence, features).")

        batch_size, seq_len, num_feats = windows.shape
        fft_coefs = torch.fft.rfft(windows, dim=1)
        num_freqs = fft_coefs.shape[1]

        num_mask = int(num_freqs * self.mask_ratio)
        if num_mask > 0:
            mask = torch.ones((batch_size, num_freqs, num_feats), device=windows.device)
            for b in range(batch_size):
                perm = torch.randperm(num_freqs, device=windows.device)[:num_mask]
                mask[b, perm, :] = 0.0
            fft_coefs = fft_coefs * mask

        masked_windows = torch.fft.irfft(fft_coefs, n=seq_len, dim=1)
        return masked_windows


def sigreg_loss(
    z_masked: torch.Tensor,
    z_clean: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 0.1,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Non-contrastive SIGReg loss preventing representational collapse.

    Combines Mean Squared Invariance Loss + Variance Regularization + Off-Diagonal Covariance Penalty.
    """
    inv_loss = F.mse_loss(z_masked, z_clean)

    std_z = torch.sqrt(torch.var(z_clean, dim=0, unbiased=False) + eps)
    var_loss = torch.mean(F.relu(1.0 - std_z))

    batch_size, latent_dim = z_clean.shape
    if batch_size > 1:
        z_centered = z_clean - torch.mean(z_clean, dim=0, keepdim=True)
        cov_mat = (z_centered.T @ z_centered) / (batch_size - 1)
        off_diag = cov_mat - torch.diag(torch.diag(cov_mat))
        cov_loss = torch.sum(off_diag.pow(2)) / latent_dim
    else:
        cov_loss = torch.tensor(0.0, device=z_clean.device)

    return inv_loss + var_weight * var_loss + cov_weight * cov_loss

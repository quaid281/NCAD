import pytest
import torch

from src.models.fei_sigreg import FrequencyMasker, sigreg_loss


def test_frequency_masker_shape():
    masker = FrequencyMasker(mask_ratio=0.30, seed=42)
    windows = torch.randn(8, 100, 4)

    masked = masker.mask_batch(windows)

    assert masked.shape == windows.shape
    assert not torch.isnan(masked).any()


def test_sigreg_loss_computation():
    z_clean = torch.randn(16, 8, requires_grad=True)
    z_masked = torch.randn(16, 8, requires_grad=True)

    loss = sigreg_loss(z_masked, z_clean)

    assert loss.ndim == 0
    assert not torch.isnan(loss).item()
    assert loss.item() > 0.0

    loss.backward()
    assert z_clean.grad is not None
    assert z_masked.grad is not None

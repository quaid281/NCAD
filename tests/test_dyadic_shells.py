"""Tests for Littlewood-Paley Dyadic Shell Layer and Multi-Scale Encoder Integration."""

import math
import pytest
import torch
import torch.nn as nn

from src.models.geometric_layers import LittlewoodPaleyDyadicBlock
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.encoders.relational_gat_encoder import RelationalGATEncoder
from src.models.jepa.reynolds_stress_jepa import ReynoldsStressJEPAModel
from src.models.jepa.potential_flow_jepa import PotentialFlowJEPAModel


def test_dyadic_partition_of_unity():
    """Verify that the Littlewood-Paley frequency filters form an exact partition of unity:
    sum_{j=0}^{J-1} Psi_j(omega) == 1.0 for all frequency bins."""
    block = LittlewoodPaleyDyadicBlock(channels=16, num_shells=4)
    F_bins = 129  # corresponds to T = 256
    device = torch.device("cpu")
    
    filters = block.compute_dyadic_filters(F_bins, device=device)  # (J, 1, F_bins)
    assert filters.shape == (4, 1, F_bins)
    
    partition_sum = filters.sum(dim=0)  # (1, F_bins)
    ones = torch.ones_like(partition_sum)
    assert torch.allclose(partition_sum, ones, atol=1e-5), (
        f"Max partition of unity error: {(partition_sum - ones).abs().max().item()}"
    )


def test_dyadic_exact_reconstruction():
    """Verify that decomposing a signal into dyadic shells and summing them reconstructs
    the original signal exactly in the time domain."""
    B, C, T = 2, 8, 128
    x = torch.randn(B, C, T)
    block = LittlewoodPaleyDyadicBlock(channels=C, num_shells=4)
    
    X_fft = torch.fft.rfft(x, dim=-1)
    F_bins = X_fft.shape[-1]
    filters = block.compute_dyadic_filters(F_bins, device=x.device)
    
    X_shells_fft = X_fft.unsqueeze(0) * filters.unsqueeze(1)
    x_shells = torch.fft.irfft(X_shells_fft, n=T, dim=-1)  # (J, B, C, T)
    
    # Sum over all dyadic octaves
    reconstructed = x_shells.sum(dim=0)  # (B, C, T)
    assert torch.allclose(reconstructed, x, atol=1e-5), (
        f"Max reconstruction error: {(reconstructed - x).abs().max().item()}"
    )


def test_dyadic_frequency_selectivity():
    """Verify that a slow diurnal trend maps to low shell and high-frequency spike maps to high shell."""
    C, T = 1, 256
    block = LittlewoodPaleyDyadicBlock(channels=C, num_shells=4)
    
    # Low frequency sine wave: 1 full cycle across window (macro trend)
    t = torch.linspace(0, 1, T)
    x_low = torch.sin(2 * math.pi * t).view(1, C, T)
    
    # High frequency alternating signal: Nyquist spike
    x_high = torch.empty(1, C, T)
    x_high[0, 0, ::2] = 1.0
    x_high[0, 0, 1::2] = -1.0
    
    def get_shell_energies(sig):
        X_fft = torch.fft.rfft(sig, dim=-1)
        filters = block.compute_dyadic_filters(X_fft.shape[-1], device=sig.device)
        X_shells = X_fft.unsqueeze(0) * filters.unsqueeze(1)
        x_sub = torch.fft.irfft(X_shells, n=T, dim=-1)
        return (x_sub ** 2).sum(dim=-1).squeeze()  # (J,)
        
    energy_low = get_shell_energies(x_low)
    energy_high = get_shell_energies(x_high)
    
    # Low frequency signal has dominant energy in shell 0
    assert energy_low[0] > energy_low[-1] * 5.0, (
        f"Expected low shell dominance: {energy_low}"
    )
    # High frequency signal has dominant energy in shell J-1
    assert energy_high[-1] > energy_high[0] * 5.0, (
        f"Expected high shell dominance: {energy_high}"
    )


def test_dyadic_block_gradient_flow():
    """Verify that gradients pass cleanly through the Littlewood-Paley block."""
    B, C, T = 2, 16, 64
    block = LittlewoodPaleyDyadicBlock(channels=C, num_shells=4)
    
    x = torch.randn(B, C, T, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert not torch.all(x.grad == 0)


def test_hybrid_tcn_encoder_dyadic_integration():
    """Verify that HybridTCNEncoder functions seamlessly with use_dyadic_shells=True."""
    B, T, C, D = 4, 128, 6, 16
    enc = HybridTCNEncoder(
        input_dim=C,
        latent_dim=D,
        filters=32,
        tcn_layers=3,
        use_dyadic_shells=True,
        num_dyadic_shells=4,
    )
    
    inputs = torch.randn(B, T, C)
    latent = enc(inputs)
    
    assert latent.shape == (B, D)
    assert torch.all(torch.isfinite(latent))


def test_relational_gat_encoder_dyadic_integration():
    """Verify that RelationalGATEncoder functions seamlessly with use_dyadic_shells=True."""
    B, T, C, D = 2, 64, 5, 16
    enc = RelationalGATEncoder(
        input_dim=C,
        latent_dim=D,
        filters=32,
        tcn_layers=2,
        gat_layers=1,
        use_dyadic_shells=True,
    )
    
    inputs = torch.randn(B, T, C)
    latent = enc(inputs)
    
    assert latent.shape == (B, D)
    assert torch.all(torch.isfinite(latent))


def test_reynolds_stress_jepa_with_dyadic_encoder():
    """Verify end-to-end forward pass and discrepancy computation for ReynoldsStressJEPA
    using the Littlewood-Paley dyadic context encoder."""
    B, T_ctx, T_tgt, C, D = 2, 128, 32, 4, 16
    encoder = HybridTCNEncoder(input_dim=C, latent_dim=D, filters=32, tcn_layers=2, use_dyadic_shells=True)
    model = ReynoldsStressJEPAModel(context_encoder=encoder, latent_dim=D, stress_dim=8)
    
    ctx = torch.randn(B, T_ctx, C)
    tgt = torch.randn(B, T_tgt, C)
    
    loss, metrics = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss)
    
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(torch.isfinite(disc))

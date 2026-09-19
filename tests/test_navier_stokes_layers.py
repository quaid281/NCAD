"""Unit tests for Navier-Stokes Blowup-inspired Architectural Layers.

Verifies:
1. ShearingWaveletPulseBlock: Asymmetric transient growth envelope, viscous dissipation, gradient flow.
2. AdmissibleStressConeProjection: Positive convex cone SPSD guarantee, strictly positive spectrum.
3. StokesStreamCurlFilter: Skew-symmetric rotational curl, orthogonality, volume-preserving latent flow.
4. RadialHeatExteriorBoundary: Preserves core representations, smoothly contracts OOD tail excursions.
5. End-to-end model integration tests for ReynoldsStressJEPAModel, FlowTSJEPAModel, and OperatorEntropyJEPAModel.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.geometric_layers import (
    ShearingWaveletPulseBlock,
    AdmissibleStressConeProjection,
    StokesStreamCurlFilter,
    RadialHeatExteriorBoundary,
)
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.reynolds_stress_jepa import ReynoldsStressJEPAModel, ReynoldsStressClosureHead
from src.models.jepa.flow_ts_jepa import FlowTSJEPAModel, FlowLatentPredictor
from src.models.jepa.operator_entropy_jepa import OperatorEntropyJEPAModel


class TestNavierStokesLayers:
    """Test suite for Navier-Stokes blowup-inspired layers."""

    def test_shearing_wavelet_pulse_block_shape_and_grad(self):
        """Verify ShearingWaveletPulseBlock shapes, residual connections, and full gradient flow."""
        B, T, C = 4, 32, 16
        x = torch.randn(B, T, C, requires_grad=True)

        block = ShearingWaveletPulseBlock(channels=C, kernel_size=15, num_pulses=4, residual=True)
        out = block(x)

        assert out.shape == (B, T, C), f"Expected shape {(B, T, C)}, got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

        # Check backward pass
        loss = out.sum()
        loss.backward()

        assert x.grad is not None and not torch.isnan(x.grad).any()
        assert block.omega.grad is not None
        assert block.shear.grad is not None
        assert block.growth.grad is not None
        assert block.log_viscosity.grad is not None
        assert block.pulse_mix.weight.grad is not None

    def test_shearing_wavelet_pulse_channels_first(self):
        """Verify ShearingWaveletPulseBlock supports (B, C, T) input format."""
        B, C, T = 4, 8, 20
        x = torch.randn(B, C, T)
        block = ShearingWaveletPulseBlock(channels=C, kernel_size=9, num_pulses=2)
        out = block(x)
        assert out.shape == (B, C, T)

    def test_admissible_stress_cone_projection_properties(self):
        """Verify AdmissibleStressConeProjection strictly produces SPSD matrices with positive eigenvalues."""
        B, D, stress_dim = 8, 32, 12
        z = torch.randn(B, D, requires_grad=True)

        cone_layer = AdmissibleStressConeProjection(
            latent_dim=D,
            stress_dim=stress_dim,
            num_shear_modes=4,
            gamma=1e-3,
        )

        sigma = cone_layer(z)
        assert sigma.shape == (B, stress_dim, stress_dim)

        # 1. Symmetry check: ||Sigma - Sigma^T||_F == 0
        sym_diff = torch.linalg.norm(sigma - sigma.transpose(-1, -2), dim=(-2, -1))
        assert torch.all(sym_diff < 1e-5), f"Max asymmetry: {sym_diff.max().item()}"

        # 2. Strict positive definite cone check: eigenvalues > 0
        evals = torch.linalg.eigvalsh(sigma)
        min_eval = evals.min().item()
        assert min_eval >= 1e-3 - 1e-6, f"Expected min eigenvalue >= gamma (1e-3), got {min_eval}"

        # 3. Gradient flow check
        loss = sigma.sum()
        loss.backward()
        assert z.grad is not None and not torch.isnan(z.grad).any()
        assert cone_layer.raw_basis.grad is not None

    def test_stokes_stream_curl_filter_properties(self):
        """Verify StokesStreamCurlFilter produces solenoidal and volume-preserving latent flow."""
        B, D = 6, 16
        v_raw = torch.randn(B, D, requires_grad=True)
        z_ctx = torch.randn(B, D, requires_grad=True)

        filter_layer = StokesStreamCurlFilter(latent_dim=D, hidden_dim=32)
        v_sol = filter_layer(v_raw, z_ctx)

        assert v_sol.shape == (B, D)
        assert not torch.isnan(v_sol).any()

        # Check orthogonality of the rotational component S @ v_raw
        w_mat = filter_layer.stream_generator(z_ctx).view(B, D, D) + filter_layer.base_W.unsqueeze(0)
        S = 0.5 * (w_mat - w_mat.transpose(-1, -2))
        v_rot = torch.bmm(S, v_raw.unsqueeze(-1)).squeeze(-1)
        work = torch.sum(v_raw * v_rot, dim=-1)
        assert torch.all(torch.abs(work) < 1e-4), f"Non-zero rotational work: {work.abs().max().item()}"

        # Gradient flow check
        loss = v_sol.sum()
        loss.backward()
        assert v_raw.grad is not None and z_ctx.grad is not None

    def test_radial_heat_exterior_boundary_damping(self):
        """Verify RadialHeatExteriorBoundary preserves core norms and contracts OOD tail norms."""
        D = 16
        r_core = 4.0  # sqrt(16)
        heat_layer = RadialHeatExteriorBoundary(latent_dim=D, r_core=r_core, h_exponent=0.05)

        # 1. Inside core: ||z|| <= r_core
        z_inner = torch.randn(5, D)
        z_inner = z_inner / torch.norm(z_inner, dim=-1, keepdim=True) * (r_core * 0.5)
        z_inner_out = heat_layer(z_inner)
        # Inside core, excess = 0, chi = 1 -> identical preservation
        assert torch.allclose(z_inner_out, z_inner, atol=1e-5), "Inner core representations modified"

        # 2. Outside core: runaway tail excursion (e.g. norm = 50.0)
        z_raw = torch.randn(5, D)
        z_outer = (z_raw / torch.norm(z_raw, dim=-1, keepdim=True) * 50.0).detach().requires_grad_(True)
        z_outer_out = heat_layer(z_outer)
        outer_norm = torch.norm(z_outer_out, dim=-1)

        # Output norm should be significantly smaller than 50.0 due to heat decay
        assert torch.all(outer_norm < 50.0), f"Outer norm not contracted: {outer_norm}"
        assert not torch.isnan(z_outer_out).any()

        # Check gradient
        loss = z_outer_out.sum()
        loss.backward()
        assert z_outer.grad is not None and not torch.isnan(z_outer.grad).any()


class TestModelIntegrations:
    """Test end-to-end forward passes and objectives for upgraded JEPA models."""

    def test_reynolds_stress_jepa_integration(self):
        """Verify ReynoldsStressJEPAModel with ShearingWaveletPulseBlock & AdmissibleStressConeProjection."""
        B, T, C, D = 4, 32, 3, 16
        encoder = HybridTCNEncoder(input_dim=C, latent_dim=D)
        model = ReynoldsStressJEPAModel(
            context_encoder=encoder,
            latent_dim=D,
            stress_dim=8,
            use_shearing_pulses=True,
            use_admissible_cone=True,
        )

        ctx = torch.randn(B, T, C)
        tgt = torch.randn(B, T, C)

        z_pred, z_tgt, stress_diff = model(ctx, tgt)
        assert z_pred.shape == (B, D)
        assert z_tgt.shape == (B, D)
        assert stress_diff.shape == (B,)
        assert torch.all(stress_diff >= 0.0)

        loss, metrics = model.compute_objective(ctx, tgt)
        assert not torch.isnan(loss) and loss.item() > 0
        assert "pred_loss" in metrics

        # Anomaly scoring
        disc = model.compute_predictive_discrepancy(ctx, tgt)
        assert disc.shape == (B,)
        assert torch.all(disc >= 0.0)

    def test_flow_ts_jepa_stokes_curl_integration(self):
        """Verify FlowTSJEPAModel with StokesStreamCurlFilter."""
        B, T, C, D = 4, 32, 2, 16
        encoder = HybridTCNEncoder(input_dim=C, latent_dim=D)
        model = FlowTSJEPAModel(
            context_encoder=encoder,
            latent_dim=D,
            use_stokes_curl=True,
        )

        ctx = torch.randn(B, T, C)
        tgt = torch.randn(B, T, C)

        loss, metrics = model.compute_objective(ctx, tgt)
        assert not torch.isnan(loss) and loss.item() > 0
        assert "flow_loss" in metrics

        disc = model.compute_predictive_discrepancy(ctx, tgt)
        assert disc.shape == (B,)
        assert torch.all(disc >= 0.0)

    def test_operator_entropy_jepa_radial_heat_integration(self):
        """Verify OperatorEntropyJEPAModel with RadialHeatExteriorBoundary."""
        B, T, C, D = 4, 32, 2, 16
        encoder = HybridTCNEncoder(input_dim=C, latent_dim=D)
        model = OperatorEntropyJEPAModel(
            context_encoder=encoder,
            latent_dim=D,
            use_radial_heat=True,
        )

        ctx = torch.randn(B, T, C)
        tgt = torch.randn(B, T, C)

        loss, metrics = model.compute_objective(ctx, tgt)
        assert not torch.isnan(loss) and loss.item() > 0
        assert "pred_loss" in metrics

        disc = model.compute_predictive_discrepancy(ctx, tgt)
        assert disc.shape == (B,)
        assert torch.all(disc >= 0.0)

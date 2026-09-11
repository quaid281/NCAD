"""Tests for Hankel Power-Sum Moment Discriminant (Chapter 7 of Ten Advances)."""

import pytest
import torch

from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.hankel_moments import HankelMomentTracker
from src.models.jepa.prototype_graph_jepa import PrototypeGraphJEPAModel


def test_hankel_power_sum_moments():
    K = 3
    D = 4
    tracker = HankelMomentTracker(n_regimes=K)

    # 3 distinct scalar regimes for simple checking
    regimes = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                            [2.0, 1.0, 4.0, 3.0],
                            [3.0, 3.0, 1.0, 2.0]])  # (K, D)

    moments = tracker.compute_power_sum_moments(regimes)
    # Shape should be (2*K - 1, D) = (5, 4)
    assert moments.shape == (5, D)

    # Check M_0 = 3
    assert torch.allclose(moments[0], torch.full((D,), 3.0))
    # Check M_1 = 1 + 2 + 3 = 6 for column 0
    assert torch.allclose(moments[1, 0], torch.tensor(6.0))
    # Check M_2 = 1^2 + 2^2 + 3^2 = 14 for column 0
    assert torch.allclose(moments[2, 0], torch.tensor(14.0))


def test_hankel_matrix_structure_and_permutation_invariance():
    K = 3
    D = 4
    tracker = HankelMomentTracker(n_regimes=K)

    regimes = torch.randn(K, D)
    moments = tracker.compute_power_sum_moments(regimes)
    H = tracker.build_hankel_matrix(moments)

    assert H.shape == (D, K, K)
    # Check Hankel symmetry: H[..., i, j] == H[..., j, i]
    assert torch.allclose(H, H.transpose(-1, -2))

    # Permutation invariance: permute the order of regimes
    perm = torch.randperm(K)
    regimes_perm = regimes[perm]
    log_det_orig = tracker.compute_hankel_determinant(regimes)
    log_det_perm = tracker.compute_hankel_determinant(regimes_perm)

    assert torch.allclose(log_det_orig, log_det_perm, atol=1e-4)


def test_hankel_determinant_collapse_on_duplicate_regimes():
    K = 3
    D = 4
    tracker = HankelMomentTracker(n_regimes=K, eps=1e-8)

    # Two identical regimes: Hankel determinant should be close to 0 (log_det very negative)
    regimes = torch.randn(K, D)
    regimes[1] = regimes[0]  # duplicate

    log_det = tracker.compute_hankel_determinant(regimes)
    # Distinct regimes
    regimes_distinct = torch.randn(K, D) * 2.0
    log_det_distinct = tracker.compute_hankel_determinant(regimes_distinct)

    assert torch.all(log_det < log_det_distinct)


def test_prototype_graph_jepa_with_hankel():
    input_dim = 3
    latent_dim = 8
    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = PrototypeGraphJEPAModel(
        context_encoder=encoder,
        latent_dim=latent_dim,
        num_prototypes=4,
    )

    ctx = torch.randn(4, 20, input_dim)
    tgt = torch.randn(4, 10, input_dim)

    loss, metrics = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss)
    assert "loss_hankel" in metrics
    assert metrics["loss_hankel"] >= 0.0

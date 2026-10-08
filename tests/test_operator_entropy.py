"""Unit tests for Operator-Entropy Regularizer and Koopman Operator JEPA."""

import math
import pytest
import torch
import torch.nn.functional as F

from src.models.jepa.operator_entropy_jepa import (
    OperatorEntropyJEPAModel,
    spectral_operator_entropy_loss,
    von_neumann_entropy,
)
from src.models.encoders.tcn_encoder import HybridTCNEncoder


def test_spectral_operator_entropy_non_degeneracy():
    """Verify that K=0 is strictly penalized and K=I achieves minimal deficiency."""
    D = 32

    # Ideal isotropic transition
    K_ideal = torch.eye(D)
    loss_ideal = spectral_operator_entropy_loss(K_ideal)
    assert loss_ideal.item() < 1e-4

    # Rank-1 dimensional collapse
    K_rank1 = torch.zeros(D, D)
    K_rank1[0, 0] = math.sqrt(D)
    loss_rank1 = spectral_operator_entropy_loss(K_rank1)
    # Deficiency should equal log(D)
    assert abs(loss_rank1.item() - math.log(D)) < 1e-3

    # Degenerate energy collapse K = 0
    K_zero = torch.zeros(D, D)
    loss_zero = spectral_operator_entropy_loss(K_zero)
    # Loss should be huge due to logarithmic energy barrier
    assert loss_zero.item() > 10.0
    assert loss_zero.item() > loss_ideal.item()


def test_operator_entropy_jepa_forward_and_backward():
    """Verify forward pass, loss computation, and gradients for OperatorEntropyJEPAModel."""
    D = 32
    K_in = 5
    encoder = HybridTCNEncoder(
        input_dim=K_in,
        latent_dim=D,
        filters=32,
        tcn_layers=2,
        kernel_size=3,
    )
    model = OperatorEntropyJEPAModel(
        context_encoder=encoder,
        latent_dim=D,
        alpha_entropy=0.5,
        use_koopman_matrix=True,
    )

    B = 4
    x_ctx = torch.randn(B, 128, K_in)
    x_tgt = torch.randn(B, 32, K_in)

    z_pred, z_tgt, diff = model(x_ctx, x_tgt)
    assert z_pred.shape == (B, D)
    assert z_tgt.shape == (B, D)
    assert diff.shape == (B,)

    loss, metrics = model.compute_objective(x_ctx, x_tgt)
    assert "loss" in metrics
    assert "pred_loss" in metrics
    assert "spectral_reg" in metrics
    assert metrics["spectral_reg"] >= 0.0

    # Ensure gradients propagate to Koopman transition matrix K
    loss.backward()
    assert model.K.grad is not None
    assert model.K.grad.norm().item() > 0.0


def test_covariance_entropy_deficiency():
    """Verify that latent representation covariance collapse is penalized."""
    D = 16
    N = 64

    # Isotropic normal embeddings
    z_isotropic = torch.randn(N, D)
    loss_iso = von_neumann_entropy(z_isotropic)

    # Constant collapsed embeddings (all identical)
    z_collapsed = torch.zeros(N, D)
    loss_coll = von_neumann_entropy(z_collapsed)

    assert loss_coll.item() > loss_iso.item()

"""Tests for Resolvent-Purified Operator Precision (Chapter 6 of Ten Advances)."""

import pytest
import torch
import torch.nn as nn

from src.models._jepa_utils import fit_covariance_batched, resolvent_precision


class DummyModel(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.fc = nn.Linear(dim, dim)


def test_resolvent_precision_identity():
    dim = 8
    u = 1e-3
    cov = torch.eye(dim)
    prec = resolvent_precision(cov, u=u)

    expected_scale = 1.0 / (1.0 + u) ** 2
    assert torch.allclose(prec, expected_scale * torch.eye(dim), atol=1e-5)
    assert prec.shape == (dim, dim)


def test_resolvent_precision_singular_stability():
    dim = 8
    u = 1e-2
    # Rank-deficient covariance: only 4 non-zero eigenvalues
    A = torch.randn(dim, 4)
    cov = A @ A.T

    prec = resolvent_precision(cov, u=u)

    # Must be finite and non-negative definite
    assert torch.all(torch.isfinite(prec))
    evals = torch.linalg.eigvalsh(prec)
    assert torch.all(evals >= -1e-7)

    # Contrast with naive inverse which would blow up or produce huge condition number
    # Resolvent precision max eigenvalue should not exceed 1 / (4 * u)
    assert evals.max().item() <= 1.0 / (4.0 * u) + 1.0


def test_fit_covariance_batched_resolvent():
    dim = 6
    model = DummyModel(dim=dim)
    N = 100
    ctx = torch.randn(N, 10, dim)
    tgt = torch.randn(N, 5, dim)

    precision_buf = torch.eye(dim)
    mean_buf = torch.zeros(dim)
    fitted_buf = torch.tensor(False)

    def residual_fn(c, t):
        return c[:, 0, :] - t[:, 0, :]

    fit_covariance_batched(
        model,
        ctx,
        tgt,
        residual_fn=residual_fn,
        dim=dim,
        batch_size=32,
        reg=1e-3,
        method="resolvent",
        precision_buffer=precision_buf,
        residual_mean_buffer=mean_buf,
        fitted_buffer=fitted_buf,
    )

    assert fitted_buf.item() is True
    assert torch.all(torch.isfinite(precision_buf))
    assert torch.all(torch.isfinite(mean_buf))

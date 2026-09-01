"""Regression tests for the NCAD-Flow-JEPA variant.

Verifies:
- End-to-end training + Mahalanobis scoring.
- Scoring without Mahalanobis.
- Validation is deterministic.
- Validation uses the joint objective (flow + contrastive), not flow-only.
- Registry/builder wiring.
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import CSMConfig
from src.engine.trainer import build_ts_jepa_model, evaluate_ts_jepa_loss, train_ts_jepa
from src.models.jepa.flow_ts_jepa import flow_matching_vicreg_loss
from src.models.jepa.ncad_flow_jepa import NCADFlowJEPAModel
from src.models.registry import canonical_model_choices, canonical_model_type, is_jepa_model


def _make_windows(n: int = 14, length: int = 12, dim: int = 3) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, length, dim)).astype(np.float32)


def _base_config(**overrides) -> CSMConfig:
    defaults = dict(
        model_type="ncad_flow_jepa",
        context_size=8,
        suspect_size=4,
        patch_size=4,
        latent_dim=4,
        filters=8,
        tcn_layers=1,
        kernel_size=3,
        dropout=0.0,
        epochs=1,
        batch_size=4,
        val_split=0.2,
        use_mahalanobis=True,
    )
    defaults.update(overrides)
    return CSMConfig(**defaults)


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #


def test_ncad_flow_jepa_is_registered():
    """ncad_flow_jepa should resolve to a canonical name and be a JEPA model."""
    assert canonical_model_type("ncad_flow_jepa") == "ncad_flow_jepa"
    assert is_jepa_model("ncad_flow_jepa")
    assert "ncad_flow_jepa" in canonical_model_choices()


def test_ncad_flow_jepa_alias_registered():
    """The v1 alias should also resolve."""
    assert canonical_model_type("ncad_flow_jepa_v1") == "ncad_flow_jepa"


# --------------------------------------------------------------------------- #
# End-to-end training + scoring
# --------------------------------------------------------------------------- #


def test_ncad_flow_jepa_trains_and_scores_with_mahalanobis():
    """ncad_flow_jepa should complete train_ts_jepa and produce discrepancy scores
    with use_mahalanobis=True."""
    windows = _make_windows()
    config = _base_config()
    model, history = train_ts_jepa(windows, config, 3, torch.device("cpu"))

    batch = torch.from_numpy(windows[:2])
    scores = model.compute_predictive_discrepancy(
        batch[:, :8], batch[:, 8:], use_mahalanobis=True
    )
    assert scores.shape == (2,)
    assert torch.all(torch.isfinite(scores))


def test_ncad_flow_jepa_scores_without_mahalanobis():
    """ncad_flow_jepa compute_predictive_discrepancy should accept use_mahalanobis=False."""
    windows = _make_windows()
    config = _base_config(use_mahalanobis=False)
    model, _ = train_ts_jepa(windows, config, 3, torch.device("cpu"))

    batch = torch.from_numpy(windows[:2])
    scores = model.compute_predictive_discrepancy(
        batch[:, :8], batch[:, 8:], use_mahalanobis=False
    )
    assert scores.shape == (2,)
    assert torch.all(torch.isfinite(scores))


# --------------------------------------------------------------------------- #
# Validation determinism
# --------------------------------------------------------------------------- #


def test_ncad_flow_jepa_validation_is_deterministic():
    """evaluate_ts_jepa_loss should return the same value on repeated calls."""
    windows = _make_windows()
    config = _base_config(use_mahalanobis=False)
    model = build_ts_jepa_model(config, 3, torch.device("cpu"))

    v1 = evaluate_ts_jepa_loss(model, windows, config, torch.device("cpu"))
    v2 = evaluate_ts_jepa_loss(model, windows, config, torch.device("cpu"))
    assert abs(v1 - v2) < 1e-6, f"Validation non-deterministic: {v1} vs {v2}"


# --------------------------------------------------------------------------- #
# Validation uses the joint objective (flow + contrastive)
# --------------------------------------------------------------------------- #


def test_ncad_flow_jepa_validation_uses_joint_loss():
    """ncad_flow_jepa validation should include the contrastive term, so the
    validation loss should differ from a pure flow-only loss on the same data."""
    windows = _make_windows()
    config = _base_config(use_mahalanobis=False)
    model = build_ts_jepa_model(config, 3, torch.device("cpu"))

    joint_val_loss = evaluate_ts_jepa_loss(model, windows, config, torch.device("cpu"))

    # Compute a pure flow-only loss on the same data for comparison
    model.eval()
    flow_only_losses = []
    with torch.no_grad():
        for i in range(0, len(windows), config.batch_size):
            batch_arr = windows[i : i + config.batch_size]
            ctx = torch.from_numpy(batch_arr[:, :8]).float()
            tgt = torch.from_numpy(batch_arr[:, 8:]).float()
            B = ctx.size(0)
            t_val = torch.full((B,), 0.5, dtype=ctx.dtype)
            z_zero = torch.zeros(B, model.latent_dim, dtype=ctx.dtype)
            z_ctx, z_tgt_true, v_pred, v_target = model(ctx, tgt, t=t_val, z_noise=z_zero)
            loss, _ = flow_matching_vicreg_loss(
                v_pred=v_pred, v_target=v_target, z_ctx=z_ctx, z_tgt_true=z_tgt_true,
                flow_weight=config.vicreg_sim_weight,
                var_weight=config.vicreg_var_weight,
                cov_weight=config.vicreg_cov_weight,
            )
            flow_only_losses.append(float(loss.item()) * len(batch_arr))
    flow_only_loss = sum(flow_only_losses) / max(len(windows), 1)

    assert abs(joint_val_loss - flow_only_loss) > 1e-4, (
        f"Validation loss ({joint_val_loss:.6f}) matches flow-only ({flow_only_loss:.6f}); "
        "the contrastive term is likely not being applied during validation."
    )


# --------------------------------------------------------------------------- #
# Direct model construction
# --------------------------------------------------------------------------- #


def test_ncad_flow_jepa_direct_construction():
    """NCADFlowJEPAModel should be constructible directly and forward-passable."""
    model = NCADFlowJEPAModel(input_dim=3, latent_dim=4, filters=8, tcn_layers=1)
    ctx = torch.randn(4, 8, 3)
    tgt = torch.randn(4, 4, 3)
    z_ctx, z_tgt_true, v_pred, v_target = model(ctx, tgt)
    assert z_ctx.shape == (4, 4)
    assert z_tgt_true.shape == (4, 4)
    assert v_pred.shape == (4, 4)
    assert v_target.shape == (4, 4)

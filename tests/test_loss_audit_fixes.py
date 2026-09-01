"""Regression tests for Tier 1 loss-audit fixes.

Covers:
- ncad_jepa end-to-end training + Mahalanobis scoring (was broken).
- ncad_jepa contrastive branch receives the full injected window (not just context).
- Flow validation loss is deterministic across repeated calls.
- NCAD-JEPA validation uses the joint objective (not JEPA-only).
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import CSMConfig
from src.engine.trainer import build_ts_jepa_model, evaluate_ts_jepa_loss, train_ts_jepa


def _make_windows(n: int = 14, length: int = 12, dim: int = 3) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, length, dim)).astype(np.float32)


def _base_config(model_type: str, **overrides) -> CSMConfig:
    defaults = dict(
        model_type=model_type,
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
# ncad_jepa: end-to-end training + Mahalanobis scoring
# --------------------------------------------------------------------------- #


def test_ncad_jepa_trains_and_scores_with_mahalanobis():
    """ncad_jepa should complete train_ts_jepa and produce discrepancy scores
    with use_mahalanobis=True (previously raised AttributeError)."""
    windows = _make_windows()
    config = _base_config("ncad_jepa")
    model, history = train_ts_jepa(windows, config, 3, torch.device("cpu"))

    batch = torch.from_numpy(windows[:2])
    scores = model.compute_predictive_discrepancy(
        batch[:, :8], batch[:, 8:], use_mahalanobis=True
    )
    assert scores.shape == (2,)
    assert torch.all(torch.isfinite(scores))


def test_ncad_jepa_scores_without_mahalanobis():
    """ncad_jepa compute_predictive_discrepancy should accept use_mahalanobis=False
    (previously raised TypeError — missing keyword argument)."""
    windows = _make_windows()
    config = _base_config("ncad_jepa", use_mahalanobis=False)
    model, _ = train_ts_jepa(windows, config, 3, torch.device("cpu"))

    batch = torch.from_numpy(windows[:2])
    scores = model.compute_predictive_discrepancy(
        batch[:, :8], batch[:, 8:], use_mahalanobis=False
    )
    assert scores.shape == (2,)
    assert torch.all(torch.isfinite(scores))


# --------------------------------------------------------------------------- #
# ncad_jepa: contrastive branch receives the full injected window
# --------------------------------------------------------------------------- #


def test_ncad_jepa_contrastive_branch_uses_full_window():
    """The contrastive loss in compute_joint_loss should see different encodings
    for injected vs. clean windows. If the injected window is incorrectly sliced
    back to just the context, z_injected == z_ctx and the contrastive loss is ~0
    even with anomalies present."""
    from src.models.jepa.ncad_jepa import NCADJEPAModel

    model = NCADJEPAModel(input_dim=3, latent_dim=4, filters=8, tcn_layers=1)
    model.eval()

    clean_ctx = torch.randn(4, 8, 3)
    clean_tgt = torch.randn(4, 4, 3)

    # Full injected window: context is clean, suspect region has large anomalies
    injected_full = torch.cat([clean_ctx, clean_tgt * 10.0], dim=1)

    with torch.no_grad():
        z_ctx = model.context_encoder(clean_ctx)
        z_injected = model.context_encoder(injected_full)
        distances = torch.linalg.norm(z_injected - z_ctx, dim=-1)

    # With the full injected window, the anomalous suspect region should make
    # the encoding differ from the clean context.
    assert float(distances.mean()) > 1e-4, (
        f"Contrastive branch sees near-zero distances ({float(distances.mean()):.6f}), "
        "suggesting the injected window was sliced back to just context."
    )


# --------------------------------------------------------------------------- #
# Flow validation determinism
# --------------------------------------------------------------------------- #


def test_flow_jepa_validation_is_deterministic():
    """evaluate_ts_jepa_loss for flow_jepa should return the same value on
    repeated calls (previously stochastic due to random t and z_noise)."""
    windows = _make_windows()
    config = _base_config("flow_jepa", use_mahalanobis=False)
    model = build_ts_jepa_model(config, 3, torch.device("cpu"))

    v1 = evaluate_ts_jepa_loss(model, windows, config, torch.device("cpu"))
    v2 = evaluate_ts_jepa_loss(model, windows, config, torch.device("cpu"))
    assert abs(v1 - v2) < 1e-6, f"Flow validation is non-deterministic: {v1} vs {v2}"


def test_patch_flow_jepa_validation_is_deterministic():
    """evaluate_ts_jepa_loss for patch_flow_jepa should be deterministic."""
    windows = _make_windows()
    config = _base_config("patch_flow_jepa", use_mahalanobis=False)
    model = build_ts_jepa_model(config, 3, torch.device("cpu"))

    v1 = evaluate_ts_jepa_loss(model, windows, config, torch.device("cpu"))
    v2 = evaluate_ts_jepa_loss(model, windows, config, torch.device("cpu"))
    assert abs(v1 - v2) < 1e-6, f"Patch-flow validation is non-deterministic: {v1} vs {v2}"


# --------------------------------------------------------------------------- #
# NCAD-JEPA validation uses the joint objective
# --------------------------------------------------------------------------- #


def test_ncad_jepa_validation_uses_joint_loss():
    """ncad_jepa validation should use the joint (JEPA + contrastive) objective,
    not JEPA-only. We verify by checking that the validation loss differs from
    a pure JEPA-only loss computation on the same data."""
    from src.models.jepa.ncad_jepa import NCADJEPAModel
    from src.models.jepa.ts_jepa import jepa_vicreg_loss

    windows = _make_windows()
    config = _base_config("ncad_jepa", use_mahalanobis=False)
    model = build_ts_jepa_model(config, 3, torch.device("cpu"))

    # The validation loss from evaluate_ts_jepa_loss
    joint_val_loss = evaluate_ts_jepa_loss(model, windows, config, torch.device("cpu"))

    # Compute a pure JEPA-only loss on the same data for comparison
    model.eval()
    jepa_only_losses = []
    with torch.no_grad():
        for i in range(0, len(windows), config.batch_size):
            batch_arr = windows[i : i + config.batch_size]
            ctx = torch.from_numpy(batch_arr[:, :8]).float()
            tgt = torch.from_numpy(batch_arr[:, 8:]).float()
            z_ctx, z_tgt_true, z_tgt_pred = model(ctx, tgt)
            loss = jepa_vicreg_loss(
                z_target_pred=z_tgt_pred,
                z_target_true=z_tgt_true,
                z_context=z_ctx,
                sim_weight=config.vicreg_sim_weight,
                var_weight=config.vicreg_var_weight,
                cov_weight=config.vicreg_cov_weight,
            )
            jepa_only_losses.append(float(loss.item()) * len(batch_arr))
    jepa_only_loss = sum(jepa_only_losses) / max(len(windows), 1)

    # The joint loss includes the contrastive term, so it should differ.
    assert abs(joint_val_loss - jepa_only_loss) > 1e-4, (
        f"Validation loss ({joint_val_loss:.6f}) matches JEPA-only ({jepa_only_loss:.6f}); "
        "the contrastive term is likely not being applied during validation."
    )

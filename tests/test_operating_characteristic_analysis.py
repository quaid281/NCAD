import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_operating_characteristic_analysis import (
    bh_adjust,
    paired_realized_test,
    realized_fpr_analysis,
    realized_fpr_metrics,
)
from scripts.run_multiseed_heldout_benchmark import TCNObsForecasterBounded


def test_realized_auc_uses_fpr_axis_not_budget_axis():
    curve = pd.DataFrame({
        "realized_fpr": [0.0, 0.10, 0.15, 0.20, 0.40, 0.50],
        "event_recall": [0.0, 0.2, 0.8, 1.0, 1.0, 1.0],
        "q": [0.001, 0.003, 0.01, 0.05, 0.3, 0.5],
    })
    result = realized_fpr_metrics(curve)
    assert result["supported"]
    assert result["realized_pauc"] == pytest.approx(0.925, abs=0.002)
    assert result["recall_at_hi"] == pytest.approx(1.0)
    assert not realized_fpr_metrics(curve[curve.realized_fpr < 0.5])["supported"]


def test_realized_auc_ties_and_no_extrapolation():
    curve = pd.DataFrame({"realized_fpr": [0.1, 0.1, 0.5],
                          "event_recall": [0.1, 0.4, 0.8]})
    result = realized_fpr_metrics(curve)
    assert result["realized_pauc"] == pytest.approx(0.6)
    assert np.isnan(realized_fpr_metrics(curve[curve.realized_fpr == 0.1])
                    ["realized_pauc"])


def test_paired_seeds_and_dataset_level_sign_permutation():
    rows = []
    for ds in ("A", "B", "C"):
        for seed in (1, 2):
            rows.append(dict(dataset=ds, channel="one", seed=seed,
                             model="TS-JEPA", mapping="trailing", realized_pauc=0.8))
            rows.append(dict(dataset=ds, channel="one", seed=seed,
                             model="TimesNet", mapping="trailing",
                             realized_pauc=0.4 if seed == 1 else np.nan))
    result = paired_realized_test(pd.DataFrame(rows), "TimesNet", "trailing",
                                  "realized_pauc", n_boot=100)
    assert result["n_paired_seeds"] == 3
    assert result["n_streams"] == 3
    assert result["n_datasets"] == 3
    assert result["delta"] == pytest.approx(0.4)
    assert result["permutation_p"] == pytest.approx(0.25)
    assert result["ci_lo"] == pytest.approx(0.4)


def test_bh_family_and_missing_support():
    assert bh_adjust([0.01, 0.04, np.nan, 0.03]).tolist()[:2] == pytest.approx([0.03, 0.04])
    long = pd.DataFrame([
        dict(dataset="A", channel="one", seed=1, model=m, mapping="trailing",
             realized_fpr=fpr, event_recall=fpr, q=i / 10)
        for m in ("TS-JEPA", "TimesNet")
        for i, fpr in enumerate((0.1, 0.5) if m == "TS-JEPA" else (0.12, 0.5))
    ])
    cells, tests = realized_fpr_analysis(long)
    assert cells.supported.sum() == 1
    assert tests.n_streams.eq(0).all()
    assert tests.bh_p.isna().all()


def test_bounded_observation_forecaster_matches_gate_contract():
    model = TCNObsForecasterBounded(input_dim=5, context_len=16,
                                    suspect_len=8, latent_dim=16)
    ctx, target = torch.randn(3, 16, 5), torch.randn(3, 8, 5)
    pred = model(ctx)
    score = model.discrepancy(ctx, target)
    assert pred.shape == (3, 8, 5)
    assert score.shape == (3,)
    assert torch.isfinite(score).all()
    assert pred.flatten(1).mean(-1).abs().max() < 1e-4

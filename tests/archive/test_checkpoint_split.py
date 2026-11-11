import numpy as np
import pandas as pd
import pytest

from scripts.reviewer_analyses.benchmark_141_uncertainty import (
    REQUIRED_MODELS,
    SEEDS,
    bh_adjust,
    load_stream_means,
    validate_complete_archive,
)
from scripts.run_tier2_exhaustive_benchmark import W, split_train_checkpoint_block


def test_checkpoint_split_is_chronological_and_keeps_calibration_separate():
    nominal = np.arange(1917, dtype=np.float32)[:, None]
    original_train = nominal[: int(0.65 * len(nominal))]
    fit, select = split_train_checkpoint_block(original_train, len(nominal))

    assert len(select) >= W + 64
    assert len(fit) >= W + 16
    np.testing.assert_array_equal(np.concatenate((fit, select)), original_train)
    assert fit[-1, 0] < select[0, 0] < nominal[len(original_train), 0]


def test_checkpoint_split_refuses_short_training_history():
    nominal = np.arange(600, dtype=np.float32)[:, None]
    with pytest.raises(ValueError, match="checkpoint"):
        split_train_checkpoint_block(nominal[: int(0.65 * len(nominal))], len(nominal))


def test_uncertainty_reader_uses_selected_benchmark_artifact(tmp_path):
    path = tmp_path / "independent.csv"
    pd.DataFrame([
        dict(model="Speculative-224Bit-Breaker", calibrator="CircuitBreaker", seed=42),
        dict(model="TS-JEPA", calibrator="SPOT", seed=42),
    ]).assign(mapping="trailing", dataset="SMAP", channel="A-1", f1=0.2,
             event_recall=0.5, mean_delay=10.0, hold_fpr=0.01, pr_auc=0.3).to_csv(path, index=False)
    means = load_stream_means(path)
    assert set(means.model) == {"Speculative-224Bit-Breaker", "TS-JEPA"}
    assert len(means) == 2


def test_focal_pair_bh_adjustment_preserves_input_order():
    assert np.allclose(bh_adjust(np.array([0.016, 0.008])), [0.016, 0.016])
    assert np.allclose(bh_adjust(np.array([0.9, 0.01, 0.03])), [0.9, 0.03, 0.045])


def test_archive_statistics_refuse_partial_or_duplicated_cells():
    rows = pd.DataFrame([
        dict(dataset="SMAP", channel="A-1", model=model, seed=seed, mapping=mapping,
             calibrator="CircuitBreaker" if "224Bit" in model else "SPOT")
        for model in REQUIRED_MODELS for seed in SEEDS for mapping in ("trailing", "smear")
    ])
    manifest = {("SMAP", "A-1")}
    validate_complete_archive(rows, manifest)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_complete_archive(rows.iloc[:-1], manifest)
    with pytest.raises(ValueError, match="duplicated"):
        validate_complete_archive(pd.concat((rows, rows.iloc[:1])), manifest)

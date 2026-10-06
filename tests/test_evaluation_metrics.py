"""
Comprehensive Unit Tests for Anomaly Detection Metric Pipeline
-----------------------------------------------------------------
Validates:
1. Exact Point-F1, Precision, Recall, and Confusion Matrix consistency.
2. Point Adjustment (PA) mechanics and edge cases.
3. Valid mask interactions and temporal support boundary handling.
4. PR-AUC (Average Precision) and ROC-AUC calculations on continuous scores.
5. All-positive reference baseline behavior under extreme class imbalance.
6. Multi-channel aggregation (Dataset-Macro, Channel-Macro, and Pooled Micro-F1).
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score

from src.scoring.event_fusion import point_adjustment, compute_metrics


class TestPointAdjustment:
    def test_single_hit_in_segment_expands(self):
        # Segment from idx 3 to 7 (len 5)
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=float)
        # Model only detected at idx 5
        preds = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

        adjusted = point_adjustment(labels, preds)
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=float)
        np.testing.assert_array_equal(adjusted, expected)

    def test_missed_segment_remains_zero(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=float)
        preds = np.zeros_like(labels)

        adjusted = point_adjustment(labels, preds)
        np.testing.assert_array_equal(adjusted, np.zeros_like(labels))

    def test_false_positives_outside_segment_preserved(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=float)
        # Hits inside at 4, but false alarm at 1 and 8
        preds = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)

        adjusted = point_adjustment(labels, preds)
        # Segment 3..5 expanded, 1 and 8 stay 1
        expected = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0], dtype=float)
        np.testing.assert_array_equal(adjusted, expected)

    def test_segment_at_very_end(self):
        labels = np.array([0, 0, 1, 1, 1], dtype=float)
        preds = np.array([0, 0, 0, 1, 0], dtype=float)

        adjusted = point_adjustment(labels, preds)
        expected = np.array([0, 0, 1, 1, 1], dtype=float)
        np.testing.assert_array_equal(adjusted, expected)

    def test_multiple_segments_selective_hit(self):
        # Seg 1: 2..4, Seg 2: 7..9
        labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=float)
        # Hits Seg 1 only at idx 3
        preds = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)

        adjusted = point_adjustment(labels, preds)
        expected = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        np.testing.assert_array_equal(adjusted, expected)


class TestComputeMetrics:
    def test_strict_metrics_match_sklearn(self):
        labels = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=float)
        preds = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0], dtype=float)

        m = compute_metrics(labels, preds, use_pa=False)
        assert m["tp"] == 3
        assert m["fp"] == 2
        assert m["fn"] == 1
        assert m["tn"] == 4

        expected_prec = precision_score(labels, preds)
        expected_rec = recall_score(labels, preds)
        expected_f1 = f1_score(labels, preds)

        assert pytest.approx(m["precision"]) == expected_prec
        assert pytest.approx(m["recall"]) == expected_rec
        assert pytest.approx(m["f1"]) == expected_f1

        # Check exact 2TP / (2TP + FP + FN)
        manual_f1 = (2 * m["tp"]) / (2 * m["tp"] + m["fp"] + m["fn"])
        assert pytest.approx(m["f1"]) == manual_f1

    def test_point_adjustment_metric_flow(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 0, 0], dtype=float)
        # Only hit index 3
        preds = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

        m_pt = compute_metrics(labels, preds, use_pa=False)
        assert m_pt["tp"] == 1
        assert m_pt["fn"] == 2
        assert m_pt["recall"] == 1 / 3

        m_pa = compute_metrics(labels, preds, use_pa=True)
        assert m_pa["tp"] == 3
        assert m_pa["fn"] == 0
        assert m_pa["recall"] == 1.0
        assert m_pa["f1"] == 1.0

    def test_valid_mask_support(self):
        labels = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=float)
        preds = np.array([1, 0, 0, 0, 1, 1, 0, 0], dtype=float)
        # Mask out first 2 points
        mask = np.array([False, False, True, True, True, True, True, True])

        m = compute_metrics(labels, preds, valid_mask=mask, use_pa=False)
        # Evaluated points: [0, 0, 1, 1, 0, 0] vs [0, 0, 1, 1, 0, 0] -> perfect match!
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["tn"] == 4
        assert m["f1"] == 1.0

    def test_continuous_scores_pr_auc_and_roc_auc(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=float)
        # High scores on anomalies
        scores = np.array([0.1, 0.2, 0.1, 0.8, 0.9, 0.7, 0.3, 0.2], dtype=float)
        preds = (scores > 0.5).astype(float)

        m = compute_metrics(labels, preds, scores=scores, use_pa=False)
        assert "pr_auc" in m
        assert "roc_auc" in m
        assert m["pr_auc"] > 0.8
        assert m["roc_auc"] == 1.0  # Perfect separation

    def test_all_positive_sanity_reference(self):
        # 10% prevalence
        labels = np.zeros(100, dtype=float)
        labels[:10] = 1.0
        preds = np.ones(100, dtype=float)

        m = compute_metrics(labels, preds, use_pa=False)
        expected_f1 = (2 * 10) / (2 * 10 + 90 + 0)  # 20 / 110 = 0.1818
        assert pytest.approx(m["f1"], rel=1e-3) == expected_f1
        assert "all_positive_f1" in m
        assert pytest.approx(m["all_positive_f1"], rel=1e-3) == expected_f1


class TestMultiTierAggregation:
    def test_dataset_macro_vs_channel_macro(self):
        """Simulates 2 datasets:
        Dataset A has 10 channels, all scoring F1 = 0.20
        Dataset B has 1 channel, scoring F1 = 0.80
        Channel-macro = (10*0.2 + 0.8) / 11 = 0.2545
        Dataset-macro = (0.2 + 0.8) / 2 = 0.5000
        """
        data = []
        for i in range(10):
            data.append({"dataset": "A", "channel": f"ch_{i}", "f1": 0.20, "tp": 20, "fp": 80, "fn": 80})
        data.append({"dataset": "B", "channel": "ch_0", "f1": 0.80, "tp": 80, "fp": 20, "fn": 20})

        import pandas as pd
        df = pd.DataFrame(data)

        channel_macro = df["f1"].mean()
        dataset_macro = df.groupby("dataset")["f1"].mean().mean()

        total_tp = df["tp"].sum()
        total_fp = df["fp"].sum()
        total_fn = df["fn"].sum()
        pooled_micro = (2 * total_tp) / (2 * total_tp + total_fp + total_fn)

        assert pytest.approx(channel_macro, rel=1e-3) == 0.2545
        assert pytest.approx(dataset_macro, rel=1e-3) == 0.5000
        assert pytest.approx(pooled_micro, rel=1e-3) == (2 * 280) / (2 * 280 + 820 + 820)


class TestEventAndRangeMetrics:
    def test_event_metrics_exact_hit(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=float)
        # Hits both events with slight boundary offset
        preds = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1], dtype=float)

        from src.scoring.event_fusion import compute_event_metrics, compute_range_metrics
        ev = compute_event_metrics(labels, preds)
        assert ev["event_precision"] == 1.0
        assert ev["event_recall"] == 1.0
        assert ev["event_f1"] == 1.0
        assert ev["tp_events"] == 2
        assert ev["fp_events"] == 0
        assert ev["fn_events"] == 0

        rng = compute_range_metrics(labels, preds)
        assert rng["range_recall"] > 0.7
        assert rng["range_precision"] > 0.6
        assert rng["range_f1"] > 0.65

    def test_event_metrics_false_alarm(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
        # Hits real event at idx 3, but also alarms at idx 7..8 (false alarm)
        preds = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0], dtype=float)

        from src.scoring.event_fusion import compute_event_metrics
        ev = compute_event_metrics(labels, preds)
        assert ev["tp_events"] == 1
        assert ev["fp_events"] == 1
        assert ev["fn_events"] == 0
        assert ev["event_precision"] == 0.5
        assert ev["event_recall"] == 1.0
        assert pytest.approx(ev["event_f1"], rel=1e-3) == 0.6667


class TestDynamicEVTThreshold:
    def test_dynamic_threshold_adapts_to_baseline_drift(self):
        from src.scoring.evt_calibrator import compute_dynamic_evt_threshold

        np.random.seed(42)
        train_scores = np.abs(np.random.normal(0, 1, 2000))
        base_threshold = float(np.percentile(train_scores, 99.0))

        # Test series with step drift of +3.0
        test_scores = np.abs(np.random.normal(0, 1, 2000))
        test_scores[1000:] += 3.0

        dyn_th = compute_dynamic_evt_threshold(test_scores, base_threshold, train_scores, window_size=100)
        assert len(dyn_th) == len(test_scores)
        # Threshold in second half should track higher than in first half
        assert np.mean(dyn_th[1200:]) > np.mean(dyn_th[:800])
        # False alarm count should be drastically lower than static threshold
        static_fps = np.sum(test_scores > base_threshold)
        dyn_fps = np.sum(test_scores > dyn_th)
        assert dyn_fps < static_fps


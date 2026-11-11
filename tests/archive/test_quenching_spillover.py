import numpy as np

from scripts.reviewer_analyses.quenching_spillover import relative_coordinate_changes


def test_quenching_spillover_distinguishes_selected_and_other_coordinates():
    before = np.ones((4, 2, 3), dtype=np.float32)
    after = before.copy()
    after[:, :, 0] += 1.0
    after[:, :, 1] += 0.25
    metrics = relative_coordinate_changes(before, after, [0])
    assert metrics["selected_median"] == 1.0
    assert metrics["other_median"] == 0.125
    assert metrics["other_above_10pct"] == 0.5


def test_no_quench_has_zero_spillover():
    before = np.ones((3, 4, 2), dtype=np.float32)
    metrics = relative_coordinate_changes(before, before, [])
    assert metrics["selected_median"] == 0
    assert metrics["other_median"] == 0

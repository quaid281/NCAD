import numpy as np

from scripts.reviewer_analyses.channel_branch_evaluation import inject_single_channel_fault


def test_injected_events_do_not_overwrite_natural_anomalies():
    x = np.zeros((3000, 3), dtype=np.float32)
    y = np.zeros(len(x), dtype=np.int64)
    y[400:600] = 1
    y[1600:1750] = 1

    injected, labels = inject_single_channel_fault(
        x, y, 1, "6_isolated_sensor_bias", x[:500]
    )
    added = (labels == 1) & (y == 0)
    assert np.array_equal(labels[y == 1], y[y == 1])
    assert added.any()
    assert not np.any((injected != x)[y == 1])
    assert np.array_equal(injected[:, [0, 2]], x[:, [0, 2]])
    assert np.all((injected[:, 1] != x[:, 1]) == added)


def test_short_nominal_regions_are_not_mislabeled_as_faults():
    x = np.zeros((800, 2), dtype=np.float32)
    y = np.ones(len(x), dtype=np.int64)
    injected, labels = inject_single_channel_fault(
        x, y, 0, "1_variance_collapse", x[:300]
    )
    assert np.array_equal(labels, y)
    assert np.array_equal(injected, x)

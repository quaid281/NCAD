import numpy as np
import pytest

from scripts.reviewer_analyses.matched_alert_rate_controls import threshold_for_calibration_rate


def test_nominal_calibration_rate_uses_only_calibration_statistic():
    calibration = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
    assert threshold_for_calibration_rate(calibration, 0.2) == pytest.approx(1.2)
    assert np.mean(calibration > threshold_for_calibration_rate(calibration, 0.2)) == pytest.approx(0.2)
    assert np.mean(calibration > threshold_for_calibration_rate(calibration, 0.0)) == 0


def test_nominal_calibration_rate_refuses_invalid_targets():
    with pytest.raises(ValueError):
        threshold_for_calibration_rate(np.array([1.0]), 1.1)

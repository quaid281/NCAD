"""Split Conformal Prediction Calibrator for Time-Series Anomaly Detection.

Provides distribution-free, finite-sample statistical guarantees on false alarm rates:
P(S_test > q_{1-alpha} | Nominal) <= alpha

Under the exchangeability of nominal calibration residuals, the split conformal threshold
ensures exact risk coverage without parametric distributional assumptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ConformalThresholdResult:
    threshold: float
    significance_level: float  # alpha (target false alarm rate, e.g. 0.01 or 0.05)
    n_calibration_samples: int
    is_calibrated: bool

    def to_dict(self) -> dict:
        return {
            "threshold": float(self.threshold),
            "significance_level": float(self.significance_level),
            "n_calibration_samples": int(self.n_calibration_samples),
            "is_calibrated": bool(self.is_calibrated),
        }


class SplitConformalCalibrator:
    """Distribution-Free Split Conformal Calibrator.

    Calibrates decision thresholds over nominal validation/training non-conformity scores
    (e.g., flow velocity discrepancy, Mahalanobis energy) to guarantee bounded false alarm rates.
    """

    def __init__(self, alpha: float = 0.01):
        """Initialize calibrator.

        Args:
            alpha: Desired upper bound on false positive rate (significance level). Default 0.01 (1%).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.calibration_scores: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.is_calibrated: bool = False

    def _to_numpy_1d(self, x: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy().astype(np.float64).ravel()
        elif isinstance(x, np.ndarray):
            arr = x.astype(np.float64).ravel()
        else:
            arr = np.asarray(x, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        return arr

    def calibrate(
        self,
        calibration_scores: Union[np.ndarray, torch.Tensor, list],
        alpha: Optional[float] = None,
    ) -> ConformalThresholdResult:
        """Fit conformal quantile threshold on nominal calibration scores.

        Args:
            calibration_scores: 1D array of non-conformity scores from nominal validation data.
            alpha: Optional override for significance level.

        Returns:
            ConformalThresholdResult with the calibrated threshold.
        """
        if alpha is not None:
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha must be in (0, 1), got {alpha}")
            self.alpha = float(alpha)

        scores = self._to_numpy_1d(calibration_scores)
        n = len(scores)
        if n == 0:
            raise ValueError("Cannot calibrate conformal threshold on empty or all-NaN score array.")

        self.calibration_scores = scores

        # Exact finite-sample conformal quantile index: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.threshold = float(np.quantile(scores, level, method="higher"))
        self.is_calibrated = True

        logger.info(
            f"Conformal calibration complete: n={n}, alpha={self.alpha:.4f}, "
            f"quantile_level={level:.4f}, threshold={self.threshold:.6f}"
        )

        return ConformalThresholdResult(
            threshold=self.threshold,
            significance_level=self.alpha,
            n_calibration_samples=n,
            is_calibrated=True,
        )

    def predict(self, test_scores: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
        """Predict binary anomaly flags for test scores (1 = anomaly, 0 = nominal).

        Args:
            test_scores: Array of test non-conformity scores.

        Returns:
            Boolean numpy array of shape (N,) indicating anomaly flags.
        """
        if not self.is_calibrated or self.threshold is None:
            raise RuntimeError("SplitConformalCalibrator has not been calibrated. Call calibrate() first.")
        scores = self._to_numpy_1d(test_scores)
        return scores > self.threshold

    def compute_p_values(self, test_scores: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
        """Compute empirical conformal p-values for test scores.

        p(s) = (1 + sum_{i=1}^n I(S_cal_i >= s)) / (n + 1)

        Small p-values (e.g. p < alpha) correspond to significant anomalies.
        """
        if not self.is_calibrated or self.calibration_scores is None:
            raise RuntimeError("SplitConformalCalibrator has not been calibrated. Call calibrate() first.")
        scores = self._to_numpy_1d(test_scores)
        cal = self.calibration_scores
        n = len(cal)

        cal_sorted = np.sort(cal)
        counts_geq = n - np.searchsorted(cal_sorted, scores, side="left")
        p_values = (1.0 + counts_geq) / (n + 1.0)
        return p_values.astype(np.float64)

    def empirical_false_alarm_rate(self, nominal_test_scores: Union[np.ndarray, torch.Tensor, list]) -> float:
        """Compute empirical false alarm rate on holdout nominal test scores."""
        preds = self.predict(nominal_test_scores)
        return float(np.mean(preds))

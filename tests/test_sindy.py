import pytest
import numpy as np

from src.models.sindy_scorer import SINDyConfig, SINDyDynamicsScorer


def test_sindy_polynomial_library():
    scorer = SINDyDynamicsScorer(SINDyConfig(poly_degree=2, include_constant=True))
    z = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    theta = scorer.build_library(z)

    # Expected columns: 1 constant + 2 linear + 3 degree-2 terms = 6 columns
    assert theta.shape == (2, 6)
    assert not np.any(np.isnan(theta))


def test_sindy_fit_and_score():
    scorer = SINDyDynamicsScorer(SINDyConfig(poly_degree=2, threshold=0.01))

    # Generate synthetic smooth trajectory z(t) = [sin(t), cos(t)]
    t = np.linspace(0, 10, 100)
    z_seq = np.column_stack([np.sin(t), np.cos(t)]).astype(np.float32)

    scorer.fit(z_seq)

    assert scorer.coefficients is not None
    assert scorer.coefficients.shape[1] == 2
    assert not np.any(np.isnan(scorer.coefficients))

    # Score normal trajectory (expect low residual)
    scores_normal = scorer.score(z_seq)
    assert scores_normal.shape == (100,)
    assert not np.any(np.isnan(scores_normal))
    assert np.mean(scores_normal) < 0.5

    # Score anomalous jump trajectory (expect higher residual at jump)
    z_anom = z_seq.copy()
    z_anom[50:55] += 10.0
    scores_anom = scorer.score(z_anom)
    assert np.max(scores_anom[50:55]) > np.mean(scores_normal)

import numpy as np
import pytest

from credit_risk.scorecard import Scorecard, percentile


@pytest.mark.parametrize("base,odds,pdo", [(600, 50, 20), (500, 10, 30), (0, 1, 5)])
def test_reference_and_doubling(base, odds, pdo):
    card = Scorecard(base, odds, pdo)
    assert card.score(1 / (1 + odds)) == pytest.approx(base)
    assert card.score(1 / (1 + 2 * odds)) == pytest.approx(base + pdo)
    assert card.score(1 / (1 + odds / 2)) == pytest.approx(base - pdo)


def test_roundtrip_monotonicity_and_endpoints():
    card = Scorecard()
    pd = np.linspace(0.00001, 0.99999, 1000)
    assert np.all(np.diff(card.score(pd)) < 0)
    np.testing.assert_allclose(card.pd(card.score(pd)), pd, atol=1e-14)
    assert np.isfinite(card.score([0, 1])).all()
    np.testing.assert_allclose(card.pd(card.score([0, 1])), [card.epsilon, 1 - card.epsilon])
    assert card.pd(-1e10) == 1
    assert card.pd(1e10) == 0


@pytest.mark.parametrize("pd", [-0.01, 1.01, np.nan, np.inf, [-1, 0.5]])
def test_invalid_probabilities(pd):
    with pytest.raises(ValueError):
        Scorecard().score(pd)


@pytest.mark.parametrize("kwargs", [dict(base_score=np.nan), dict(base_good_to_bad_odds=0),
                                   dict(points_to_double_good_to_bad_odds=-2), dict(epsilon=0.5)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Scorecard(**kwargs)


def test_percentile_ties_limits_and_monotone_scale_invariance():
    reference = [400, 500, 500, 600]
    np.testing.assert_allclose(percentile([399, 400, 500, 601], reference), [0, 25, 75, 100])
    p = np.array([0.05, 0.1, 0.2, 0.4])
    for card in [Scorecard(), Scorecard(700, 20, 30)]:
        assert percentile(card.score(0.1), card.score(p)) == 75
    for invalid in [[], [1, np.nan], [[1, 2]]]:
        with pytest.raises(ValueError):
            percentile(1, invalid)

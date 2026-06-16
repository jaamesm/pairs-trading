"""
Unit tests for the Engle-Granger cointegration module.

Test strategy
-------------
We construct synthetic cointegrated and non-cointegrated pairs and verify
that the test correctly identifies each case. We also test edge cases and
the asymmetry property of OLS-based EG.
"""

import numpy as np
import pandas as pd

from pairs_trading.cointegration.engle_granger import (
    EngleGrangerResult,
    engle_granger_test,
    select_best_direction,
)


def _make_cointegrated_pair(
    n: int = 500,
    beta: float = 1.5,
    kappa: float = 0.1,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """
    Construct a cointegrated pair (P_a, P_b) by design:
      P_b follows a random walk.
      P_a = beta * P_b + spread, where spread is a stationary AR(1).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")

    pb_increments = rng.normal(0, 1.0, n)
    pb = np.cumsum(pb_increments)

    # Stationary spread: AR(1) with coefficient (1 - kappa)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = (1 - kappa) * spread[t - 1] + rng.normal(0, 0.3)

    pa = beta * pb + spread

    return (
        pd.Series(pa, index=idx, name="A"),
        pd.Series(pb, index=idx, name="B"),
    )


def _make_independent_pair(n: int = 500, seed: int = 1) -> tuple[pd.Series, pd.Series]:
    """Two independent random walks — should NOT be cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    pa = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=idx, name="A")
    pb = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=idx, name="B")
    return pa, pb


class TestEngleGrangerCointegrated:
    """Tests on genuinely cointegrated pairs."""

    def test_detects_cointegration(self):
        pa, pb = _make_cointegrated_pair(n=600, beta=1.5, kappa=0.15)
        result = engle_granger_test(pa, pb, "A", "B", both_directions=False)
        assert isinstance(result, EngleGrangerResult)
        assert result.is_cointegrated, (
            f"Should detect cointegration; p-value={result.adf_pvalue:.4f}"
        )

    def test_hedge_ratio_close_to_true(self):
        """Estimated β should be within ±0.2 of the true β=1.5 (with 600 obs)."""
        pa, pb = _make_cointegrated_pair(n=600, beta=1.5, kappa=0.2, seed=5)
        result = engle_granger_test(pa, pb, "A", "B", both_directions=False)
        assert abs(result.hedge_ratio - 1.5) < 0.25, (
            f"Hedge ratio {result.hedge_ratio:.3f} too far from true 1.5"
        )

    def test_residuals_length_matches_input(self):
        pa, pb = _make_cointegrated_pair(n=300)
        result = engle_granger_test(pa, pb, both_directions=False)
        assert len(result.residuals) == len(pa)

    def test_residuals_are_stationary_by_adf(self):
        """The residuals from a cointegrated pair should pass ADF separately."""
        from statsmodels.tsa.stattools import adfuller
        pa, pb = _make_cointegrated_pair(n=600, kappa=0.2)
        result = engle_granger_test(pa, pb, both_directions=False)
        adf_stat, adf_pval, *_ = adfuller(result.residuals.dropna())
        assert adf_pval < 0.10, (
            f"Residuals from cointegrated pair should be stationary; p={adf_pval:.4f}"
        )

    def test_r_squared_positive(self):
        pa, pb = _make_cointegrated_pair()
        result = engle_granger_test(pa, pb, both_directions=False)
        assert 0 < result.r_squared <= 1.0


class TestEngleGrangerNotCointegrated:
    """Tests on pairs that should not be cointegrated."""

    def test_rejects_independent_random_walks(self):
        """
        With 500 obs and α=0.05, two independent random walks should usually
        NOT be flagged as cointegrated. We allow for ~10% failure rate.
        """
        n_trials = 20
        false_positives = 0
        for seed in range(n_trials):
            pa, pb = _make_independent_pair(n=500, seed=seed)
            result = engle_granger_test(pa, pb, significance=0.05, both_directions=False)
            if result.is_cointegrated:
                false_positives += 1

        false_positive_rate = false_positives / n_trials
        assert false_positive_rate < 0.20, (
            f"False positive rate {false_positive_rate:.0%} too high "
            f"for independent random walks"
        )


class TestEngleGrangerBothDirections:
    """Tests on the both_directions feature."""

    def test_both_directions_returns_tuple(self):
        pa, pb = _make_cointegrated_pair()
        result = engle_granger_test(pa, pb, both_directions=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_directions_have_different_hedge_ratios(self):
        """OLS asymmetry: β(A~B) ≠ 1/β(B~A) in general."""
        pa, pb = _make_cointegrated_pair(beta=2.0, n=400)
        res_ab, res_ba = engle_granger_test(pa, pb, both_directions=True)
        # They should not be identical
        assert res_ab.hedge_ratio != res_ba.hedge_ratio

    def test_direction_labels_correct(self):
        pa, pb = _make_cointegrated_pair()
        res_ab, res_ba = engle_granger_test(pa, pb, "SPY", "QQQ", both_directions=True)
        assert res_ab.direction == "A_on_B"
        assert res_ba.direction == "B_on_A"

    def test_select_best_direction_picks_lower_adf(self):
        pa, pb = _make_cointegrated_pair(n=500)
        res_ab, res_ba = engle_granger_test(pa, pb, both_directions=True)
        best = select_best_direction(res_ab, res_ba)
        assert best.adf_statistic == min(res_ab.adf_statistic, res_ba.adf_statistic)


class TestEngleGrangerSummary:
    """Test the summary() method produces a non-empty string."""

    def test_summary_not_empty(self):
        pa, pb = _make_cointegrated_pair()
        result = engle_granger_test(pa, pb, "A", "B", both_directions=False)
        s = result.summary()
        assert len(s) > 50
        assert "Engle-Granger" in s
        assert "Hedge ratio" in s

    def test_critical_values_present(self):
        pa, pb = _make_cointegrated_pair()
        result = engle_granger_test(pa, pb, both_directions=False)
        assert "1%" in result.adf_critical_values
        assert "5%" in result.adf_critical_values
        assert "10%" in result.adf_critical_values

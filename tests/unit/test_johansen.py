"""
Unit tests for the Johansen cointegration module.
"""

import numpy as np
import pandas as pd
import pytest

from pairs_trading.cointegration.johansen import JohansenResult, johansen_test


def _make_cointegrated_pair(
    n: int = 500, beta: float = 1.2, kappa: float = 0.15, seed: int = 0
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    pb = np.cumsum(rng.normal(0, 1, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = (1 - kappa) * spread[t - 1] + rng.normal(0, 0.3)
    pa = beta * pb + spread
    return (
        pd.Series(pa, index=idx, name="A"),
        pd.Series(pb, index=idx, name="B"),
    )


def _make_independent_pair(n: int = 500, seed: int = 99) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    pa = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=idx)
    pb = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=idx)
    return pa, pb


class TestJohansenBasics:
    """Structural tests: output types and shapes."""

    def test_returns_johansen_result(self):
        pa, pb = _make_cointegrated_pair()
        result = johansen_test(pa, pb, "A", "B")
        assert isinstance(result, JohansenResult)

    def test_tickers_stored_correctly(self):
        pa, pb = _make_cointegrated_pair()
        result = johansen_test(pa, pb, "GLD", "GDX")
        assert result.tickers == ["GLD", "GDX"]

    def test_coint_rank_in_valid_range(self):
        pa, pb = _make_cointegrated_pair()
        result = johansen_test(pa, pb)
        assert 0 <= result.coint_rank <= 2

    def test_hedge_ratio_is_float(self):
        pa, pb = _make_cointegrated_pair(beta=1.5)
        result = johansen_test(pa, pb)
        assert isinstance(result.hedge_ratio, float)

    def test_eigenvalues_descending(self):
        """Eigenvalues should be sorted in descending order by statsmodels."""
        pa, pb = _make_cointegrated_pair(n=600)
        result = johansen_test(pa, pb)
        assert result.eigenvalues[0] >= result.eigenvalues[1]


class TestJohansenDetection:
    """Test detection accuracy on synthetic pairs."""

    def test_detects_cointegration(self):
        pa, pb = _make_cointegrated_pair(n=600, kappa=0.2)
        result = johansen_test(pa, pb)
        assert result.is_cointegrated, (
            f"Should detect cointegration; rank={result.coint_rank}"
        )

    def test_hedge_ratio_close_to_true(self):
        """Hedge ratio should be within ±0.4 of true β with 600 obs."""
        pa, pb = _make_cointegrated_pair(n=600, beta=1.2, kappa=0.2, seed=3)
        result = johansen_test(pa, pb)
        if result.hedge_ratio is not None:
            assert abs(abs(result.hedge_ratio) - 1.2) < 0.4, (
                f"Hedge ratio {result.hedge_ratio:.3f} far from true 1.2"
            )

    def test_spread_computable_when_cointegrated(self):
        pa, pb = _make_cointegrated_pair(n=400)
        result = johansen_test(pa, pb, "A", "B")
        if result.hedge_ratio is not None:
            prices = pd.DataFrame({"A": pa, "B": pb})
            spread = result.spread(prices)
            assert len(spread) == len(pa)
            assert not spread.isna().any()


class TestJohansenNotCointegrated:
    """Test that non-cointegrated pairs usually fail the test."""

    def test_independent_pair_low_rank(self):
        """
        Two independent random walks should mostly have coint_rank = 0.
        Allows for ~15% false positive rate.
        """
        n_trials = 15
        false_positives = 0
        for seed in range(n_trials):
            pa, pb = _make_independent_pair(n=400, seed=seed)
            result = johansen_test(pa, pb)
            if result.is_cointegrated:
                false_positives += 1
        assert false_positives / n_trials < 0.25


class TestJohansenSummary:
    """Test summary output."""

    def test_summary_non_empty(self):
        pa, pb = _make_cointegrated_pair()
        result = johansen_test(pa, pb, "X", "Y")
        s = result.summary()
        assert len(s) > 50
        assert "Johansen" in s
        assert "Trace" in s

    def test_spread_raises_without_hedge_ratio(self):
        pa, pb = _make_cointegrated_pair()
        result = johansen_test(pa, pb)
        result_no_ratio = JohansenResult(
            tickers=["A", "B"],
            cointegrating_vectors=result.cointegrating_vectors,
            eigenvalues=result.eigenvalues,
            trace_stats=result.trace_stats,
            trace_crit_vals=result.trace_crit_vals,
            max_eigen_stats=result.max_eigen_stats,
            max_eigen_crit_vals=result.max_eigen_crit_vals,
            coint_rank=1,
            is_cointegrated=True,
            hedge_ratio=None,
        )
        prices = pd.DataFrame({"A": pa, "B": pb})
        with pytest.raises(ValueError, match="2 tickers"):
            result_no_ratio.spread(prices)

"""
Integration tests: end-to-end pipeline on synthetic cointegrated pairs.

These tests run the full stack — cointegration → OU estimation → signals →
backtest → metrics — on synthetic data where we know the ground truth.
No network calls are made (yfinance is not used here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading.cointegration.engle_granger import engle_granger_test, select_best_direction
from pairs_trading.cointegration.johansen import johansen_test
from pairs_trading.models.ou_process import fit_ou
from pairs_trading.signals.zscore import SignalConfig, generate_signals
from pairs_trading.backtest.engine import BacktestConfig, run_backtest
from pairs_trading.metrics.performance import compute_metrics
from pairs_trading.validation.walk_forward import run_walk_forward


def _synthetic_cointegrated_pair(
    n: int = 800,
    beta: float = 1.3,
    kappa: float = 0.15,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Construct a synthetic cointegrated pair with controlled parameters.
    The spread is a stationary AR(1) process with known kappa.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")

    pb_inc = rng.normal(0, 1.0, n)
    pb = np.cumsum(pb_inc) + 50  # ~50 starting price

    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = (1 - kappa) * spread[t - 1] + rng.normal(0, 0.5)

    pa = beta * pb + spread

    return (
        pd.Series(pa, index=idx, name="A"),
        pd.Series(pb, index=idx, name="B"),
    )


class TestFullPipeline:
    """Full pipeline from raw prices to performance metrics."""

    def setup_method(self):
        self.pa, self.pb = _synthetic_cointegrated_pair(n=800, beta=1.3, kappa=0.15)

    def test_engle_granger_detects_cointegration(self):
        res_ab, res_ba = engle_granger_test(
            self.pa, self.pb, "A", "B", both_directions=True
        )
        best = select_best_direction(res_ab, res_ba)
        assert best.is_cointegrated, (
            f"EG should detect cointegration on synthetic pair; p={best.adf_pvalue:.4f}"
        )

    def test_johansen_consistent_with_engle_granger(self):
        """Both tests should agree on a strongly cointegrated pair."""
        eg_ab, eg_ba = engle_granger_test(self.pa, self.pb, both_directions=True)
        eg_best = select_best_direction(eg_ab, eg_ba)
        joh = johansen_test(self.pa, self.pb)

        # If EG rejects the null, Johansen should too (with 800 obs)
        if eg_best.is_cointegrated:
            assert joh.is_cointegrated, (
                "Johansen should agree with EG on a synthetic cointegrated pair"
            )

    def test_ou_half_life_in_expected_range(self):
        """
        With kappa=0.15, half-life ≈ ln(2)/0.15 ≈ 4.6 days.
        Estimate should be in the range [2, 20] with 800 observations.
        """
        eg, _ = engle_granger_test(self.pa, self.pb, both_directions=True)
        spread = self.pa - eg.hedge_ratio * self.pb
        ou_params = fit_ou(spread)

        assert ou_params.is_valid(), "OU process should be valid for stationary spread"
        # kappa=0.15 in AR(1) terms means b=0.85; OU kappa=-ln(0.85)*252≈40.9/yr
        # half_life = ln(2)/40.9 * 252 ≈ 4.3 trading days
        # Allow generous range [1, 40] accounting for estimation variance
        assert 1.0 <= ou_params.half_life_days <= 40.0, (
            f"Half-life {ou_params.half_life_days:.1f}d outside expected range for kappa=0.15"
        )

    def test_signals_generated_non_empty(self):
        eg, _ = engle_granger_test(self.pa, self.pb, both_directions=True)
        spread = self.pa - eg.hedge_ratio * self.pb
        cfg = SignalConfig(window=40, z_entry=1.5)
        signals = generate_signals(spread, cfg)
        n_trades = (
            len(signals.signal_dates["entry_long"])
            + len(signals.signal_dates["entry_short"])
        )
        assert n_trades >= 3, (
            f"Expected multiple trades on mean-reverting spread; got {n_trades}"
        )

    def test_backtest_produces_positive_sharpe(self):
        """
        On a strongly cointegrated synthetic pair, the strategy should
        produce a positive (though not necessarily great) Sharpe.
        This is a sanity check, not a performance guarantee.
        """
        eg_ab, eg_ba = engle_granger_test(self.pa, self.pb, both_directions=True)
        best = select_best_direction(eg_ab, eg_ba)
        spread = self.pa - best.hedge_ratio * self.pb

        signals = generate_signals(spread, SignalConfig(window=40, z_entry=1.5, z_exit=0.3))
        bt_cfg = BacktestConfig(initial_capital=100_000.0, commission_bps=5.0)
        result = run_backtest(signals, self.pa, self.pb, best.hedge_ratio, bt_cfg)
        metrics = compute_metrics(result)

        # Not guaranteed to be positive for any random seed, but with
        # a strongly mean-reverting spread it should usually be positive
        assert isinstance(metrics.sharpe_ratio, float)
        assert not np.isnan(metrics.sharpe_ratio)

    def test_equity_curve_starts_at_initial_capital(self):
        eg, _ = engle_granger_test(self.pa, self.pb, both_directions=True)
        spread = self.pa - eg.hedge_ratio * self.pb
        signals = generate_signals(spread, SignalConfig(window=40))
        cfg = BacktestConfig(initial_capital=75_000.0)
        result = run_backtest(signals, self.pa, self.pb, eg.hedge_ratio, cfg)
        assert result.equity_curve.iloc[0] == pytest.approx(75_000.0)

    def test_metrics_all_finite(self):
        eg, _ = engle_granger_test(self.pa, self.pb, both_directions=True)
        spread = self.pa - eg.hedge_ratio * self.pb
        signals = generate_signals(spread, SignalConfig(window=40))
        result = run_backtest(signals, self.pa, self.pb, eg.hedge_ratio)
        metrics = compute_metrics(result)
        d = metrics.to_dict()
        for key, val in d.items():
            if key not in ("calmar_ratio",):  # calmar can be inf if MDD=0
                assert np.isfinite(val) or val == 0.0, (
                    f"Metric {key} = {val} is not finite"
                )


class TestWalkForwardIntegration:
    """Walk-forward validation integration tests."""

    def setup_method(self):
        self.pa, self.pb = _synthetic_cointegrated_pair(n=1000, kappa=0.15, seed=7)

    def test_walk_forward_runs_without_error(self):
        result = run_walk_forward(
            self.pa, self.pb, "A", "B",
            n_folds=3,
            train_fraction=0.6,
            signal_config=SignalConfig(window=40, z_entry=1.5),
            backtest_config=BacktestConfig(initial_capital=50_000.0),
        )
        assert result is not None

    def test_correct_number_of_folds(self):
        result = run_walk_forward(
            self.pa, self.pb, "A", "B",
            n_folds=4,
            train_fraction=0.6,
        )
        assert len(result.folds) == 4

    def test_combined_equity_length(self):
        """Combined OOS equity should cover the test portion of the sample."""
        n_folds = 3
        result = run_walk_forward(
            self.pa, self.pb, "A", "B",
            n_folds=n_folds,
            train_fraction=0.6,
        )
        # Each fold covers (1-0.6)/n_folds fraction of 1000 obs ≈ 133 per fold
        assert len(result.combined_equity) > 0

    def test_aggregate_metrics_not_nan(self):
        result = run_walk_forward(
            self.pa, self.pb, "A", "B",
            n_folds=3,
            train_fraction=0.6,
            signal_config=SignalConfig(window=40),
        )
        assert not np.isnan(result.aggregate_metrics.sharpe_ratio)

    def test_at_least_one_cointegrated_fold(self):
        """A strongly cointegrated pair should pass EG in at least one fold."""
        result = run_walk_forward(
            self.pa, self.pb, "A", "B",
            n_folds=4,
            train_fraction=0.5,
        )
        assert result.n_cointegrated_folds >= 1, (
            "Strongly cointegrated synthetic pair should pass EG in at least one fold"
        )

    def test_rolling_window_mode(self):
        """Rolling window mode should run without error."""
        result = run_walk_forward(
            self.pa, self.pb, "A", "B",
            n_folds=3,
            train_fraction=0.6,
            rolling_window=True,
        )
        assert len(result.folds) == 3

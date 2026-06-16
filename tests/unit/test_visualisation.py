"""
Unit tests for visualisation module.

We test that plot functions return valid matplotlib Figure objects
with the expected structure. We use Agg backend to avoid display issues in CI.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from pairs_trading.backtest.engine import BacktestConfig, BacktestResult
from pairs_trading.metrics.performance import compute_drawdown_series
from pairs_trading.models.ou_process import OUParameters
from pairs_trading.signals.zscore import SignalConfig, SignalSeries, generate_signals
from pairs_trading.validation.walk_forward import WalkForwardFold, WalkForwardResult
from pairs_trading.metrics.performance import PerformanceMetrics
from pairs_trading.visualisation.plots import (
    drawdown_chart,
    equity_curve,
    ou_diagnostics,
    spread_with_signals,
    walk_forward_summary,
)


def _make_spread_and_signals(n: int = 200, seed: int = 0) -> tuple:
    """Create a synthetic spread and signal series for plotting tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    spread_vals = np.zeros(n)
    for t in range(1, n):
        spread_vals[t] = 0.9 * spread_vals[t - 1] + rng.normal(0, 1)
    spread = pd.Series(spread_vals, index=idx)
    signals = generate_signals(spread, SignalConfig(window=30, z_entry=1.5))
    return spread, signals


def _make_backtest_result(n: int = 200) -> BacktestResult:
    """Synthetic BacktestResult for plot testing."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    equity = pd.Series(100_000 * np.cumprod(1 + np.random.default_rng(0).normal(0.0002, 0.005, n)), index=idx)
    returns = equity.pct_change().fillna(0)
    positions = pd.Series(np.random.default_rng(1).choice([-1, 0, 1], n).astype(float), index=idx)
    return BacktestResult(
        equity_curve=equity,
        daily_returns=returns,
        positions=positions,
        trades=pd.DataFrame(),
        total_costs=500.0,
        n_trades=10,
        config=BacktestConfig(),
    )


def _make_ou_params() -> OUParameters:
    return OUParameters(
        kappa=40.0, mu=0.0, sigma=0.3,
        sigma_eq=0.3 / np.sqrt(80), half_life_days=4.3,
        ar1_coef=0.85, r_squared=0.92, n_obs=200,
    )


def _make_wf_result() -> WalkForwardResult:
    """Minimal WalkForwardResult for plot testing."""
    idx = pd.date_range("2021-01-01", periods=200, freq="B")
    returns = pd.Series(np.random.default_rng(5).normal(0.0002, 0.005, 200), index=idx)
    equity = 100_000 * (1 + returns).cumprod()

    def _make_metrics(sharpe: float) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_return=0.05, annualised_return=0.06, annualised_volatility=0.08,
            sharpe_ratio=sharpe, max_drawdown=-0.05, calmar_ratio=1.2,
            hit_rate=0.55, n_trades=8, total_costs=200.0, cost_drag_bps=20.0,
            avg_daily_return=0.0002, skewness=0.1, kurtosis=0.5,
        )

    folds = []
    fold_size = 50
    for i in range(4):
        bt = BacktestResult(
            equity_curve=equity.iloc[i*fold_size:(i+1)*fold_size],
            daily_returns=returns.iloc[i*fold_size:(i+1)*fold_size],
            positions=pd.Series(np.ones(fold_size), index=idx[i*fold_size:(i+1)*fold_size]),
            trades=pd.DataFrame(),
            total_costs=50.0,
            n_trades=2,
            config=BacktestConfig(),
        )
        folds.append(WalkForwardFold(
            fold_id=i,
            train_start=idx[0],
            train_end=idx[i * fold_size],
            test_start=idx[i * fold_size],
            test_end=idx[min((i + 1) * fold_size - 1, len(idx) - 1)],
            is_cointegrated=(i % 2 == 0),
            hedge_ratio=1.3,
            ou_params=_make_ou_params(),
            backtest=bt,
            metrics=_make_metrics(sharpe=0.5 * (i - 1)),
        ))

    return WalkForwardResult(
        folds=folds,
        combined_equity=equity,
        combined_returns=returns,
        aggregate_metrics=_make_metrics(sharpe=0.8),
        n_cointegrated_folds=2,
    )


class TestSpreadWithSignals:
    def test_returns_figure(self):
        _, signals = _make_spread_and_signals()
        fig = spread_with_signals(signals, "SPY", "QQQ")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_three_axes(self):
        _, signals = _make_spread_and_signals()
        fig = spread_with_signals(signals)
        assert len(fig.get_axes()) == 3
        plt.close(fig)

    def test_custom_title(self):
        _, signals = _make_spread_and_signals()
        fig = spread_with_signals(signals, title="My Custom Title")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestEquityCurve:
    def test_returns_figure(self):
        result = _make_backtest_result()
        fig = equity_curve(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_benchmarks(self):
        result = _make_backtest_result()
        idx = result.equity_curve.index
        pa = pd.Series(np.ones(len(idx)) * 100, index=idx)
        pb = pd.Series(np.ones(len(idx)) * 50, index=idx)
        fig = equity_curve(result, pa, pb, "A", "B")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_single_axis(self):
        result = _make_backtest_result()
        fig = equity_curve(result)
        assert len(fig.get_axes()) == 1
        plt.close(fig)


class TestDrawdownChart:
    def test_returns_figure(self):
        result = _make_backtest_result()
        fig = drawdown_chart(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_single_axis(self):
        result = _make_backtest_result()
        fig = drawdown_chart(result)
        assert len(fig.get_axes()) == 1
        plt.close(fig)

    def test_custom_title(self):
        result = _make_backtest_result()
        fig = drawdown_chart(result, title="Custom DD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestOUDiagnostics:
    def test_returns_figure(self):
        rng = np.random.default_rng(0)
        n = 100
        spread = pd.Series(
            np.cumsum(rng.normal(0, 0.3, n)) * 0 + rng.normal(0, 1, n),
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        params = _make_ou_params()
        fig = ou_diagnostics(spread, params, "A", "B")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_two_axes(self):
        rng = np.random.default_rng(1)
        spread = pd.Series(rng.normal(0, 1, 80), index=pd.date_range("2020-01-01", periods=80, freq="B"))
        params = _make_ou_params()
        fig = ou_diagnostics(spread, params)
        assert len(fig.get_axes()) == 2
        plt.close(fig)


class TestWalkForwardSummary:
    def test_returns_figure(self):
        wf = _make_wf_result()
        fig = walk_forward_summary(wf, "A", "B")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_two_axes(self):
        wf = _make_wf_result()
        fig = walk_forward_summary(wf)
        assert len(fig.get_axes()) == 2
        plt.close(fig)

    def test_custom_title(self):
        wf = _make_wf_result()
        fig = walk_forward_summary(wf, title="WF Summary Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

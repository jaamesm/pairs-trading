"""
Unit tests for the backtest engine.

Key invariants
--------------
1. Zero-cost flat position earns exactly zero P&L.
2. Transaction costs reduce equity relative to the zero-cost case.
3. Equity curve starts at initial_capital.
4. A perfectly mean-reverting trade (manual setup) earns positive P&L.
5. Positions are consumed with one-day lag (no same-day execution).
"""

import numpy as np
import pandas as pd
import pytest

from pairs_trading.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from pairs_trading.signals.zscore import SignalConfig, SignalSeries, generate_signals


def _make_signal_series(
    position_vals: list[float],
    spread_vals: list[float] | None = None,
) -> SignalSeries:
    """Construct a minimal SignalSeries for testing."""
    n = len(position_vals)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    if spread_vals is None:
        spread_vals = list(range(n))

    spread = pd.Series(spread_vals, index=idx, dtype=float)
    position = pd.Series(position_vals, index=idx, dtype=float)
    z = pd.Series(np.zeros(n), index=idx)

    return SignalSeries(
        spread=spread,
        zscore=z,
        rolling_mean=pd.Series(np.zeros(n), index=idx),
        rolling_std=pd.Series(np.ones(n), index=idx),
        position=position,
        signal_dates={
            "entry_long": pd.DatetimeIndex([]),
            "entry_short": pd.DatetimeIndex([]),
            "exit": pd.DatetimeIndex([]),
            "stop_loss": pd.DatetimeIndex([]),
        },
        config=SignalConfig(),
    )


def _make_flat_prices(n: int, price_a: float = 100.0, price_b: float = 50.0) -> tuple:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    pa = pd.Series([price_a] * n, index=idx)
    pb = pd.Series([price_b] * n, index=idx)
    return pa, pb


class TestEquityCurveBasics:
    """Basic equity accounting tests."""

    def test_starts_at_initial_capital(self):
        signals = _make_signal_series([0.0] * 50)
        pa, pb = _make_flat_prices(50)
        cfg = BacktestConfig(initial_capital=100_000.0)
        result = run_backtest(signals, pa, pb, hedge_ratio=1.0, config=cfg)
        assert result.equity_curve.iloc[0] == pytest.approx(100_000.0)

    def test_flat_position_earns_zero(self):
        """If always flat, equity should not change (ignoring costs on no trades)."""
        n = 100
        signals = _make_signal_series([0.0] * n)
        pa, pb = _make_flat_prices(n)
        cfg = BacktestConfig(initial_capital=50_000.0, commission_bps=5.0)
        result = run_backtest(signals, pa, pb, hedge_ratio=1.0, config=cfg)
        assert result.equity_curve.iloc[-1] == pytest.approx(50_000.0, rel=1e-6)
        assert result.total_costs == pytest.approx(0.0)

    def test_equity_curve_length_matches_prices(self):
        n = 80
        signals = _make_signal_series([0.0] * n)
        pa, pb = _make_flat_prices(n)
        result = run_backtest(signals, pa, pb, hedge_ratio=1.0)
        assert len(result.equity_curve) == n

    def test_returns_series_length_matches(self):
        n = 60
        signals = _make_signal_series([0.0] * n)
        pa, pb = _make_flat_prices(n)
        result = run_backtest(signals, pa, pb, hedge_ratio=1.0)
        assert len(result.daily_returns) == n


class TestTransactionCosts:
    """Transaction cost accounting tests."""

    def test_costs_reduce_equity(self):
        """With non-zero costs and a trade, final equity should be lower."""
        n = 50
        # Enter long at day 10, exit at day 40
        pos = [0.0] * 10 + [1.0] * 30 + [0.0] * 10
        spread_vals = [0.0] * n  # flat spread so raw PnL is zero

        sig_costly = _make_signal_series(pos, spread_vals)
        sig_free = _make_signal_series(pos, spread_vals)
        pa, pb = _make_flat_prices(n)

        cfg_costly = BacktestConfig(commission_bps=10.0, slippage_bps=5.0)
        cfg_free = BacktestConfig(commission_bps=0.0, slippage_bps=0.0)

        res_costly = run_backtest(sig_costly, pa, pb, hedge_ratio=1.0, config=cfg_costly)
        res_free = run_backtest(sig_free, pa, pb, hedge_ratio=1.0, config=cfg_free)

        assert res_costly.equity_curve.iloc[-1] < res_free.equity_curve.iloc[-1], (
            "Costly backtest should have lower final equity"
        )

    def test_total_costs_positive_when_trading(self):
        """Total costs should be > 0 if there are position changes."""
        n = 50
        pos = [0.0] * 10 + [1.0] * 20 + [0.0] * 20
        sig = _make_signal_series(pos, [0.0] * n)
        pa, pb = _make_flat_prices(n)
        cfg = BacktestConfig(commission_bps=5.0, slippage_bps=3.0)
        result = run_backtest(sig, pa, pb, hedge_ratio=1.0, config=cfg)
        assert result.total_costs > 0.0

    def test_zero_costs_when_flat(self):
        sig = _make_signal_series([0.0] * 50, [0.0] * 50)
        pa, pb = _make_flat_prices(50)
        cfg = BacktestConfig(commission_bps=10.0)
        result = run_backtest(sig, pa, pb, hedge_ratio=1.0, config=cfg)
        assert result.total_costs == 0.0


class TestPnLAccounting:
    """P&L calculation tests."""

    def test_one_day_lag(self):
        """
        Position signal on day t should only earn returns from t+1 onward.
        If we enter long (position=1) on day 5, we should earn from day 6.
        Verify: changing the spread on day 5 (entry day) doesn't affect P&L.
        """
        n = 30
        # Enter long at day 10
        pos = [0.0] * 10 + [1.0] * 20
        spread = [0.0] * 10 + [5.0] + [5.0] * 19  # jump at entry day
        sig = _make_signal_series(pos, spread)
        pa, pb = _make_flat_prices(n)
        cfg = BacktestConfig(commission_bps=0.0, slippage_bps=0.0)
        result = run_backtest(sig, pa, pb, hedge_ratio=1.0, config=cfg)
        # The return on day 10 (entry day) should be zero since pos[9]=0 (was flat)
        assert result.daily_returns.iloc[10] == pytest.approx(0.0, abs=1e-10)

    def test_long_spread_profits_from_positive_move(self):
        """
        Long spread position should profit when spread increases.
        Setup: enter long spread at day 2, spread moves from 0 to +5 by day 5.
        """
        n = 20
        # Position: flat [0:2], long [2:10], flat [10:]
        pos = [0.0, 0.0] + [1.0] * 8 + [0.0] * 10
        # Spread: rises from 0 to 10 during the long position
        spread = [0.0] * 2 + list(np.linspace(0, 10, 8)) + [10.0] * 10
        sig = _make_signal_series(pos, spread)
        pa = pd.Series([100.0] * n, index=pd.date_range("2020-01-01", periods=n, freq="B"))
        pb = pd.Series([50.0] * n, index=pa.index)
        cfg = BacktestConfig(commission_bps=0.0, slippage_bps=0.0, initial_capital=10_000.0)
        result = run_backtest(sig, pa, pb, hedge_ratio=1.0, config=cfg)
        # Should have made money
        assert result.equity_curve.iloc[-1] > cfg.initial_capital

    def test_short_spread_profits_from_negative_move(self):
        """Short spread position should profit when spread decreases."""
        n = 20
        pos = [0.0, 0.0] + [-1.0] * 8 + [0.0] * 10
        spread = [5.0] * 2 + list(np.linspace(5, 0, 8)) + [0.0] * 10
        sig = _make_signal_series(pos, spread)
        pa = pd.Series([100.0] * n, index=pd.date_range("2020-01-01", periods=n, freq="B"))
        pb = pd.Series([50.0] * n, index=pa.index)
        cfg = BacktestConfig(commission_bps=0.0, slippage_bps=0.0, initial_capital=10_000.0)
        result = run_backtest(sig, pa, pb, hedge_ratio=1.0, config=cfg)
        assert result.equity_curve.iloc[-1] > cfg.initial_capital


class TestBacktestResultStructure:
    """Test the structure and types of the BacktestResult."""

    def setup_method(self):
        n = 100
        spread = _make_spread_ar1(n)
        sig = generate_signals(spread, SignalConfig(window=20))
        pa = pd.Series([100.0] * n, index=spread.index)
        pb = pd.Series([50.0] * n, index=spread.index)
        self.result = run_backtest(sig, pa, pb, hedge_ratio=1.0)

    def test_positions_index_matches_equity(self):
        assert self.result.positions.index.equals(self.result.equity_curve.index)

    def test_daily_returns_index_matches_equity(self):
        assert self.result.daily_returns.index.equals(self.result.equity_curve.index)

    def test_equity_is_positive(self):
        """Equity should never go to zero or negative with reasonable sizing."""
        assert (self.result.equity_curve > 0).all()

    def test_n_trades_non_negative(self):
        assert self.result.n_trades >= 0


def _make_spread_ar1(n: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.85 * s[t - 1] + rng.normal(0, 1)
    return pd.Series(s, index=pd.date_range("2020-01-01", periods=n, freq="B"))

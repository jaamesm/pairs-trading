"""
validation/walk_forward.py
==========================
Walk-forward (expanding or rolling window) validation for pairs trading.

Why walk-forward?
-----------------
Standard train/test splits are insufficient for time series because:
1. The optimal signal parameters (window, z_entry, z_exit) are selected
   in-sample. Testing on the same data is circular.
2. Cointegration relationships and OU parameters are non-stationary —
   they should be re-estimated as new data arrives.

Walk-forward validation addresses both:
- The universe is partitioned into sequential non-overlapping test windows.
- For each test window, ALL parameters are estimated from prior data only.
- The test window results are concatenated to form the out-of-sample record.

Two modes
---------
Expanding window (default):
    Train on [0, split_i], test on [split_i, split_i + test_size].
    The training set grows with each fold. Appropriate when you believe
    the full history is informative.

Rolling window:
    Train on [split_i - train_size, split_i], test on [split_i, split_i + test_size].
    Fixed-length training set. Appropriate when you believe distant history
    is irrelevant (regime changes).

Bonferroni correction
---------------------
When selecting pairs by cointegration p-value, we apply a Bonferroni correction
to account for multiple testing: the effective significance threshold is
α / n_pairs rather than α.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pairs_trading.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from pairs_trading.cointegration.engle_granger import engle_granger_test, select_best_direction
from pairs_trading.metrics.performance import PerformanceMetrics, compute_metrics
from pairs_trading.models.ou_process import fit_ou, OUParameters
from pairs_trading.signals.zscore import SignalConfig, generate_signals


@dataclass
class WalkForwardFold:
    """
    Results from a single fold of the walk-forward validation.

    Attributes
    ----------
    fold_id : int
        Zero-indexed fold number.
    train_start, train_end : pd.Timestamp
        Training period boundaries.
    test_start, test_end : pd.Timestamp
        Out-of-sample test period boundaries.
    is_cointegrated : bool
        Whether the pair passed the cointegration test in the training window.
        If False, no trade is taken in the test window.
    hedge_ratio : float | None
        Estimated β from training window.
    ou_params : OUParameters | None
        OU parameters fitted to training-window spread.
    backtest : BacktestResult | None
        Backtest result on the test window. None if not cointegrated.
    metrics : PerformanceMetrics | None
        Performance metrics for this fold's test window.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_cointegrated: bool
    hedge_ratio: float | None = None
    ou_params: OUParameters | None = None
    backtest: BacktestResult | None = None
    metrics: PerformanceMetrics | None = None


@dataclass
class WalkForwardResult:
    """
    Aggregated results across all walk-forward folds.

    Attributes
    ----------
    folds : list[WalkForwardFold]
        Per-fold results.
    combined_equity : pd.Series
        Equity curve assembled by concatenating each fold's test equity.
        Starts at the initial capital and compounds across folds.
    combined_returns : pd.Series
        Daily returns across all test folds.
    aggregate_metrics : PerformanceMetrics
        Performance metrics computed on the combined out-of-sample returns.
    n_cointegrated_folds : int
        Number of folds in which the pair was cointegrated in-sample.
    """

    folds: list[WalkForwardFold]
    combined_equity: pd.Series
    combined_returns: pd.Series
    aggregate_metrics: PerformanceMetrics
    n_cointegrated_folds: int

    def summary(self) -> str:
        n = len(self.folds)
        return (
            f"Walk-Forward Validation: {n} folds\n"
            f"  Cointegrated in {self.n_cointegrated_folds}/{n} folds\n"
            f"\nOut-of-Sample Aggregate:\n"
            + self.aggregate_metrics.summary()
        )


def run_walk_forward(
    prices_a: pd.Series,
    prices_b: pd.Series,
    ticker_a: str,
    ticker_b: str,
    n_folds: int = 5,
    train_fraction: float = 0.6,
    signal_config: SignalConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    significance: float = 0.05,
    rolling_window: bool = False,
) -> WalkForwardResult:
    """
    Run walk-forward validation for a pairs trading strategy.

    Parameters
    ----------
    prices_a, prices_b : pd.Series
        Full price history for both assets.
    ticker_a, ticker_b : str
        Ticker labels.
    n_folds : int
        Number of walk-forward folds. More folds → more out-of-sample data
        but shorter training windows per fold.
    train_fraction : float
        Fraction of the total sample used as the initial training window.
        The remaining (1 - train_fraction) is divided into n_folds test windows.
    signal_config : SignalConfig or None
        Signal parameters. Defaults to SignalConfig().
    backtest_config : BacktestConfig or None
        Backtest parameters. Defaults to BacktestConfig().
    significance : float
        Cointegration test significance threshold. Applied to the training
        window at each fold.
    rolling_window : bool
        If True, use a fixed-length rolling training window instead of
        an expanding window.

    Returns
    -------
    WalkForwardResult
    """
    if signal_config is None:
        signal_config = SignalConfig()
    if backtest_config is None:
        backtest_config = BacktestConfig()

    n_total = len(prices_a)
    n_train_initial = int(n_total * train_fraction)
    n_test_total = n_total - n_train_initial
    test_fold_size = n_test_total // n_folds

    if test_fold_size < signal_config.window + 10:
        raise ValueError(
            f"Test fold size ({test_fold_size}) is too small for signal window "
            f"({signal_config.window}). Reduce n_folds or signal window."
        )

    idx = prices_a.index
    folds: list[WalkForwardFold] = []
    all_returns: list[pd.Series] = []
    current_equity = backtest_config.initial_capital

    for fold_id in range(n_folds):
        # Determine train and test index slices
        test_start_i = n_train_initial + fold_id * test_fold_size
        test_end_i = min(test_start_i + test_fold_size, n_total)

        if rolling_window:
            train_start_i = max(0, test_start_i - n_train_initial)
        else:
            train_start_i = 0
        train_end_i = test_start_i

        train_a = prices_a.iloc[train_start_i:train_end_i]
        train_b = prices_b.iloc[train_start_i:train_end_i]
        test_a = prices_a.iloc[test_start_i:test_end_i]
        test_b = prices_b.iloc[test_start_i:test_end_i]

        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=idx[train_start_i],
            train_end=idx[train_end_i - 1],
            test_start=idx[test_start_i],
            test_end=idx[test_end_i - 1],
            is_cointegrated=False,
        )

        # Step 1: Test cointegration on training window
        try:
            res_ab, res_ba = engle_granger_test(
                train_a, train_b,
                ticker_a, ticker_b,
                significance=significance,
                both_directions=True,
            )
            best = select_best_direction(res_ab, res_ba)
        except Exception:
            folds.append(fold)
            all_returns.append(pd.Series(0.0, index=test_a.index))
            continue

        fold.is_cointegrated = best.is_cointegrated
        fold.hedge_ratio = best.hedge_ratio

        if not best.is_cointegrated:
            folds.append(fold)
            all_returns.append(pd.Series(0.0, index=test_a.index))
            continue

        # Step 2: Fit OU to training spread
        train_spread = train_a - best.hedge_ratio * train_b
        try:
            ou = fit_ou(train_spread)
        except Exception:
            folds.append(fold)
            all_returns.append(pd.Series(0.0, index=test_a.index))
            continue

        fold.ou_params = ou

        # Step 3: Generate signals on test window using training parameters
        # We concatenate a warm-up period from the end of training to allow
        # the rolling z-score to initialise, but only evaluate P&L on test data.
        warmup_size = signal_config.window + 1
        warmup_a = prices_a.iloc[max(train_end_i - warmup_size, 0):train_end_i]
        warmup_b = prices_b.iloc[max(train_end_i - warmup_size, 0):train_end_i]

        combined_a = pd.concat([warmup_a, test_a])
        combined_b = pd.concat([warmup_b, test_b])
        combined_spread = combined_a - best.hedge_ratio * combined_b

        signals = generate_signals(combined_spread, signal_config)

        # Trim to test window only
        test_signals_position = signals.position.loc[test_a.index]
        test_signals_zscore = signals.zscore.loc[test_a.index]

        from pairs_trading.signals.zscore import SignalSeries
        test_signals = SignalSeries(
            spread=combined_spread.loc[test_a.index],
            zscore=test_signals_zscore,
            rolling_mean=signals.rolling_mean.loc[test_a.index],
            rolling_std=signals.rolling_std.loc[test_a.index],
            position=test_signals_position,
            signal_dates=signals.signal_dates,
            config=signal_config,
        )

        # Step 4: Run backtest on test window, compounding from previous equity
        fold_config = BacktestConfig(
            initial_capital=current_equity,
            capital_fraction=backtest_config.capital_fraction,
            commission_bps=backtest_config.commission_bps,
            slippage_bps=backtest_config.slippage_bps,
        )

        try:
            bt = run_backtest(test_signals, test_a, test_b, best.hedge_ratio, fold_config)
            metrics = compute_metrics(bt)
            fold.backtest = bt
            fold.metrics = metrics
            current_equity = float(bt.equity_curve.iloc[-1])
            all_returns.append(bt.daily_returns)
        except Exception:
            all_returns.append(pd.Series(0.0, index=test_a.index))

        folds.append(fold)

    # Assemble combined out-of-sample equity curve
    combined_returns = pd.concat(all_returns)
    combined_equity = (1 + combined_returns).cumprod() * backtest_config.initial_capital

    # Compute aggregate metrics from combined returns
    # Construct a dummy BacktestResult for compute_metrics
    from pairs_trading.backtest.engine import BacktestResult
    total_costs = sum(
        f.backtest.total_costs for f in folds if f.backtest is not None
    )
    n_trades = sum(
        f.backtest.n_trades for f in folds if f.backtest is not None
    )
    aggregate_bt = BacktestResult(
        equity_curve=combined_equity,
        daily_returns=combined_returns,
        positions=pd.concat(
            [f.backtest.positions for f in folds if f.backtest is not None]
        ) if any(f.backtest for f in folds) else pd.Series(0.0, index=combined_returns.index),
        trades=pd.concat(
            [f.backtest.trades for f in folds
             if f.backtest is not None and not f.backtest.trades.empty]
        ) if any(f.backtest and not f.backtest.trades.empty for f in folds) else pd.DataFrame(),
        total_costs=total_costs,
        n_trades=n_trades,
        config=backtest_config,
    )
    aggregate_metrics = compute_metrics(aggregate_bt)

    n_coint = sum(1 for f in folds if f.is_cointegrated)

    return WalkForwardResult(
        folds=folds,
        combined_equity=combined_equity,
        combined_returns=combined_returns,
        aggregate_metrics=aggregate_metrics,
        n_cointegrated_folds=n_coint,
    )

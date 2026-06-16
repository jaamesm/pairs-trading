"""
backtest/engine.py
==================
Event-driven backtest engine for pairs trading strategies.

Position sizing
---------------
We size positions so that the dollar value of each leg is equal (dollar-neutral):

    Notional per leg = portfolio_value * capital_fraction / 2

The number of shares in each leg:
    shares_a = notional / P_a,t
    shares_b = (notional * hedge_ratio) / P_b,t     [to maintain β-neutrality]

Transaction costs and slippage
-------------------------------
Every time a position changes (entry or exit), we deduct:

    cost = notional * (commission_bps / 10_000)   [per leg, both legs]
    slippage = notional * (slippage_bps / 10_000)  [per leg, both legs]

This is applied to both legs at every trade, giving a round-trip cost of
approximately 2 * (commission_bps + slippage_bps) basis points of notional.

For liquid ETFs (SPY, QQQ, GLD, GDX) with daily execution, 5 bps commission
and 2-5 bps slippage per leg is a realistic conservative estimate.

P&L accounting
--------------
The strategy P&L on day t is:

    PnL_t = position_{t-1} * (S_t - S_{t-1}) * shares

where position_{t-1} is the position held going into day t (determined by
yesterday's signal), and the spread change is the dollar return on the spread.

Crucially, position_{t-1} not position_t: we use yesterday's position to
compute today's return. This prevents acting on today's signal and earning
today's return — a subtle lookahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pairs_trading.signals.zscore import SignalSeries


@dataclass
class BacktestConfig:
    """
    Configuration for the backtest engine.

    Attributes
    ----------
    initial_capital : float
        Starting portfolio value in dollars.
    capital_fraction : float
        Fraction of portfolio deployed per trade (both legs combined).
        E.g. 0.20 means 10% per leg for a dollar-neutral trade.
    commission_bps : float
        Commission per leg per trade in basis points (1 bps = 0.01%).
    slippage_bps : float
        Slippage per leg per trade in basis points.
    """

    initial_capital: float = 100_000.0
    capital_fraction: float = 0.20
    commission_bps: float = 5.0
    slippage_bps: float = 3.0


@dataclass
class BacktestResult:
    """
    Output of the backtest engine.

    Attributes
    ----------
    equity_curve : pd.Series
        Portfolio value over time, starting at initial_capital.
    daily_returns : pd.Series
        Daily portfolio returns (arithmetic, not log).
    positions : pd.Series
        Position series (+1, -1, 0) that was backtested.
    trades : pd.DataFrame
        Record of every trade: date, direction, spread value, cost.
    total_costs : float
        Total transaction costs incurred over the backtest.
    n_trades : int
        Total number of completed round-trip trades (entry + exit pairs).
    config : BacktestConfig
    """

    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    total_costs: float
    n_trades: int
    config: BacktestConfig


def run_backtest(
    signals: SignalSeries,
    prices_a: pd.Series,
    prices_b: pd.Series,
    hedge_ratio: float,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run a backtest from a pre-generated SignalSeries.

    Parameters
    ----------
    signals : SignalSeries
        Output of generate_signals(), containing the position series.
    prices_a, prices_b : pd.Series
        Raw price series for the two assets (aligned with signals).
    hedge_ratio : float
        Estimated β from cointegration regression. Used to determine
        the relative sizing of the two legs.
    config : BacktestConfig or None
        Backtest parameters. Defaults to BacktestConfig().

    Returns
    -------
    BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    pos = signals.position
    spread = signals.spread
    idx = pos.index

    # Align all series on the same index
    pa = prices_a.reindex(idx)
    pb = prices_b.reindex(idx)

    n = len(idx)
    equity = np.empty(n)
    equity[0] = config.initial_capital
    daily_returns = np.zeros(n)
    total_costs = 0.0

    trade_records: list[dict] = []
    open_trade_date: pd.Timestamp | None = None
    open_trade_pos: int = 0

    # Pre-compute position changes (trades)
    pos_values = pos.values
    pos_prev = np.concatenate([[0.0], pos_values[:-1]])
    trades_mask = pos_values != pos_prev  # True on days where position changes

    for i in range(1, n):
        current_equity = equity[i - 1]
        notional_per_leg = current_equity * config.capital_fraction / 2.0

        # Spread P&L: position entered YESTERDAY earns TODAY's spread change.
        # pos_prev[i] is the position from end of day i-1.
        spread_change = spread.iloc[i] - spread.iloc[i - 1]
        raw_pnl = pos_prev[i] * spread_change * (notional_per_leg / pa.iloc[i])

        # Transaction costs: charged when position changes
        cost = 0.0
        if trades_mask[i]:
            cost_bps = (config.commission_bps + config.slippage_bps) / 10_000.0
            cost = 2.0 * notional_per_leg * cost_bps  # both legs

            total_costs += cost
            trade_direction = int(pos_values[i])
            trade_records.append(
                {
                    "date": idx[i],
                    "position": trade_direction,
                    "spread": spread.iloc[i],
                    "price_a": pa.iloc[i],
                    "price_b": pb.iloc[i],
                    "notional_per_leg": notional_per_leg,
                    "cost": cost,
                }
            )

            # Track round trips
            if open_trade_pos != 0 and trade_direction == 0:
                open_trade_pos = 0
            elif trade_direction != 0:
                open_trade_pos = trade_direction

        net_pnl = raw_pnl - cost
        equity[i] = current_equity + net_pnl
        daily_returns[i] = net_pnl / current_equity if current_equity != 0 else 0.0

    equity_series = pd.Series(equity, index=idx, name="equity")
    returns_series = pd.Series(daily_returns, index=idx, name="daily_return")
    trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

    # Count round-trip trades (entry → exit pairs)
    if not trades_df.empty:
        n_entries = (trades_df["position"] != 0).sum()
        n_round_trips = n_entries // 2  # rough: each entry has a corresponding exit
    else:
        n_round_trips = 0

    return BacktestResult(
        equity_curve=equity_series,
        daily_returns=returns_series,
        positions=pos,
        trades=trades_df,
        total_costs=total_costs,
        n_trades=int(n_round_trips),
        config=config,
    )

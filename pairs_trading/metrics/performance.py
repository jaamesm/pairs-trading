"""
metrics/performance.py
======================
Performance metric computation for pairs trading backtests.

Metrics implemented
-------------------
Sharpe Ratio
    (E[R] - R_f) / σ(R) * sqrt(252)
    Annualised using 252 trading days. Risk-free rate defaults to 0
    (appropriate for a market-neutral strategy where the dollar-neutral
    position earns roughly zero net financing).

Maximum Drawdown (MDD)
    max_{t∈[0,T]} (peak_t - trough_t) / peak_t
    where peak_t = max_{s≤t} equity_s.

Calmar Ratio
    Annualised return / |Maximum Drawdown|
    A risk-adjusted return metric that penalises deep drawdowns.
    Preferred over Sharpe by some practitioners for strategies with
    fat tails or skewed return distributions.

Hit Rate
    Fraction of completed trades that are profitable.

Profit Factor
    Sum of winning trade P&L / |Sum of losing trade P&L|
    > 1.0 indicates the strategy makes more on winners than losers.

Average Trade Duration
    Mean number of days between position entry and exit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from pairs_trading.backtest.engine import BacktestResult

TRADING_DAYS_PER_YEAR = 252.0


@dataclass
class PerformanceMetrics:
    """
    Full suite of performance metrics for a pairs trading backtest.

    Attributes
    ----------
    total_return : float
        Arithmetic total return over the full backtest period.
    annualised_return : float
        Geometric annualised return.
    annualised_volatility : float
        Annualised standard deviation of daily returns.
    sharpe_ratio : float
        Annualised Sharpe ratio (zero risk-free rate).
    max_drawdown : float
        Maximum peak-to-trough drawdown (negative, e.g. -0.12 = -12%).
    calmar_ratio : float
        Annualised return / |max drawdown|.
    hit_rate : float
        Fraction of trading days in a position where daily P&L > 0.
    n_trades : int
        Number of completed round-trip trades.
    total_costs : float
        Total transaction costs in dollars.
    cost_drag_bps : float
        Transaction cost drag expressed in annual basis points.
    avg_daily_return : float
        Mean daily return.
    skewness : float
        Skewness of daily returns.
    kurtosis : float
        Excess kurtosis of daily returns (0 = Gaussian).
    """

    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float
    n_trades: int
    total_costs: float
    cost_drag_bps: float
    avg_daily_return: float
    skewness: float
    kurtosis: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Performance Summary\n"
            f"{'─' * 40}\n"
            f"  Total return        : {self.total_return:+.2%}\n"
            f"  Annualised return   : {self.annualised_return:+.2%}\n"
            f"  Annualised vol      : {self.annualised_volatility:.2%}\n"
            f"  Sharpe ratio        : {self.sharpe_ratio:.3f}\n"
            f"  Max drawdown        : {self.max_drawdown:.2%}\n"
            f"  Calmar ratio        : {self.calmar_ratio:.3f}\n"
            f"  Hit rate            : {self.hit_rate:.2%}\n"
            f"  Trades (round trips): {self.n_trades}\n"
            f"  Total costs         : ${self.total_costs:,.2f}\n"
            f"  Cost drag           : {self.cost_drag_bps:.1f} bps/year\n"
            f"  Skewness            : {self.skewness:.3f}\n"
            f"  Excess kurtosis     : {self.kurtosis:.3f}\n"
        )


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown.

    Returns a negative float (e.g. -0.15 for a 15% drawdown).
    """
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    return float(drawdown.min())


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Full time series of drawdown from the rolling peak.
    Useful for visualising the drawdown chart.
    """
    cummax = equity.cummax()
    return (equity - cummax) / cummax


def compute_sharpe(
    returns: pd.Series,
    risk_free_daily: float = 0.0,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily arithmetic returns.
    risk_free_daily : float
        Daily risk-free rate (typically 0 for market-neutral strategies).
    """
    excess = returns - risk_free_daily
    if excess.std() < 1e-10:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_metrics(result: BacktestResult) -> PerformanceMetrics:
    """
    Compute the full performance metric suite from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        Output of run_backtest().

    Returns
    -------
    PerformanceMetrics
    """
    returns = result.daily_returns
    equity = result.equity_curve
    initial = equity.iloc[0]
    final = equity.iloc[-1]

    total_return = (final - initial) / initial

    # Annualised return via geometric compounding
    n_days = len(returns)
    n_years = n_days / TRADING_DAYS_PER_YEAR
    annualised_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    annualised_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = compute_sharpe(returns)
    mdd = compute_max_drawdown(equity)
    calmar = annualised_return / abs(mdd) if mdd != 0 else float("inf")

    # Hit rate: fraction of days in an active position where return > 0
    active_mask = result.positions.shift(1).abs() > 0  # in position yesterday
    active_returns = returns[active_mask]
    hit_rate = float((active_returns > 0).mean()) if len(active_returns) > 0 else 0.0

    # Cost drag in bps/year
    cost_drag_bps = (result.total_costs / (initial * n_years)) * 10_000 if n_years > 0 else 0.0

    # Higher moments (exclude zero-return days from skew/kurt to avoid distortion)
    nonzero_ret = returns[returns != 0]
    skewness = float(nonzero_ret.skew()) if len(nonzero_ret) > 3 else 0.0
    kurtosis = float(nonzero_ret.kurtosis()) if len(nonzero_ret) > 3 else 0.0

    return PerformanceMetrics(
        total_return=float(total_return),
        annualised_return=float(annualised_return),
        annualised_volatility=float(annualised_vol),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(mdd),
        calmar_ratio=float(calmar),
        hit_rate=float(hit_rate),
        n_trades=result.n_trades,
        total_costs=float(result.total_costs),
        cost_drag_bps=float(cost_drag_bps),
        avg_daily_return=float(returns.mean()),
        skewness=float(skewness),
        kurtosis=float(kurtosis),
    )

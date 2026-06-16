"""
visualisation/plots.py
======================
Publication-quality charts for pairs trading analysis.

Four core plots
---------------
1. spread_with_signals  : Raw spread + z-score bands + entry/exit markers
2. equity_curve         : Portfolio value over time with buy-and-hold comparison
3. drawdown_chart       : Rolling drawdown from equity peak
4. walk_forward_summary : Per-fold Sharpe ratios + combined OOS equity curve

All functions return matplotlib Figure objects so callers can save, display,
or embed them without side effects. Call plt.show() in scripts/notebooks.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from pairs_trading.backtest.engine import BacktestResult
from pairs_trading.metrics.performance import compute_drawdown_series
from pairs_trading.signals.zscore import SignalSeries
from pairs_trading.validation.walk_forward import WalkForwardResult

# ── Colour palette (colourblind-friendly) ────────────────────────────────────
COLOURS = {
    "spread": "#2c7bb6",
    "mean": "#d7191c",
    "band_upper": "#fdae61",
    "band_lower": "#abd9e9",
    "entry_long": "#1a9641",
    "entry_short": "#d7191c",
    "stop": "#762a83",
    "equity": "#2c7bb6",
    "benchmark": "#999999",
    "drawdown": "#d7191c",
    "fold_bar": "#4393c3",
}

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def spread_with_signals(
    signals: SignalSeries,
    ticker_a: str = "A",
    ticker_b: str = "B",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Three-panel plot: raw spread, z-score with signal bands, position state.

    Panel 1 — Spread S_t with rolling mean ± entry/exit bands (in spread units).
    Panel 2 — Z-score z_t with horizontal lines at ±z_entry, ±z_exit, ±z_stop.
               Entry/exit/stop markers overlaid.
    Panel 3 — Position state over time (+1, 0, -1) as a filled step plot.
    """
    _apply_style()
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    spread = signals.spread
    z = signals.zscore
    pos = signals.position
    mu = signals.rolling_mean
    sigma = signals.rolling_std
    cfg = signals.config

    pair_label = f"{ticker_a} / {ticker_b}"
    fig.suptitle(
        title or f"Pairs Trading Signals — {pair_label}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # ── Panel 1: Spread ───────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(spread.index, spread, color=COLOURS["spread"], lw=0.8, label="Spread $S_t$")
    ax1.plot(mu.index, mu, color=COLOURS["mean"], lw=1.0, ls="--", label="Rolling mean")
    ax1.fill_between(
        mu.index,
        mu + cfg.z_entry * sigma,
        mu - cfg.z_entry * sigma,
        alpha=0.12,
        color=COLOURS["band_upper"],
        label=f"±{cfg.z_entry}σ entry band",
    )
    ax1.set_ylabel("Spread ($)")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.7)

    # ── Panel 2: Z-score ──────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(z.index, z, color=COLOURS["spread"], lw=0.8, label="Z-score $z_t$")

    for level, ls, color, label in [
        (cfg.z_entry, "--", COLOURS["entry_short"], f"±{cfg.z_entry} entry"),
        (-cfg.z_entry, "--", COLOURS["entry_short"], None),
        (cfg.z_exit, ":", COLOURS["mean"], f"±{cfg.z_exit} exit"),
        (-cfg.z_exit, ":", COLOURS["mean"], None),
        (cfg.z_stop, "-.", COLOURS["stop"], f"±{cfg.z_stop} stop"),
        (-cfg.z_stop, "-.", COLOURS["stop"], None),
    ]:
        ax2.axhline(level, ls=ls, color=color, lw=1.0, alpha=0.8, label=label)

    ax2.axhline(0, color="black", lw=0.5, alpha=0.4)

    # Signal markers
    for dates, color, marker, label in [
        (signals.signal_dates["entry_long"], COLOURS["entry_long"], "^", "Enter long"),
        (signals.signal_dates["entry_short"], COLOURS["entry_short"], "v", "Enter short"),
        (signals.signal_dates["exit"], "black", "o", "Exit"),
        (signals.signal_dates["stop_loss"], COLOURS["stop"], "x", "Stop loss"),
    ]:
        if len(dates) > 0:
            z_vals = z.reindex(dates).dropna()
            ax2.scatter(
                z_vals.index, z_vals.values,
                color=color, marker=marker, s=40, zorder=5, label=label,
            )

    ax2.set_ylabel("Z-score")
    ax2.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.7)

    # ── Panel 3: Position ─────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(
        pos.index, pos, step="post",
        color=COLOURS["equity"], alpha=0.3, label="Position",
    )
    ax3.step(pos.index, pos, where="post", color=COLOURS["equity"], lw=0.8)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["Short spread", "Flat", "Long spread"], fontsize=8)
    ax3.set_ylabel("Position")
    ax3.set_xlabel("Date")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    return fig


def equity_curve(
    result: BacktestResult,
    prices_a: Optional[pd.Series] = None,
    prices_b: Optional[pd.Series] = None,
    ticker_a: str = "A",
    ticker_b: str = "B",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Portfolio equity curve, optionally with buy-and-hold benchmarks for each leg.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    eq = result.equity_curve
    initial = eq.iloc[0]

    ax.plot(eq.index, eq, color=COLOURS["equity"], lw=1.5, label="Strategy")

    if prices_a is not None:
        pa = prices_a.reindex(eq.index)
        bh_a = initial * (pa / pa.iloc[0])
        ax.plot(bh_a.index, bh_a, color=COLOURS["benchmark"], lw=1.0,
                ls="--", alpha=0.7, label=f"Buy & Hold {ticker_a}")

    if prices_b is not None:
        pb = prices_b.reindex(eq.index)
        bh_b = initial * (pb / pb.iloc[0])
        ax.plot(bh_b.index, bh_b, color=COLOURS["drawdown"], lw=1.0,
                ls=":", alpha=0.7, label=f"Buy & Hold {ticker_b}")

    ax.axhline(initial, color="black", lw=0.5, alpha=0.3, ls="--")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.set_title(title or f"Equity Curve — {ticker_a}/{ticker_b}", fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    return fig


def drawdown_chart(
    result: BacktestResult,
    ticker_a: str = "A",
    ticker_b: str = "B",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """
    Rolling drawdown from equity peak. Filled area below zero.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    dd = compute_drawdown_series(result.equity_curve) * 100  # as percentage
    mdd = dd.min()

    ax.fill_between(dd.index, dd, 0, color=COLOURS["drawdown"], alpha=0.4)
    ax.plot(dd.index, dd, color=COLOURS["drawdown"], lw=0.8)
    ax.axhline(mdd, color=COLOURS["stop"], lw=1.0, ls="--",
               label=f"Max Drawdown: {mdd:.1f}%")
    ax.axhline(0, color="black", lw=0.5)

    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_title(title or f"Drawdown — {ticker_a}/{ticker_b}", fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    return fig


def walk_forward_summary(
    wf_result: WalkForwardResult,
    ticker_a: str = "A",
    ticker_b: str = "B",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Two-panel walk-forward summary.

    Panel 1 — Per-fold Sharpe ratios (bar chart), colour-coded by whether
               the pair was cointegrated in that fold's training window.
    Panel 2 — Combined out-of-sample equity curve across all folds, with
               vertical lines separating folds.
    """
    _apply_style()
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35)

    pair_label = f"{ticker_a}/{ticker_b}"
    fig.suptitle(
        title or f"Walk-Forward Validation — {pair_label}",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Per-fold Sharpe ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    fold_ids = []
    sharpes = []
    colours_bar = []
    fold_labels = []

    for fold in wf_result.folds:
        fold_ids.append(fold.fold_id)
        s = fold.metrics.sharpe_ratio if fold.metrics else 0.0
        sharpes.append(s)
        colours_bar.append(COLOURS["fold_bar"] if fold.is_cointegrated else COLOURS["benchmark"])
        fold_labels.append(
            f"Fold {fold.fold_id + 1}\n"
            f"{fold.test_start.strftime('%Y-%m')}"
        )

    bars = ax1.bar(fold_ids, sharpes, color=colours_bar, alpha=0.8, edgecolor="white")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axhline(1.0, color=COLOURS["entry_long"], lw=1.0, ls="--", alpha=0.6, label="Sharpe = 1")

    for bar, s in zip(bars, sharpes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03 * (1 if s >= 0 else -1),
            f"{s:.2f}", ha="center", va="bottom", fontsize=8,
        )

    ax1.set_xticks(fold_ids)
    ax1.set_xticklabels(fold_labels, fontsize=8)
    ax1.set_ylabel("Out-of-Sample Sharpe")
    ax1.set_title("Per-Fold Sharpe Ratio (blue = cointegrated in-sample, grey = not)")
    ax1.legend(fontsize=8, framealpha=0.7)

    # ── Panel 2: Combined OOS equity curve ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    eq = wf_result.combined_equity
    ax2.plot(eq.index, eq, color=COLOURS["equity"], lw=1.4, label="OOS equity")
    ax2.axhline(eq.iloc[0], color="black", lw=0.5, ls="--", alpha=0.4)

    # Vertical lines between folds
    for fold in wf_result.folds[1:]:
        ax2.axvline(
            fold.test_start, color=COLOURS["benchmark"],
            lw=0.8, ls=":", alpha=0.6,
        )

    agg = wf_result.aggregate_metrics
    subtitle = (
        f"OOS Sharpe: {agg.sharpe_ratio:.2f}  |  "
        f"OOS Max DD: {agg.max_drawdown:.1%}  |  "
        f"OOS Return: {agg.total_return:.1%}"
    )
    ax2.set_title(subtitle, fontsize=9)
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    return fig


def ou_diagnostics(
    spread: pd.Series,
    ou_params,
    ticker_a: str = "A",
    ticker_b: str = "B",
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """
    Two-panel OU diagnostics: spread with OU equilibrium band, and ACF of spread.
    """
    from statsmodels.graphics.tsaplots import plot_acf

    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    pair_label = f"{ticker_a}/{ticker_b}"
    fig.suptitle(f"OU Process Diagnostics — {pair_label}", fontsize=12, fontweight="bold")

    # Panel 1: Spread with OU equilibrium
    ax1.plot(spread.index, spread, color=COLOURS["spread"], lw=0.8, label="Spread $S_t$")
    ax1.axhline(ou_params.mu, color=COLOURS["mean"], lw=1.2, ls="--", label=f"μ = {ou_params.mu:.4f}")
    ax1.fill_between(
        spread.index,
        ou_params.mu - 2 * ou_params.sigma_eq,
        ou_params.mu + 2 * ou_params.sigma_eq,
        alpha=0.15, color=COLOURS["band_upper"],
        label=f"μ ± 2σ_eq  (σ_eq={ou_params.sigma_eq:.4f})",
    )
    ax1.set_title(
        f"Half-life: {ou_params.half_life_days:.1f} trading days  |  "
        f"κ = {ou_params.kappa:.4f}",
        fontsize=9,
    )
    ax1.set_ylabel("Spread")
    ax1.set_xlabel("Date")
    ax1.legend(fontsize=8, framealpha=0.7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Panel 2: ACF — mean-reverting series shows fast decay
    plot_acf(spread.dropna(), lags=40, ax=ax2, color=COLOURS["spread"], alpha=0.05)
    ax2.set_title("Autocorrelation Function of Spread", fontsize=9)
    ax2.set_xlabel("Lag (days)")
    ax2.set_ylabel("ACF")

    fig.tight_layout()
    return fig

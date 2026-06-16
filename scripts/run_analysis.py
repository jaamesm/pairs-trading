"""
scripts/run_analysis.py
========================
Run the full pairs trading analysis pipeline on the target pairs.

Usage
-----
    python scripts/run_analysis.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                   [--output-dir ./output] [--no-plots]

Target pairs
------------
  SPY/QQQ   : Large-cap US equities — broad market vs tech-heavy
  GLD/GDX   : Gold commodity ETF vs gold miners equity ETF
  XLE/XOM   : Energy sector ETF vs largest energy constituent
  EWJ/EWH   : Japan vs Hong Kong equity ETFs (cross-regional; regime breakdown demo)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/server environments
import matplotlib.pyplot as plt

# Ensure the package is importable when run from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pairs_trading.backtest.engine import BacktestConfig
from pairs_trading.cointegration.engle_granger import engle_granger_test, select_best_direction
from pairs_trading.cointegration.johansen import johansen_test
from pairs_trading.data.loader import fetch_pair
from pairs_trading.metrics.performance import compute_metrics
from pairs_trading.models.ou_process import fit_ou
from pairs_trading.signals.zscore import SignalConfig, generate_signals
from pairs_trading.backtest.engine import run_backtest
from pairs_trading.validation.walk_forward import run_walk_forward
from pairs_trading.visualisation.plots import (
    drawdown_chart,
    equity_curve,
    ou_diagnostics,
    spread_with_signals,
    walk_forward_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Default configuration ─────────────────────────────────────────────────────
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2024-12-31"

TARGET_PAIRS = [
    ("SPY", "QQQ"),   # Broad market vs tech-heavy — expected strong cointegration
    ("GLD", "GDX"),   # Gold ETF vs gold miners — economic link via gold price
    ("XLE", "XOM"),   # Energy sector vs largest constituent — structural link
    ("EWJ", "EWH"),   # Japan vs Hong Kong — cross-regional, regime change demo
]

SIGNAL_CONFIG = SignalConfig(
    window=60,
    z_entry=2.0,
    z_exit=0.5,
    z_stop=3.5,
)

BACKTEST_CONFIG = BacktestConfig(
    initial_capital=100_000.0,
    capital_fraction=0.20,
    commission_bps=5.0,
    slippage_bps=3.0,
)


def analyse_pair(
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
    output_dir: Path,
    save_plots: bool = True,
) -> dict:
    """
    Run the full analysis for a single pair and return a results summary dict.
    """
    pair_label = f"{ticker_a}/{ticker_b}"
    logger.info("=" * 60)
    logger.info("Analysing pair: %s", pair_label)
    logger.info("=" * 60)

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    logger.info("Fetching data...")
    pair_data = fetch_pair(ticker_a, ticker_b, start, end, min_obs=252)
    logger.info(
        "  %d observations from %s to %s",
        pair_data.n_obs, pair_data.start, pair_data.end,
    )

    pa, pb = pair_data.prices_a, pair_data.prices_b

    # ── 2. Cointegration tests ────────────────────────────────────────────────
    logger.info("Running cointegration tests...")

    eg_ab, eg_ba = engle_granger_test(pa, pb, ticker_a, ticker_b, both_directions=True)
    eg_best = select_best_direction(eg_ab, eg_ba)
    print(eg_best.summary())

    joh = johansen_test(pa, pb, ticker_a, ticker_b)
    print(joh.summary())

    both_agree = eg_best.is_cointegrated and joh.is_cointegrated
    logger.info(
        "  Cointegration: EG=%s  Johansen=%s  Agreement=%s",
        eg_best.is_cointegrated, joh.is_cointegrated, both_agree,
    )

    # ── 3. OU process estimation ──────────────────────────────────────────────
    spread = pa - eg_best.hedge_ratio * pb
    logger.info("Fitting OU process to spread...")
    ou_params = fit_ou(spread)
    print(ou_params.summary())

    if save_plots:
        fig = ou_diagnostics(spread, ou_params, ticker_a, ticker_b)
        fig.savefig(output_dir / f"{ticker_a}_{ticker_b}_ou_diagnostics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 4. Signal generation ──────────────────────────────────────────────────
    logger.info("Generating trading signals...")
    signals = generate_signals(spread, SIGNAL_CONFIG)
    n_long = len(signals.signal_dates["entry_long"])
    n_short = len(signals.signal_dates["entry_short"])
    n_stops = len(signals.signal_dates["stop_loss"])
    logger.info("  Long entries: %d  Short entries: %d  Stop-losses: %d", n_long, n_short, n_stops)

    if save_plots:
        fig = spread_with_signals(signals, ticker_a, ticker_b)
        fig.savefig(output_dir / f"{ticker_a}_{ticker_b}_signals.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 5. In-sample backtest ─────────────────────────────────────────────────
    logger.info("Running in-sample backtest...")
    bt_result = run_backtest(signals, pa, pb, eg_best.hedge_ratio, BACKTEST_CONFIG)
    metrics = compute_metrics(bt_result)
    print(metrics.summary())

    if save_plots:
        fig_eq = equity_curve(bt_result, pa, pb, ticker_a, ticker_b)
        fig_eq.savefig(output_dir / f"{ticker_a}_{ticker_b}_equity.png", dpi=150, bbox_inches="tight")
        plt.close(fig_eq)

        fig_dd = drawdown_chart(bt_result, ticker_a, ticker_b)
        fig_dd.savefig(output_dir / f"{ticker_a}_{ticker_b}_drawdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig_dd)

    # ── 6. Walk-forward validation ────────────────────────────────────────────
    logger.info("Running walk-forward validation (5 folds)...")
    wf_result = run_walk_forward(
        pa, pb, ticker_a, ticker_b,
        n_folds=5,
        train_fraction=0.6,
        signal_config=SIGNAL_CONFIG,
        backtest_config=BACKTEST_CONFIG,
    )
    print(wf_result.summary())

    if save_plots:
        fig_wf = walk_forward_summary(wf_result, ticker_a, ticker_b)
        fig_wf.savefig(output_dir / f"{ticker_a}_{ticker_b}_walk_forward.png", dpi=150, bbox_inches="tight")
        plt.close(fig_wf)

    return {
        "pair": pair_label,
        "eg_pvalue": eg_best.adf_pvalue,
        "eg_cointegrated": eg_best.is_cointegrated,
        "johansen_cointegrated": joh.is_cointegrated,
        "hedge_ratio": eg_best.hedge_ratio,
        "half_life_days": ou_params.half_life_days,
        "is_tradeable": ou_params.is_tradeable(),
        "insample_sharpe": metrics.sharpe_ratio,
        "insample_max_dd": metrics.max_drawdown,
        "insample_total_return": metrics.total_return,
        "oos_sharpe": wf_result.aggregate_metrics.sharpe_ratio,
        "oos_max_dd": wf_result.aggregate_metrics.max_drawdown,
        "oos_total_return": wf_result.aggregate_metrics.total_return,
        "n_cointegrated_folds": wf_result.n_cointegrated_folds,
    }


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted comparison table across all pairs."""
    print("\n" + "=" * 80)
    print("CROSS-PAIR SUMMARY")
    print("=" * 80)
    headers = ["Pair", "EG p-val", "Coint?", "Half-life", "IS Sharpe", "OOS Sharpe", "OOS MDD"]
    col_widths = [12, 10, 8, 11, 10, 11, 10]
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * 80)
    for r in results:
        coint = "✓" if r["eg_cointegrated"] and r["johansen_cointegrated"] else (
            "~" if r["eg_cointegrated"] or r["johansen_cointegrated"] else "✗"
        )
        row_vals = [
            r["pair"],
            f"{r['eg_pvalue']:.4f}",
            coint,
            f"{r['half_life_days']:.1f}d",
            f"{r['insample_sharpe']:.3f}",
            f"{r['oos_sharpe']:.3f}",
            f"{r['oos_max_dd']:.1%}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row_vals, col_widths)))
    print("=" * 80)
    print()
    print("Coint legend: ✓ = both EG and Johansen agree  ~ = one test only  ✗ = neither")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairs trading analysis pipeline")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", default="./output", help="Directory for output plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument(
        "--pair", nargs=2, metavar=("TICKER_A", "TICKER_B"),
        help="Run on a single custom pair instead of the default list",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(args.pair[0], args.pair[1])] if args.pair else TARGET_PAIRS

    all_results = []
    for ticker_a, ticker_b in pairs:
        try:
            result = analyse_pair(
                ticker_a, ticker_b,
                args.start, args.end,
                output_dir,
                save_plots=not args.no_plots,
            )
            all_results.append(result)
        except Exception as exc:
            logger.error("Failed to analyse %s/%s: %s", ticker_a, ticker_b, exc, exc_info=True)

    if all_results:
        print_summary_table(all_results)

    logger.info("Analysis complete. Plots saved to: %s", output_dir.resolve())


if __name__ == "__main__":
    main()

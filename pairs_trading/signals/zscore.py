"""
signals/zscore.py
=================
Rolling z-score signal generator for pairs trading.

Signal logic
------------
The spread S_t is normalised by its rolling mean and standard deviation:

    z_t = (S_t - μ̂_{t-1}) / σ̂_{t-1}

where μ̂ and σ̂ are estimated from the window [t-N, t-1] — crucially
excluding the current observation to prevent lookahead bias.

Entry signals are generated when |z_t| exceeds z_entry:
  - z_t > +z_entry : spread is abnormally high
      → Short the spread (short asset A, long asset B)
  - z_t < -z_entry : spread is abnormally low
      → Long the spread (long asset A, short asset B)

Exit signals are generated when |z_t| falls below z_exit (spread reverted).
Stop-loss signals are generated when |z_t| exceeds z_stop (relationship
may have broken; cut the position to limit drawdown).

Position encoding
-----------------
  +1 : long the spread (long A, short B)
  -1 : short the spread (short A, long B)
   0 : flat (no position)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
    """
    Configuration for z-score signal generation.

    Attributes
    ----------
    window : int
        Rolling window length in trading days for computing μ̂ and σ̂.
        The first `window` observations will have no signal (warm-up period).
    z_entry : float
        |z| threshold to enter a trade. Typical values: 1.5–2.5.
        Higher values → fewer, higher-conviction trades.
    z_exit : float
        |z| threshold to close a trade (spread has reverted). Typical: 0–0.5.
    z_stop : float
        |z| threshold for stop-loss. If the spread continues moving against
        the position beyond this level, exit. Typical: 3.0–4.0.
    min_spread_std : float
        Minimum spread standard deviation to generate a signal. If σ̂ < this
        threshold, the window is degenerate and no signal is generated.
    """

    window: int = 60
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.5
    min_spread_std: float = 1e-8


@dataclass
class SignalSeries:
    """
    Output of the signal generator.

    Attributes
    ----------
    spread : pd.Series
        The raw spread S_t = P_a - β*P_b.
    zscore : pd.Series
        Rolling z-score z_t. NaN during the warm-up window.
    rolling_mean : pd.Series
        Rolling mean μ̂_t used in z-score computation.
    rolling_std : pd.Series
        Rolling std σ̂_t used in z-score computation.
    position : pd.Series
        Desired position: +1 (long spread), -1 (short spread), 0 (flat).
        This is the target position — the backtest engine computes trades
        as the difference between consecutive positions.
    signal_dates : dict[str, pd.DatetimeIndex]
        Dates of each signal type for diagnostic purposes.
    config : SignalConfig
        The configuration used.
    """

    spread: pd.Series
    zscore: pd.Series
    rolling_mean: pd.Series
    rolling_std: pd.Series
    position: pd.Series
    signal_dates: dict[str, pd.DatetimeIndex]
    config: SignalConfig


def compute_rolling_zscore(
    spread: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute the rolling z-score of a spread series.

    Uses shift(1) on the rolling statistics to ensure that z_t is computed
    from [t-window, t-1] — no lookahead bias.

    Parameters
    ----------
    spread : pd.Series
        The spread series S_t.
    window : int
        Rolling window length.

    Returns
    -------
    zscore, rolling_mean, rolling_std : pd.Series
        All with the same index as `spread`. NaN for the first `window` rows.
    """
    # shift(1) ensures each rolling value is computed from data up to t-1
    rolling_mean = spread.rolling(window=window).mean().shift(1)
    rolling_std = spread.rolling(window=window).std(ddof=1).shift(1)

    zscore = (spread - rolling_mean) / rolling_std
    return zscore, rolling_mean, rolling_std


def generate_signals(
    spread: pd.Series,
    config: SignalConfig | None = None,
) -> SignalSeries:
    """
    Generate z-score-based entry, exit, and stop-loss signals.

    The signal generator is a simple state machine with three states:
    FLAT, LONG_SPREAD, SHORT_SPREAD. Transitions:

        FLAT → LONG_SPREAD   : z_t < -z_entry
        FLAT → SHORT_SPREAD  : z_t > +z_entry
        LONG_SPREAD  → FLAT  : z_t > -z_exit  (reversion) OR z_t < -z_stop (stop)
        SHORT_SPREAD → FLAT  : z_t < +z_exit  (reversion) OR z_t > +z_stop (stop)

    Parameters
    ----------
    spread : pd.Series
        The raw spread S_t. Should be stationary (passed cointegration test).
    config : SignalConfig or None
        Signal parameters. Defaults to SignalConfig().

    Returns
    -------
    SignalSeries
    """
    if config is None:
        config = SignalConfig()

    zscore, rolling_mean, rolling_std = compute_rolling_zscore(spread, config.window)

    n = len(spread)
    position = np.zeros(n, dtype=np.float64)

    entry_long_dates: list = []
    entry_short_dates: list = []
    exit_dates: list = []
    stop_dates: list = []

    current_pos = 0  # current state: -1, 0, +1

    z_vals = zscore.values
    idx = spread.index

    for i in range(n):
        z = z_vals[i]
        std = rolling_std.iloc[i] if not pd.isna(rolling_std.iloc[i]) else 0.0

        # Skip warm-up period or degenerate windows
        if pd.isna(z) or std < config.min_spread_std:
            position[i] = 0
            current_pos = 0
            continue

        if current_pos == 0:
            # FLAT: check for entry
            if z < -config.z_entry:
                current_pos = 1
                entry_long_dates.append(idx[i])
            elif z > config.z_entry:
                current_pos = -1
                entry_short_dates.append(idx[i])

        elif current_pos == 1:
            # LONG SPREAD: check for exit or stop
            if z > -config.z_exit:
                current_pos = 0
                exit_dates.append(idx[i])
            elif z < -config.z_stop:
                current_pos = 0
                stop_dates.append(idx[i])

        elif current_pos == -1:
            # SHORT SPREAD: check for exit or stop
            if z < config.z_exit:
                current_pos = 0
                exit_dates.append(idx[i])
            elif z > config.z_stop:
                current_pos = 0
                stop_dates.append(idx[i])

        position[i] = current_pos

    position_series = pd.Series(position, index=spread.index, name="position")

    return SignalSeries(
        spread=spread,
        zscore=zscore,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        position=position_series,
        signal_dates={
            "entry_long": pd.DatetimeIndex(entry_long_dates),
            "entry_short": pd.DatetimeIndex(entry_short_dates),
            "exit": pd.DatetimeIndex(exit_dates),
            "stop_loss": pd.DatetimeIndex(stop_dates),
        },
        config=config,
    )

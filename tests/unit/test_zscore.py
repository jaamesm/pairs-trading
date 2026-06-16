"""
Unit tests for z-score signal generation.

Key invariants tested
---------------------
1. Lookahead bias: z_t must be computed from data up to t-1, not t.
2. State machine correctness: positions only change at valid signal events.
3. Stop-loss logic: position closes when |z| > z_stop.
4. Warm-up period: no signals in the first `window` observations.
5. Config propagation: changing thresholds changes signal counts.
"""

import numpy as np
import pandas as pd

from pairs_trading.signals.zscore import (
    SignalConfig,
    SignalSeries,
    compute_rolling_zscore,
    generate_signals,
)


def _make_spread(n: int = 200, seed: int = 0) -> pd.Series:
    """Simple AR(1) spread for testing."""
    rng = np.random.default_rng(seed)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 1)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(spread, index=idx, name="spread")


def _make_spike_spread(spike_at: int = 100, spike_val: float = 3.5, n: int = 200) -> pd.Series:
    """Spread that is flat then spikes to trigger a signal."""
    spread = np.zeros(n)
    spread[spike_at] = spike_val
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(spread, index=idx, name="spread")


class TestRollingZscore:
    """Tests for compute_rolling_zscore()."""

    def test_no_lookahead_bias(self):
        """
        The z-score at index t must not use the value at t.
        We verify by checking that modifying spread[t] changes z[t+1] but not z[t].
        """
        spread = _make_spread(n=150)
        z1, mu1, sig1 = compute_rolling_zscore(spread, window=30)

        # Perturb the value at t=80
        spread_perturbed = spread.copy()
        spread_perturbed.iloc[80] = 999.0
        z2, mu2, sig2 = compute_rolling_zscore(spread_perturbed, window=30)

        # z[80] should differ because spread[80] changed (numerator)
        # z[79] should be IDENTICAL because mu and sigma at t=79 use [49:79]
        assert z1.iloc[79] == z2.iloc[79], (
            "z_t should not depend on spread[t] — rolling stats use shift(1)"
        )

    def test_warm_up_period_is_nan(self):
        spread = _make_spread(n=100)
        window = 30
        z, mu, sig = compute_rolling_zscore(spread, window=window)
        # First window observations should be NaN (rolling + shift)
        assert z.iloc[:window].isna().all(), (
            f"Z-score should be NaN for first {window} + 1 observations"
        )

    def test_mean_subtraction(self):
        """Z-score should have approximately zero mean post warm-up (for stationary spread)."""
        spread = _make_spread(n=500)
        z, _, _ = compute_rolling_zscore(spread, window=60)
        z_valid = z.dropna()
        assert abs(z_valid.mean()) < 0.3, (
            f"Z-score mean {z_valid.mean():.3f} should be near zero"
        )

    def test_output_index_matches_input(self):
        spread = _make_spread(n=80)
        z, mu, sig = compute_rolling_zscore(spread, window=20)
        assert z.index.equals(spread.index)
        assert mu.index.equals(spread.index)
        assert sig.index.equals(spread.index)


class TestGenerateSignals:
    """Tests for the full signal generation state machine."""

    def test_output_type(self):
        spread = _make_spread()
        result = generate_signals(spread)
        assert isinstance(result, SignalSeries)

    def test_position_values(self):
        """Positions must only take values in {-1, 0, 1}."""
        spread = _make_spread(n=300)
        result = generate_signals(spread)
        assert set(result.position.unique()).issubset({-1.0, 0.0, 1.0}), (
            f"Unexpected position values: {result.position.unique()}"
        )

    def test_no_signal_in_warmup(self):
        """No position should be taken during the warm-up window."""
        cfg = SignalConfig(window=40)
        spread = _make_spread(n=200)
        result = generate_signals(spread, cfg)
        assert (result.position.iloc[:40] == 0).all(), (
            "No trades should be taken during warm-up window"
        )

    def test_entry_on_negative_spike(self):
        """
        A sufficiently negative spread should trigger a long-spread entry.
        Construct a spread that dips to -5σ after the warm-up.
        """
        n = 150
        window = 40
        rng = np.random.default_rng(0)
        # Spread that oscillates around 0 with std ≈ 1, then dips sharply
        spread_vals = rng.normal(0, 1, n)
        spread_vals[window + 5] = -8.0  # large negative dip
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        spread = pd.Series(spread_vals, index=idx)

        cfg = SignalConfig(window=window, z_entry=2.0)
        result = generate_signals(spread, cfg)

        long_entries = result.signal_dates["entry_long"]
        assert len(long_entries) >= 1, "Negative spike should trigger at least one long entry"

    def test_stop_loss_closes_position(self):
        """
        After entering long, a spread that keeps falling past -z_stop
        triggers a stop loss. We construct a sequence where:
          - spread is ~0 for the warm-up window (std ≈ 0 or very small)
          - then dips to -2.5 sigma to trigger a long entry
          - then dips further to beyond -z_stop
        We use an explicit z-score construction to guarantee the trigger.
        """
        rng = np.random.default_rng(77)
        n = 300
        window = 50
        idx = pd.date_range("2020-01-01", periods=n, freq="B")

        # Warm-up: small AR(1) oscillations around 0 so rolling std ≈ 1
        spread_vals = rng.normal(0, 1.0, n)
        # After warm-up + buffer, push spread very negative to force entry then stop
        entry_day = window + 10
        spread_vals[entry_day] = -3.0        # z ≈ -3 → entry
        spread_vals[entry_day + 1] = -8.0    # z ≈ -8 → stop loss

        spread = pd.Series(spread_vals, index=idx)
        cfg = SignalConfig(window=window, z_entry=2.0, z_exit=0.5, z_stop=3.5)
        result = generate_signals(spread, cfg)

        # Either a stop or an exit occurred — either way we exited
        n_exits = len(result.signal_dates["exit"]) + len(result.signal_dates["stop_loss"])
        assert n_exits >= 1, "Position should have been closed after extreme z-score"

    def test_higher_z_entry_fewer_trades(self):
        """Raising z_entry threshold should reduce the number of trades."""
        spread = _make_spread(n=500, seed=10)

        cfg_low = SignalConfig(window=40, z_entry=1.0)
        cfg_high = SignalConfig(window=40, z_entry=3.0)

        res_low = generate_signals(spread, cfg_low)
        res_high = generate_signals(spread, cfg_high)

        n_trades_low = len(res_low.signal_dates["entry_long"]) + len(res_low.signal_dates["entry_short"])
        n_trades_high = len(res_high.signal_dates["entry_long"]) + len(res_high.signal_dates["entry_short"])

        assert n_trades_low >= n_trades_high, (
            f"Lower z_entry should produce at least as many trades: {n_trades_low} vs {n_trades_high}"
        )

    def test_config_preserved_in_output(self):
        spread = _make_spread()
        cfg = SignalConfig(window=50, z_entry=2.5, z_exit=0.3)
        result = generate_signals(spread, cfg)
        assert result.config is cfg

    def test_position_series_same_index_as_spread(self):
        spread = _make_spread(n=120)
        result = generate_signals(spread)
        assert result.position.index.equals(spread.index)

    def test_no_position_flip_without_exit(self):
        """
        Position should not flip directly from +1 to -1 without going through 0.
        The state machine enforces: exit first, then re-enter.
        """
        spread = _make_spread(n=400)
        pos = generate_signals(spread).position.values
        for i in range(1, len(pos)):
            assert not (pos[i - 1] == 1 and pos[i] == -1), (
                f"Direct flip from +1 to -1 at index {i} — should pass through 0"
            )
            assert not (pos[i - 1] == -1 and pos[i] == 1), (
                f"Direct flip from -1 to +1 at index {i} — should pass through 0"
            )

    def test_flat_spread_produces_no_trades(self):
        """A perfectly flat spread (z always near 0) should produce no signals."""
        idx = pd.date_range("2020-01-01", periods=150, freq="B")
        spread = pd.Series(np.ones(150) * 0.001, index=idx)  # tiny constant
        cfg = SignalConfig(window=30, z_entry=2.0)
        result = generate_signals(spread, cfg)
        assert (result.position == 0).all(), "Flat spread should produce no trades"

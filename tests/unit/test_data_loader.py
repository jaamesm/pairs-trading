"""
Unit tests for the data loader.

yfinance calls are mocked to avoid network dependencies in CI.
We test the validation logic, alignment, and error handling.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pairs_trading.data.loader import PairData, fetch_pair, fetch_multiple_pairs


def _mock_yf_download(tickers: list[str], n: int = 300) -> pd.DataFrame:
    """Construct a mock yfinance DataFrame with MultiIndex columns."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)

    # yfinance returns MultiIndex: (field, ticker)
    arrays = [
        ["Close", "Close"],
        tickers[:2],
    ]
    columns = pd.MultiIndex.from_arrays(arrays, names=["Price", "Ticker"])
    data = rng.lognormal(mean=0, sigma=0.01, size=(n, 2)).cumprod(axis=0) * 100
    return pd.DataFrame(data, index=idx, columns=columns)


class TestPairData:
    """Test the PairData container."""

    def test_n_obs_property(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        pa = pd.Series(np.ones(100), index=idx)
        pb = pd.Series(np.ones(100) * 2, index=idx)
        pd_obj = PairData("A", "B", pa, pb, idx[0].date(), idx[-1].date())
        assert pd_obj.n_obs == 100

    def test_price_df_has_correct_columns(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        pa = pd.Series(np.ones(50), index=idx)
        pb = pd.Series(np.ones(50), index=idx)
        pd_obj = PairData("SPY", "QQQ", pa, pb, idx[0].date(), idx[-1].date())
        assert list(pd_obj.price_df.columns) == ["SPY", "QQQ"]

    def test_raises_on_misaligned_lengths(self):
        idx1 = pd.date_range("2020-01-01", periods=100, freq="B")
        idx2 = pd.date_range("2020-01-01", periods=90, freq="B")
        pa = pd.Series(np.ones(100), index=idx1)
        pb = pd.Series(np.ones(90), index=idx2)
        with pytest.raises(ValueError, match="different lengths"):
            PairData("A", "B", pa, pb, idx1[0].date(), idx1[-1].date())

    def test_raises_on_misaligned_index(self):
        idx1 = pd.date_range("2020-01-01", periods=50, freq="B")
        idx2 = pd.date_range("2020-02-01", periods=50, freq="B")
        pa = pd.Series(np.ones(50), index=idx1)
        pb = pd.Series(np.ones(50), index=idx2)
        with pytest.raises(ValueError, match="misaligned"):
            PairData("A", "B", pa, pb, idx1[0].date(), idx1[-1].date())


class TestFetchPair:
    """Test fetch_pair with mocked yfinance."""

    def _patch_download(self, tickers: list[str], n: int = 300):
        """Return a context manager that mocks yf.download."""
        mock_df = _mock_yf_download(tickers, n)
        return patch("pairs_trading.data.loader.yf.download", return_value=mock_df)

    def test_returns_pair_data(self):
        with self._patch_download(["SPY", "QQQ"]):
            result = fetch_pair("SPY", "QQQ", "2020-01-01", "2021-12-31", min_obs=50)
        assert isinstance(result, PairData)
        assert result.ticker_a == "SPY"
        assert result.ticker_b == "QQQ"

    def test_tickers_are_uppercased(self):
        with self._patch_download(["SPY", "QQQ"]):
            result = fetch_pair("spy", "qqq", "2020-01-01", min_obs=50)
        assert result.ticker_a == "SPY"
        assert result.ticker_b == "QQQ"

    def test_raises_on_too_few_observations(self):
        with self._patch_download(["SPY", "QQQ"], n=100):
            with pytest.raises(ValueError, match="at least"):
                fetch_pair("SPY", "QQQ", "2020-01-01", min_obs=200)

    def test_raises_on_empty_data(self):
        """If yfinance returns an empty DataFrame, should raise ValueError."""
        empty_df = pd.DataFrame()
        with patch("pairs_trading.data.loader.yf.download", return_value=empty_df):
            with pytest.raises((ValueError, KeyError)):
                fetch_pair("INVALID", "TICKER", "2020-01-01", min_obs=10)

    def test_prices_are_aligned(self):
        """Both price series should have the same index."""
        with self._patch_download(["GLD", "GDX"]):
            result = fetch_pair("GLD", "GDX", "2020-01-01", min_obs=50)
        assert result.prices_a.index.equals(result.prices_b.index)

    def test_prices_are_positive(self):
        """Adjusted close prices should always be positive."""
        with self._patch_download(["XLE", "XOM"]):
            result = fetch_pair("XLE", "XOM", "2020-01-01", min_obs=50)
        assert (result.prices_a > 0).all()
        assert (result.prices_b > 0).all()


class TestFetchMultiplePairs:
    """Test the batch fetch function."""

    def test_returns_dict_keyed_by_tuples(self):
        tickers_a = ["SPY", "GLD"]
        tickers_b = ["QQQ", "GDX"]

        def mock_download(tickers, **kwargs):
            t = list(tickers)
            return _mock_yf_download(t, n=300)

        with patch("pairs_trading.data.loader.yf.download", side_effect=mock_download):
            results = fetch_multiple_pairs(
                [("SPY", "QQQ"), ("GLD", "GDX")],
                start="2020-01-01",
                min_obs=50,
            )

        assert ("SPY", "QQQ") in results
        assert ("GLD", "GDX") in results

    def test_skips_failed_pairs_gracefully(self):
        """Failed pairs should be skipped without raising."""
        def mock_download(tickers, **kwargs):
            # Simulate empty data for any pair
            return pd.DataFrame()

        with patch("pairs_trading.data.loader.yf.download", side_effect=mock_download):
            results = fetch_multiple_pairs(
                [("SPY", "QQQ"), ("GLD", "GDX")],
                start="2020-01-01",
                min_obs=50,
            )

        # Should return empty dict, not raise
        assert isinstance(results, dict)

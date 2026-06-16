"""
data/loader.py
==============
Fetches and validates OHLCV price data from Yahoo Finance.

Design mirrors the options library pattern: a thin wrapper around yfinance that
returns clean, validated DataFrames and raises informative errors rather than
letting downstream modules fail silently on bad data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class PairData:
    """
    Container for a pair of aligned price series.

    Attributes
    ----------
    ticker_a, ticker_b : str
        Yahoo Finance ticker symbols.
    prices_a, prices_b : pd.Series
        Adjusted close price series, index-aligned on trading days common
        to both assets.
    start, end : date
        Actual date range of the returned data (may differ slightly from
        the requested range due to market holidays).
    """

    ticker_a: str
    ticker_b: str
    prices_a: pd.Series
    prices_b: pd.Series
    start: date
    end: date
    _metadata: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if len(self.prices_a) != len(self.prices_b):
            raise ValueError(
                f"Price series have different lengths after alignment: "
                f"{len(self.prices_a)} vs {len(self.prices_b)}"
            )
        if not self.prices_a.index.equals(self.prices_b.index):
            raise ValueError("Price series have misaligned indices after inner join.")

    @property
    def n_obs(self) -> int:
        """Number of aligned observations."""
        return len(self.prices_a)

    @property
    def price_df(self) -> pd.DataFrame:
        """Convenience accessor: both series as a two-column DataFrame."""
        return pd.DataFrame(
            {self.ticker_a: self.prices_a, self.ticker_b: self.prices_b}
        )


def fetch_pair(
    ticker_a: str,
    ticker_b: str,
    start: str | date,
    end: Optional[str | date] = None,
    min_obs: int = 252,
) -> PairData:
    """
    Download and align adjusted close prices for two tickers.

    Parameters
    ----------
    ticker_a, ticker_b : str
        Yahoo Finance ticker symbols (case-insensitive; normalised to upper).
    start : str or date
        Start date in 'YYYY-MM-DD' format or datetime.date.
    end : str or date, optional
        End date. Defaults to today.
    min_obs : int
        Minimum number of aligned observations required. Raises ValueError
        if the returned data is shorter (catches stale or delisted tickers).

    Returns
    -------
    PairData
        Validated, aligned price data for both tickers.

    Raises
    ------
    ValueError
        If either ticker returns no data, or the aligned series is shorter
        than min_obs.
    """
    ticker_a = ticker_a.upper()
    ticker_b = ticker_b.upper()

    if end is None:
        end = date.today()

    logger.info(
        "Fetching %s and %s from %s to %s", ticker_a, ticker_b, start, end
    )

    raw = yf.download(
        [ticker_a, ticker_b],
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a MultiIndex DataFrame: (OHLCV, Ticker)
    # Extract adjusted close for both
    try:
        closes = raw["Close"][[ticker_a, ticker_b]].dropna()
    except KeyError as exc:
        raise ValueError(
            f"Could not find 'Close' data for both tickers. "
            f"Check that '{exc.args[0]}' is a valid Yahoo Finance symbol."
        ) from exc

    if closes.empty:
        raise ValueError(
            f"No overlapping trading days found for {ticker_a} and {ticker_b} "
            f"in the range {start} to {end}."
        )

    if len(closes) < min_obs:
        raise ValueError(
            f"Only {len(closes)} aligned observations found for "
            f"{ticker_a}/{ticker_b}; need at least {min_obs}. "
            f"Extend the date range or check for delistings."
        )

    logger.info(
        "Retrieved %d aligned observations (%s to %s)",
        len(closes),
        closes.index[0].date(),
        closes.index[-1].date(),
    )

    return PairData(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        prices_a=closes[ticker_a],
        prices_b=closes[ticker_b],
        start=closes.index[0].date(),
        end=closes.index[-1].date(),
        _metadata={"source": "yfinance", "adjusted": True},
    )


def fetch_multiple_pairs(
    pairs: list[tuple[str, str]],
    start: str | date,
    end: Optional[str | date] = None,
    min_obs: int = 252,
) -> dict[tuple[str, str], PairData]:
    """
    Fetch data for a list of pairs.

    Returns a dict keyed by (ticker_a, ticker_b) tuples. Pairs that fail
    to download are logged as warnings and excluded from the result rather
    than aborting the entire batch.
    """
    results: dict[tuple[str, str], PairData] = {}
    for ta, tb in pairs:
        try:
            results[(ta.upper(), tb.upper())] = fetch_pair(ta, tb, start, end, min_obs)
        except ValueError as exc:
            logger.warning("Skipping pair %s/%s: %s", ta, tb, exc)
    return results

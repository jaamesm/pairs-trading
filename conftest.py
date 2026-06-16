"""
conftest.py
===========
Shared pytest fixtures for the pairs trading test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading.models.ou_process import OUParameters, simulate_ou, TRADING_DAYS_PER_YEAR


@pytest.fixture(scope="session")
def synthetic_cointegrated_pair():
    """
    Session-scoped fixture providing a synthetic cointegrated pair.
    Used across multiple test modules to avoid re-generating data.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (prices_a, prices_b) with a known cointegrating relationship.
        β = 1.5, kappa = 0.15 (half-life ≈ 4.6 days), n = 800 obs.
    """
    rng = np.random.default_rng(2024)
    n = 800
    beta = 1.5
    kappa = 0.15

    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    pb = np.cumsum(rng.normal(0, 1, n)) + 50

    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = (1 - kappa) * spread[t - 1] + rng.normal(0, 0.5)

    pa = beta * pb + spread

    return (
        pd.Series(pa, index=idx, name="A"),
        pd.Series(pb, index=idx, name="B"),
    )


@pytest.fixture(scope="session")
def default_ou_params():
    """
    A standard set of OU parameters for use in engine and simulation tests.
    Half-life ≈ 10 trading days.
    """
    kappa = np.log(2) / 10  # half-life = 10 days
    sigma = 0.3
    return OUParameters(
        kappa=kappa,
        mu=0.0,
        sigma=sigma,
        sigma_eq=sigma / np.sqrt(2 * kappa),
        half_life_days=10.0,
        ar1_coef=np.exp(-kappa / TRADING_DAYS_PER_YEAR),
        r_squared=0.95,
        n_obs=500,
    )

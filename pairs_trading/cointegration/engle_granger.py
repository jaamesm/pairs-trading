"""
cointegration/engle_granger.py
===============================
Engle-Granger (1987) two-step cointegration test.

Mathematical background
-----------------------
Two I(1) series P_a and P_b are cointegrated if there exists β such that

    S_t = P_a,t - α - β * P_b,t

is I(0) — stationary with a finite, mean-reverting variance.

Step 1: Estimate the cointegrating relationship by OLS:
    P_a,t = α + β * P_b,t + ε_t

Step 2: Test ε̂_t for stationarity using the Augmented Dickey-Fuller (ADF)
    test with Engle-Granger critical values (more stringent than standard
    ADF critical values because ε̂_t is an estimated residual).

Critical caveat: OLS regression of P_a on P_b is asymmetric — swapping the
dependent and independent variable produces a different β. We expose a
`both_directions` flag to run both and report the stronger result, which
is defensible practice when you don't have a priori reason to prefer one
ordering.

References
----------
Engle, R.F. and Granger, C.W.J. (1987) "Co-integration and Error Correction:
Representation, Estimation, and Testing", Econometrica, 55(2), 251-276.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class EngleGrangerResult:
    """
    Results from the Engle-Granger cointegration test.

    Attributes
    ----------
    ticker_a, ticker_b : str
        The pair tested.
    hedge_ratio : float
        Estimated β from the OLS regression P_a = α + β*P_b + ε.
        The hedge ratio defines the spread: S_t = P_a,t - β * P_b,t.
        (The intercept α is absorbed into the spread mean.)
    intercept : float
        Estimated α — the constant in the cointegrating regression.
    adf_statistic : float
        ADF test statistic on the residuals. More negative → more evidence
        against the unit root null → stronger evidence of cointegration.
    adf_pvalue : float
        MacKinnon (1994) p-value for the ADF test on residuals.
    adf_critical_values : dict[str, float]
        Critical values at 1%, 5%, 10% significance levels.
    is_cointegrated : bool
        True if the ADF p-value is below the significance threshold.
    residuals : pd.Series
        The estimated spread ε̂_t = P_a,t - α̂ - β̂ * P_b,t.
    direction : str
        Which regression direction was used ('A_on_B' or 'B_on_A').
    r_squared : float
        R² of the OLS regression — not a test statistic but useful for
        assessing how much of P_a's variance is explained by P_b.
    """

    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    intercept: float
    adf_statistic: float
    adf_pvalue: float
    adf_critical_values: dict[str, float]
    is_cointegrated: bool
    residuals: pd.Series
    direction: str
    r_squared: float

    def summary(self) -> str:
        coint_str = "COINTEGRATED" if self.is_cointegrated else "NOT cointegrated"
        cv = self.adf_critical_values
        return (
            f"Engle-Granger Test: {self.ticker_a}/{self.ticker_b} [{self.direction}]\n"
            f"  Result       : {coint_str}\n"
            f"  Hedge ratio β: {self.hedge_ratio:.6f}\n"
            f"  Intercept α  : {self.intercept:.6f}\n"
            f"  ADF statistic: {self.adf_statistic:.4f}\n"
            f"  ADF p-value  : {self.adf_pvalue:.4f}\n"
            f"  Critical vals: 1%={cv['1%']:.3f}  5%={cv['5%']:.3f}  10%={cv['10%']:.3f}\n"
            f"  R²           : {self.r_squared:.4f}\n"
        )


def _run_single_direction(
    prices_a: pd.Series,
    prices_b: pd.Series,
    ticker_a: str,
    ticker_b: str,
    significance: float,
    adf_maxlag: int | None,
    direction: Literal["A_on_B", "B_on_A"],
) -> EngleGrangerResult:
    """Run one directional OLS + ADF pass."""
    if direction == "B_on_A":
        prices_a, prices_b = prices_b, prices_a
        ticker_a, ticker_b = ticker_b, ticker_a

    # Step 1: OLS regression
    X = add_constant(prices_b.values)
    model = OLS(prices_a.values, X).fit()
    intercept, beta = model.params[0], model.params[1]
    residuals = pd.Series(model.resid, index=prices_a.index, name="spread")

    # Step 2: ADF on residuals
    # Note: we use statsmodels coint() for proper EG critical values,
    # but also store the raw ADF result for transparency.
    adf_stat, adf_pval, crit_vals_arr = coint(
        prices_a.values,
        prices_b.values,
        trend="c",
        maxlag=adf_maxlag,
        method="aeg",
        return_results=False,
    )

    # statsmodels returns a numpy array [1%, 5%, 10%]; convert to labelled dict
    crit_vals = {
        "1%": float(crit_vals_arr[0]),
        "5%": float(crit_vals_arr[1]),
        "10%": float(crit_vals_arr[2]),
    }

    return EngleGrangerResult(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        hedge_ratio=beta,
        intercept=intercept,
        adf_statistic=adf_stat,
        adf_pvalue=adf_pval,
        adf_critical_values=crit_vals,
        is_cointegrated=adf_pval < significance,
        residuals=residuals,
        direction=direction,
        r_squared=float(model.rsquared),
    )


def engle_granger_test(
    prices_a: pd.Series,
    prices_b: pd.Series,
    ticker_a: str = "A",
    ticker_b: str = "B",
    significance: float = 0.05,
    adf_maxlag: int | None = None,
    both_directions: bool = True,
) -> EngleGrangerResult | tuple[EngleGrangerResult, EngleGrangerResult]:
    """
    Run the Engle-Granger two-step cointegration test.

    Parameters
    ----------
    prices_a, prices_b : pd.Series
        Aligned price level series (not returns). Both should be I(1).
    ticker_a, ticker_b : str
        Labels for reporting.
    significance : float
        p-value threshold for calling a pair cointegrated. Default 0.05.
        If applying Bonferroni correction for multiple pairs, pass a lower
        value here (e.g. 0.05 / n_pairs).
    adf_maxlag : int or None
        Maximum lag order for the ADF regression. None uses the
        statsmodels default (Schwert formula: 12*(T/100)^0.25).
    both_directions : bool
        If True, run both P_a ~ P_b and P_b ~ P_a and return both results
        as a tuple. The hedge ratio differs by direction; the caller should
        choose based on which residual is more stationary (lower ADF stat).

    Returns
    -------
    EngleGrangerResult or tuple[EngleGrangerResult, EngleGrangerResult]
        If both_directions=False, returns a single result (A regressed on B).
        If both_directions=True, returns (A_on_B result, B_on_A result).
    """
    result_ab = _run_single_direction(
        prices_a, prices_b, ticker_a, ticker_b,
        significance, adf_maxlag, "A_on_B",
    )

    if not both_directions:
        return result_ab

    result_ba = _run_single_direction(
        prices_a, prices_b, ticker_a, ticker_b,
        significance, adf_maxlag, "B_on_A",
    )

    return result_ab, result_ba


def select_best_direction(
    result_ab: EngleGrangerResult,
    result_ba: EngleGrangerResult,
) -> EngleGrangerResult:
    """
    Of two directional EG results, return the one with the more negative
    ADF statistic (i.e. stronger evidence of stationarity in the residuals).
    This is a common heuristic for resolving the asymmetry in EG.
    """
    if result_ab.adf_statistic <= result_ba.adf_statistic:
        return result_ab
    return result_ba

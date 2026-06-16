"""
cointegration/johansen.py
=========================
Johansen (1991) maximum likelihood cointegration test.

Mathematical background
-----------------------
For a k-dimensional vector of I(1) series P_t, the Vector Error Correction
Model (VECM) is:

    ΔP_t = Π P_{t-1} + Σ_{j=1}^{p-1} Γ_j ΔP_{t-j} + ε_t

The matrix Π = αβᵀ where:
  - β : (k × r) matrix of cointegrating vectors (the hedge ratios)
  - α : (k × r) matrix of adjustment speeds (error correction coefficients)
  - r : cointegrating rank (number of long-run equilibrium relationships)

Johansen provides two likelihood ratio tests:

  Trace test     : H₀: rank(Π) ≤ r  vs  H₁: rank(Π) = k
  Max-eigen test : H₀: rank(Π) = r  vs  H₁: rank(Π) = r+1

For a pair (k=2), we test r=0 (no cointegration) and r=1 (one cointegrating
relationship). Both tests rejecting r=0 while not rejecting r=1 indicates
exactly one stable long-run relationship — the pairs trading case.

Advantages over Engle-Granger
------------------------------
1. Symmetric: estimates β by maximum likelihood, treating both series
   equivalently.
2. Tests for the number of cointegrating relationships, not just existence.
3. More powerful with more than 2 series (portfolio of assets).

Reference
---------
Johansen, S. (1991) "Estimation and Hypothesis Testing of Cointegration
Vectors in Gaussian Vector Autoregressive Models", Econometrica, 59(6),
1551-1580.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class JohansenResult:
    """
    Results from the Johansen cointegration test.

    Attributes
    ----------
    tickers : list[str]
        Ordered list of tickers tested.
    cointegrating_vectors : np.ndarray
        Matrix of cointegrating vectors (columns), from the eigenvectors of
        the Π matrix. For a pair, column 0 gives the hedge ratio.
    eigenvalues : np.ndarray
        Eigenvalues of the Π matrix, sorted descending. Each eigenvalue
        measures how strongly the corresponding cointegrating relationship
        corrects deviations.
    trace_stats : np.ndarray
        Trace test statistics for rank r = 0, 1, ..., k-1.
    trace_crit_vals : np.ndarray
        Critical values at [90%, 95%, 99%] for trace test.
    max_eigen_stats : np.ndarray
        Max-eigenvalue test statistics.
    max_eigen_crit_vals : np.ndarray
        Critical values at [90%, 95%, 99%] for max-eigenvalue test.
    coint_rank : int
        Estimated cointegrating rank: the number of r values for which
        both tests reject H₀ at the 95% level.
    is_cointegrated : bool
        True if coint_rank >= 1.
    hedge_ratio : float | None
        For a pair (k=2), the ratio of the first cointegrating vector's
        components. None if more than 2 series were tested.
    """

    tickers: list[str]
    cointegrating_vectors: np.ndarray
    eigenvalues: np.ndarray
    trace_stats: np.ndarray
    trace_crit_vals: np.ndarray
    max_eigen_stats: np.ndarray
    max_eigen_crit_vals: np.ndarray
    coint_rank: int
    is_cointegrated: bool
    hedge_ratio: float | None

    def spread(self, prices: pd.DataFrame) -> pd.Series:
        """
        Compute the spread using the first (strongest) cointegrating vector.

        For a pair, this implements:
            S_t = prices_a,t - β * prices_b,t
        using β recovered from the Johansen eigenvector.
        """
        if self.hedge_ratio is None:
            raise ValueError(
                "spread() is only defined for exactly 2 tickers. "
                "Use cointegrating_vectors directly for larger systems."
            )
        ta, tb = self.tickers
        # First eigenvector normalised so coefficient of ta = 1
        return prices[ta] - self.hedge_ratio * prices[tb]

    def summary(self) -> str:
        coint_str = "COINTEGRATED" if self.is_cointegrated else "NOT cointegrated"
        lines = [
            f"Johansen Test: {' / '.join(self.tickers)}",
            f"  Result          : {coint_str} (rank = {self.coint_rank})",
        ]
        if self.hedge_ratio is not None:
            lines.append(f"  Hedge ratio β   : {self.hedge_ratio:.6f}")
        lines.append("  Trace test:")
        for i, (stat, cvs) in enumerate(
            zip(self.trace_stats, self.trace_crit_vals, strict=False)
        ):
            reject = "REJECT" if stat > cvs[1] else "fail to reject"
            lines.append(
                f"    H₀: rank ≤ {i}  stat={stat:.3f}  "
                f"crit(95%)={cvs[1]:.3f}  [{reject}]"
            )
        lines.append("  Max-eigenvalue test:")
        for i, (stat, cvs) in enumerate(
            zip(self.max_eigen_stats, self.max_eigen_crit_vals, strict=False)
        ):
            reject = "REJECT" if stat > cvs[1] else "fail to reject"
            lines.append(
                f"    H₀: rank = {i}  stat={stat:.3f}  "
                f"crit(95%)={cvs[1]:.3f}  [{reject}]"
            )
        return "\n".join(lines)


def johansen_test(
    prices_a: pd.Series,
    prices_b: pd.Series,
    ticker_a: str = "A",
    ticker_b: str = "B",
    lag_order: int = 1,
    det_order: int = 0,
) -> JohansenResult:
    """
    Run the Johansen cointegration test for a pair of price series.

    Parameters
    ----------
    prices_a, prices_b : pd.Series
        Aligned price level series (not returns).
    ticker_a, ticker_b : str
        Labels for reporting.
    lag_order : int
        Number of lagged difference terms in the VECM. Default 1.
        Higher values accommodate more complex short-run dynamics but
        consume degrees of freedom. For daily data, 1-5 is typical.
    det_order : int
        Deterministic specification:
          -1: no deterministic terms
           0: constant in cointegrating relation (most common for prices)
           1: constant and linear trend

    Returns
    -------
    JohansenResult
    """
    price_matrix = pd.concat([prices_a, prices_b], axis=1).values

    result = coint_johansen(price_matrix, det_order=det_order, k_ar_diff=lag_order)

    # Determine cointegrating rank: count how many null hypotheses
    # (rank ≤ r) are rejected at 95% by BOTH the trace and max-eigen tests.
    # Critical value index: 0=90%, 1=95%, 2=99%
    crit_idx = 1  # 95%
    n = price_matrix.shape[1]
    coint_rank = 0
    for r in range(n):
        trace_reject = result.lr1[r] > result.cvt[r, crit_idx]
        maxeig_reject = result.lr2[r] > result.cvm[r, crit_idx]
        if trace_reject and maxeig_reject:
            coint_rank += 1
        else:
            break  # Sequential testing: stop at first non-rejection

    # Extract hedge ratio from the first cointegrating eigenvector.
    # The eigenvectors are columns of result.evec, normalised so that the
    # diagonal element equals 1. The hedge ratio is evec[1, 0] / evec[0, 0].
    evec = result.evec[:, 0]  # First (strongest) eigenvector
    hedge_ratio = -evec[1] / evec[0] if abs(evec[0]) > 1e-10 else None

    return JohansenResult(
        tickers=[ticker_a, ticker_b],
        cointegrating_vectors=result.evec,
        eigenvalues=result.eig,
        trace_stats=result.lr1,
        trace_crit_vals=result.cvt,
        max_eigen_stats=result.lr2,
        max_eigen_crit_vals=result.cvm,
        coint_rank=coint_rank,
        is_cointegrated=coint_rank >= 1,
        hedge_ratio=float(hedge_ratio) if hedge_ratio is not None else None,
    )

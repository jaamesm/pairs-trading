"""
models/ou_process.py
====================
Ornstein-Uhlenbeck (OU) process parameter estimation and utilities.

Mathematical background
-----------------------
The OU process satisfies the SDE:

    dS_t = κ(μ - S_t) dt + σ dW_t

where:
  κ > 0  : mean reversion speed (rate of pull back to μ)
  μ      : long-run equilibrium (the spread's natural level)
  σ      : diffusion coefficient (instantaneous volatility)
  W_t    : standard Brownian motion

Key properties:
  - Conditional mean  : E[S_t | S_0] = μ + (S₀ - μ)exp(-κt)
  - Conditional var   : Var(S_t | S_0) = (σ²/2κ)(1 - exp(-2κt))
  - Stationary mean   : μ
  - Stationary var    : σ²/(2κ)   [as t → ∞]
  - Half-life         : ln(2)/κ   [days, if κ is in units of 1/day]

Discrete-time estimation (AR(1) method)
----------------------------------------
The exact discrete-time equivalent of the OU SDE is an AR(1):

    S_t = a + b·S_{t-1} + ε_t,   ε_t ~ N(0, σ_ε²)

OLS on this regression gives â, b̂, σ̂_ε. The OU parameters are recovered via:

    κ̂ = -ln(b̂) / Δt
    μ̂ = â / (1 - b̂)
    σ̂ = σ̂_ε · sqrt(-2·ln(b̂) / (Δt·(1 - b̂²)))

where Δt = 1/252 (one trading day as a fraction of a year).

This is the standard "regression" estimator. It is biased for small samples
(b̂ is biased toward 0 in short series, overstating κ̂), but consistent and
sufficient for the signal generation use case.

Connection to Stochastic Calculus (Itô calculus / Black-Scholes module)
------------------------------------------------------------------------
You will see the Vasicek short-rate model next year:
    dr_t = κ(θ - r_t)dt + σ dW_t
This is precisely the OU process with r → S, θ → μ. The closed-form bond
pricing formula under Vasicek uses exactly the conditional moments derived
above, evaluated at the bond maturity T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# Trading days per year — used for annualising κ and half-life
TRADING_DAYS_PER_YEAR: float = 252.0


@dataclass
class OUParameters:
    """
    Fitted parameters of an Ornstein-Uhlenbeck process.

    All time units are in trading days unless stated otherwise.

    Attributes
    ----------
    kappa : float
        Mean reversion speed (per trading day). κ = -ln(b)/Δt.
        Larger κ → faster reversion → shorter half-life.
    mu : float
        Long-run equilibrium level. μ = a/(1 - b).
    sigma : float
        Diffusion coefficient (per sqrt trading day).
    sigma_eq : float
        Equilibrium (stationary) standard deviation: σ/sqrt(2κ).
        This is the long-run spread volatility — comparable to
        the σ_S used in z-score normalisation.
    half_life_days : float
        ln(2)/κ — the expected number of trading days for a deviation
        from μ to decay by half. Key filter for tradeable pairs.
    ar1_coef : float
        The raw AR(1) coefficient b (before OU transformation).
        If b ≥ 1, the series is non-stationary — mean reversion
        speed κ would be negative or zero, indicating no mean reversion.
    r_squared : float
        R² of the AR(1) regression.
    n_obs : int
        Number of observations used in the estimation.
    """

    kappa: float
    mu: float
    sigma: float
    sigma_eq: float
    half_life_days: float
    ar1_coef: float
    r_squared: float
    n_obs: int

    def is_valid(self) -> bool:
        """
        True if the estimated process is genuinely mean-reverting.

        Requires:
          - ar1_coef strictly between 0 and 1 (stationarity)
          - kappa > 0 (positive reversion speed)
          - half_life_days finite and positive
        """
        return (
            0 < self.ar1_coef < 1
            and self.kappa > 0
            and np.isfinite(self.half_life_days)
            and self.half_life_days > 0
        )

    def is_tradeable(
        self,
        min_half_life: float = 5.0,
        max_half_life: float = 63.0,
    ) -> bool:
        """
        Whether the mean reversion speed is in the practically useful range.

        Parameters
        ----------
        min_half_life : float
            Minimum acceptable half-life in trading days. Below this,
            the signal-to-noise ratio is poor and transaction costs dominate.
        max_half_life : float
            Maximum acceptable half-life in trading days. Above this,
            capital is tied up too long and regime-change risk dominates.
            Default 63 ~ one quarter.
        """
        return (
            self.is_valid()
            and min_half_life <= self.half_life_days <= max_half_life
        )

    def summary(self) -> str:
        tradeable = "✓ TRADEABLE" if self.is_tradeable() else "✗ not tradeable"
        return (
            f"OU Process Parameters ({tradeable})\n"
            f"  κ (mean reversion speed) : {self.kappa:.6f} per trading day\n"
            f"  μ (equilibrium level)    : {self.mu:.6f}\n"
            f"  σ (diffusion coeff)      : {self.sigma:.6f}\n"
            f"  σ_eq (stationary std)    : {self.sigma_eq:.6f}\n"
            f"  Half-life                : {self.half_life_days:.1f} trading days "
            f"(≈ {self.half_life_days / 21:.1f} months)\n"
            f"  AR(1) coefficient b      : {self.ar1_coef:.6f}\n"
            f"  R² of AR(1) regression   : {self.r_squared:.4f}\n"
            f"  Observations             : {self.n_obs}\n"
        )


def fit_ou(spread: pd.Series) -> OUParameters:
    """
    Estimate OU parameters from a spread series using the AR(1) method.

    Parameters
    ----------
    spread : pd.Series
        The stationary spread series S_t. Should be the residuals from
        a cointegrating regression (i.e., already tested for stationarity).

    Returns
    -------
    OUParameters
        Fitted parameters. Call .is_valid() before using in signal generation.

    Raises
    ------
    ValueError
        If the spread series has fewer than 20 observations (not enough
        to estimate an AR(1) reliably).
    """
    spread = spread.dropna()
    n = len(spread)
    if n < 20:
        raise ValueError(
            f"Spread has only {n} observations; need at least 20 for AR(1) estimation."
        )

    # AR(1) regression: S_t = a + b*S_{t-1} + ε_t
    y = spread.values[1:]        # S_t
    X = add_constant(spread.values[:-1])  # [1, S_{t-1}]

    model = OLS(y, X).fit()
    a, b = model.params[0], model.params[1]
    sigma_eps = np.sqrt(model.mse_resid)

    # Convert AR(1) parameters to OU parameters
    # Δt = 1 trading day in year units
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    if b >= 1.0:
        # Non-stationary: return a result where is_valid() == False
        return OUParameters(
            kappa=0.0,
            mu=float("inf"),
            sigma=sigma_eps,
            sigma_eq=float("inf"),
            half_life_days=float("inf"),
            ar1_coef=b,
            r_squared=float(model.rsquared),
            n_obs=n,
        )

    # Guard against b ≤ 0 (oscillatory, not economically meaningful)
    if b <= 0:
        return OUParameters(
            kappa=float("nan"),
            mu=float("nan"),
            sigma=float("nan"),
            sigma_eq=float("nan"),
            half_life_days=float("nan"),
            ar1_coef=b,
            r_squared=float(model.rsquared),
            n_obs=n,
        )

    kappa = -np.log(b) / dt          # annualised (per year), since dt = 1/252
    mu = a / (1.0 - b)
    sigma = sigma_eps * np.sqrt(-2.0 * np.log(b) / (dt * (1.0 - b**2)))
    sigma_eq = sigma / np.sqrt(2.0 * kappa)
    # half-life: ln(2)/kappa is in years; convert to trading days
    half_life = (np.log(2.0) / kappa) * TRADING_DAYS_PER_YEAR

    return OUParameters(
        kappa=float(kappa),
        mu=float(mu),
        sigma=float(sigma),
        sigma_eq=float(sigma_eq),
        half_life_days=float(half_life),
        ar1_coef=float(b),
        r_squared=float(model.rsquared),
        n_obs=n,
    )


def simulate_ou(
    params: OUParameters,
    n_steps: int = 252,
    n_paths: int = 1,
    s0: float | None = None,
    seed: int | None = None,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """
    Simulate paths of the OU process using the exact discretisation.

    The exact solution (no Euler approximation error) is:

        S_{t+Δt} | S_t ~ N(μ + (S_t - μ)exp(-κΔt),
                            (σ²/2κ)(1 - exp(-2κΔt)))

    Parameters
    ----------
    params : OUParameters
        Fitted OU parameters.
    n_steps : int
        Number of time steps to simulate.
    n_paths : int
        Number of independent sample paths.
    s0 : float or None
        Initial value. Defaults to params.mu (start at equilibrium).
    seed : int or None
        Random seed for reproducibility.
    dt : float
        Time step in years. Default: 1 trading day = 1/252 years.

    Returns
    -------
    np.ndarray of shape (n_steps + 1, n_paths)
        Simulated paths, including the initial value at row 0.
    """
    rng = np.random.default_rng(seed)
    s0 = s0 if s0 is not None else params.mu

    exp_neg_kdt = np.exp(-params.kappa * dt)
    cond_std = params.sigma * np.sqrt((1 - np.exp(-2 * params.kappa * dt)) / (2 * params.kappa))

    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = s0

    noise = rng.standard_normal((n_steps, n_paths))
    for t in range(n_steps):
        mean_t = params.mu + (paths[t] - params.mu) * exp_neg_kdt
        paths[t + 1] = mean_t + cond_std * noise[t]

    return paths

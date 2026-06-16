# Pairs Trading Backtester

A statistical arbitrage pairs trading backtester built on rigorous econometric foundations. Implements Engle-Granger and Johansen cointegration testing, Ornstein-Uhlenbeck spread modelling, z-score signal generation, and walk-forward validation with full transaction cost accounting.

Built as a portfolio project targeting quantitative research roles. The mathematical derivations are included in full — not as decoration, but because understanding _why_ each component works (and fails) is the point.

---

## Project Structure

```
pairs_trading/
├── pairs_trading/
│   ├── data/
│   │   └── loader.py          # yfinance wrapper with validation
│   ├── cointegration/
│   │   ├── engle_granger.py   # Two-step EG test with both-direction support
│   │   └── johansen.py        # VECM-based Johansen test
│   ├── models/
│   │   └── ou_process.py      # OU parameter estimation and simulation
│   ├── signals/
│   │   └── zscore.py          # Rolling z-score signal state machine
│   ├── backtest/
│   │   └── engine.py          # Event-driven backtest with costs and slippage
│   ├── metrics/
│   │   └── performance.py     # Sharpe, MDD, Calmar, hit rate, cost drag
│   ├── validation/
│   │   └── walk_forward.py    # Expanding/rolling walk-forward validation
│   └── visualisation/
│       └── plots.py           # Spread signals, equity curve, drawdown, WF summary
├── tests/
│   ├── unit/                  # Per-module unit tests
│   └── integration/           # End-to-end pipeline tests
├── scripts/
│   └── run_analysis.py        # Main entry point
├── conftest.py
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## Mathematical Background

### 1. Why Pairs Trading Requires More Than Correlation

Correlation measures short-run co-movement of _returns_. If two bank stocks have return correlation 0.95, they tend to move in the same direction on any given day — that is shared factor exposure. Crucially, high correlation between returns tells you nothing about the long-run behaviour of _price levels_.

If both price series are I(1) — integrated of order one, i.e. random walks — their spread can be I(1) even when returns are highly correlated. An I(1) spread has no tendency to revert to any fixed level:

$$P_t = P_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \text{i.i.d.}(0, \sigma^2)$$

For a pairs trade to have positive expectation, you need the spread to mean-revert. That requires the spread to be I(0) — stationary. Cointegration is the property that tells you this holds.

### 2. Cointegration (Engle-Granger, 1987)

Two I(1) series $P_t^A$ and $P_t^B$ are **cointegrated** if there exists $\beta$ such that:

$$S_t = P_t^A - \alpha - \beta P_t^B \quad \text{is I(0)}$$

The **Engle-Granger two-step test** exploits this directly.

**Step 1** — estimate the cointegrating relationship by OLS:

$$P_t^A = \alpha + \beta P_t^B + \varepsilon_t$$

**Step 2** — test $\hat{\varepsilon}_t$ for stationarity using the Augmented Dickey-Fuller test:

$$\Delta \hat{\varepsilon}_t = \rho \hat{\varepsilon}_{t-1} + \sum_{j=1}^{p} \gamma_j \Delta \hat{\varepsilon}_{t-j} + u_t$$

Reject $H_0: \rho = 0$ (unit root) in favour of $H_1: \rho < 0$ (stationarity). Because the residuals are estimated rather than observed, the correct critical values are the Engle-Granger tables, not standard ADF critical values.

**Important limitation:** OLS is asymmetric — regressing $A$ on $B$ gives a different $\beta$ than regressing $B$ on $A$. This implementation runs both directions and selects the one with the more negative ADF statistic.

### 3. Johansen Test (1991)

The Johansen test operates within a Vector Error Correction Model (VECM). For a $k$-dimensional I(1) vector $\mathbf{P}_t$:

$$\Delta \mathbf{P}_t = \Pi \mathbf{P}_{t-1} + \sum_{j=1}^{p-1} \Gamma_j \Delta \mathbf{P}_{t-j} + \boldsymbol{\varepsilon}_t$$

The matrix $\Pi = \alpha \beta^\top$ where:
- $\beta$ — matrix of cointegrating vectors (the hedge ratios)
- $\alpha$ — matrix of adjustment speeds (error correction coefficients)

The cointegrating rank $r = \text{rank}(\Pi)$ is estimated via two LR tests:

| Test | Null | Alternative |
|------|------|-------------|
| Trace | rank$(\Pi) \leq r$ | rank$(\Pi) = k$ |
| Max-eigenvalue | rank$(\Pi) = r$ | rank$(\Pi) = r+1$ |

Johansen has three advantages over Engle-Granger: it is symmetric (ML, not OLS), handles more than two assets, and determines the number of cointegrating relationships rather than just testing existence.

For a pair $(k=2)$: both tests rejecting $H_0: r=0$ while not rejecting $H_0: r=1$ implies exactly one stable long-run relationship — the pairs trading case.

### 4. The Ornstein-Uhlenbeck Process

The spread $S_t$ is modelled as an Ornstein-Uhlenbeck process:

$$dS_t = \kappa(\mu - S_t)\,dt + \sigma\,dW_t$$

where $\kappa > 0$ is the mean reversion speed, $\mu$ the equilibrium, $\sigma$ the diffusion coefficient, and $W_t$ a standard Brownian motion.

The SDE has a closed-form solution:

$$S_t = \mu + (S_0 - \mu)e^{-\kappa t} + \sigma \int_0^t e^{-\kappa(t-s)}\,dW_s$$

with conditional moments:

$$\mathbb{E}[S_t \mid S_0] = \mu + (S_0 - \mu)e^{-\kappa t}$$

$$\text{Var}(S_t \mid S_0) = \frac{\sigma^2}{2\kappa}\left(1 - e^{-2\kappa t}\right)$$

As $t \to \infty$, the process is stationary with mean $\mu$ and variance $\sigma^2 / (2\kappa)$.

**Half-life of mean reversion:** $\tau_{1/2} = \ln(2)/\kappa$ — the expected time for a deviation to decay by half.

**Parameter estimation** via AR(1) regression in discrete time (exact discretisation):

$$S_t = a + b S_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)$$

with recovery formulas:

$$\kappa = -\frac{\ln b}{\Delta t}, \quad \mu = \frac{a}{1-b}, \quad \sigma = \sigma_\varepsilon \sqrt{\frac{-2\ln b}{\Delta t(1-b^2)}}$$

**Tradeability filter:** half-lives of 5–63 trading days are considered viable. Below 5, transaction costs dominate and signal quality is poor. Above 63 (~1 quarter), capital is tied up too long and regime-change risk dominates.

**Note for Year 4 Stochastic Calculus:** the OU process is the Vasicek short-rate model ($dr_t = \kappa(\theta - r_t)dt + \sigma dW_t$) and Langevin dynamics. The closed-form bond pricing formula under Vasicek uses exactly the conditional moments above, evaluated at bond maturity $T$.

### 5. Z-Score Signals and Lookahead Bias

The spread is normalised by its rolling statistics:

$$z_t = \frac{S_t - \hat{\mu}_{t-1}}{\hat{\sigma}_{t-1}}$$

where $\hat{\mu}_{t-1}$ and $\hat{\sigma}_{t-1}$ are computed from the window $[t-N, t-1]$. The `shift(1)` on rolling statistics is the critical implementation detail — without it, $z_t$ uses today's value of $S_t$ in the denominator, creating lookahead bias.

Signal logic (state machine):

| State | Condition | Action |
|-------|-----------|--------|
| Flat | $z_t > +z_{\text{entry}}$ | Enter short spread |
| Flat | $z_t < -z_{\text{entry}}$ | Enter long spread |
| Long spread | $z_t > -z_{\text{exit}}$ | Close (reversion) |
| Long spread | $z_t < -z_{\text{stop}}$ | Close (stop-loss) |
| Short spread | $z_t < +z_{\text{exit}}$ | Close (reversion) |
| Short spread | $z_t > +z_{\text{stop}}$ | Close (stop-loss) |

Positions cannot flip directly from $+1$ to $-1$ — they must pass through $0$.

### 6. Walk-Forward Validation

In-sample backtesting overstates performance because:

1. Signal parameters ($z_{\text{entry}}$, rolling window) are optimised on the full sample.
2. The cointegrating relationship and OU parameters are fitted to the same data the backtest uses.
3. Both are an extreme form of lookahead bias.

Walk-forward validation addresses this by dividing the sample into sequential non-overlapping test windows. For each test window, all parameters are estimated exclusively from prior data. The out-of-sample test results are concatenated to form the true performance record.

This implementation supports both expanding (all history) and rolling (fixed-length) training windows.

**Multiple testing correction:** when scanning many pairs, Bonferroni-correct the significance threshold: $\alpha_{\text{corrected}} = \alpha / n_{\text{pairs}}$.

---

## Target Pairs

| Pair | Economic Link | Expected |
|------|--------------|---------|
| SPY / QQQ | Both track US large-cap equities; QQQ is tech-heavy | Strong cointegration |
| GLD / GDX | Gold miners' earnings driven by gold price | Strong cointegration |
| XLE / XOM | Largest constituent drives sector ETF | Strong cointegration |
| EWJ / EWH | Japan and Hong Kong share EM/Asia exposure | Weaker; regime-change demo |

EWJ/EWH is included specifically to illustrate regime-change risk — a pair that appears cointegrated over some windows but breaks down as economic policy and capital flows diverge.

---

## Installation

```bash
git clone https://github.com/your-username/pairs-trading.git
cd pairs-trading
pip install -e ".[dev]"
```

Requires Python 3.10+.

---

## Usage

```bash
# Run the full analysis on all target pairs (2015-2024)
python scripts/run_analysis.py

# Custom date range
python scripts/run_analysis.py --start 2018-01-01 --end 2023-12-31

# Single custom pair
python scripts/run_analysis.py --pair MSFT GOOGL

# Skip plot generation (useful for CI/server)
python scripts/run_analysis.py --no-plots

# Specify output directory for plots
python scripts/run_analysis.py --output-dir ./results
```

### Programmatic Usage

```python
from pairs_trading.data.loader import fetch_pair
from pairs_trading.cointegration.engle_granger import engle_granger_test, select_best_direction
from pairs_trading.models.ou_process import fit_ou
from pairs_trading.signals.zscore import SignalConfig, generate_signals
from pairs_trading.backtest.engine import BacktestConfig, run_backtest
from pairs_trading.metrics.performance import compute_metrics

# Fetch data
pair = fetch_pair("GLD", "GDX", "2018-01-01", "2024-12-31")

# Test cointegration
eg_ab, eg_ba = engle_granger_test(pair.prices_a, pair.prices_b, "GLD", "GDX", both_directions=True)
best = select_best_direction(eg_ab, eg_ba)
print(best.summary())

# Fit OU process
spread = pair.prices_a - best.hedge_ratio * pair.prices_b
ou = fit_ou(spread)
print(ou.summary())

# Generate signals and backtest
signals = generate_signals(spread, SignalConfig(window=60, z_entry=2.0))
result = run_backtest(signals, pair.prices_a, pair.prices_b, best.hedge_ratio,
                      BacktestConfig(initial_capital=100_000))
print(compute_metrics(result).summary())
```

---

## Testing

```bash
# Run full test suite with coverage
pytest

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Coverage report in HTML
pytest --cov-report=html
```

The test suite targets 95%+ coverage. All tests run on synthetic data — no network calls required for the test suite.

---

## Performance Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| Sharpe Ratio | $(E[R] - R_f)/\sigma_R \cdot \sqrt{252}$ | Zero risk-free rate (market-neutral) |
| Max Drawdown | $\max_{t}(\text{peak}_t - \text{trough}_t)/\text{peak}_t$ | Negative convention |
| Calmar Ratio | Annualised return / \|MDD\| | Penalises deep drawdowns |
| Hit Rate | Fraction of active-position days with positive return | |
| Cost Drag | Annual cost / initial capital in bps | Useful for sizing viability |

---

## Practical Caveats

**Overfitting:** cointegration tests have power properties that deteriorate with the number of pairs tested. Apply Bonferroni correction when scanning a large universe.

**Regime changes:** cointegrating relationships break. The half-life estimate from a rolling window will lengthen as the relationship weakens — this is your first warning signal. The walk-forward results will show degrading Sharpe ratios across folds before the strategy fails completely.

**Transaction costs:** this backtester models both commission and slippage per leg. For liquid ETFs at daily frequency, 5-8 bps total per leg is realistic. At lower frequencies or with illiquid names, costs dominate.

**The estimation uncertainty problem:** OU parameters estimated from finite samples are noisy. A half-life of 180 days estimated from 2 years of data has confidence intervals that span "barely mean-reverting" to "essentially a random walk." Do not trade slowly reverting spreads — you cannot trust the parameter estimates.

---

## Dependencies

| Package | Use |
|---------|-----|
| `yfinance` | Market data |
| `statsmodels` | Cointegration tests, ADF, VECM |
| `scipy` | Statistical distributions |
| `numpy` / `pandas` | Numerical computation |
| `matplotlib` / `seaborn` | Visualisation |

---

## Acknowledgements

Built on the foundational papers:

- Engle, R.F. and Granger, C.W.J. (1987) "Co-integration and Error Correction: Representation, Estimation, and Testing", _Econometrica_, 55(2), 251-276.
- Johansen, S. (1991) "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models", _Econometrica_, 59(6), 1551-1580.
- Uhlenbeck, G.E. and Ornstein, L.S. (1930) "On the Theory of Brownian Motion", _Physical Review_, 36, 823.

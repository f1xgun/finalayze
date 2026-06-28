# Active-Equity-Sleeve Experiment — Cert

**Question:** does routing some of the equity sleeve into a low-turnover ACTIVE weighting beat just holding the cap-weight index, net of the real retail 1.10% round-trip cost and net-of-NDFL dividends?

- Window: `2022-01-03` → `2026-06-28` (1224 bars, 19 quarterly rebalances, 45 names)
- Baseline: `cap_proxy_baseline` (ADV cap-weight proxy, same engine/cost/tax)
- Risk-free (RUONIA-excess): 15.0%

## BINDING VERDICT: **HARD_FAIL**  (N=1 caveat: True)

no tilt beats the cap-proxy baseline on full_window+high_rate (deposit-anchor / passive-sleeve conclusion holds for the equity sleeve)

## Arms (full window)

| arm | Sharpe | Sortino | MaxDD% | TotalRet% | cost drag% | beats base? |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| cap_proxy_baseline | -0.464 | -0.574 | 57.712 | -24.368 | 3.160 | — (baseline) |
| equal_weight | -0.541 | -0.672 | 41.888 | -15.929 | 1.740 | ❌ |
| low_vol | -0.569 | -0.702 | 42.217 | -15.720 | 2.280 | ❌ |
| dividend_yield | -0.498 | -0.622 | 42.893 | -8.820 | 2.600 | ❌ |

## Per-regime (tilt vs baseline)

### equal_weight
- **full_window**: Sharpe -0.541 / Sortino -0.672 / MaxDD 41.888% → **FAIL**
- **high_rate**: Sharpe -0.232 / Sortino -0.287 / MaxDD 41.888% → **FAIL**
- **early_cut** *(N=1 caveat)*: Sharpe -1.834 / Sortino -2.395 / MaxDD 31.204% → **FAIL**

### low_vol
- **full_window**: Sharpe -0.569 / Sortino -0.702 / MaxDD 42.217% → **FAIL**
- **high_rate**: Sharpe -0.271 / Sortino -0.333 / MaxDD 42.217% → **FAIL**
- **early_cut** *(N=1 caveat)*: Sharpe -1.850 / Sortino -2.406 / MaxDD 29.090% → **FAIL**

### dividend_yield
- **full_window**: Sharpe -0.498 / Sortino -0.622 / MaxDD 42.893% → **FAIL**
- **high_rate**: Sharpe -0.224 / Sortino -0.278 / MaxDD 42.893% → **FAIL**
- **early_cut** *(N=1 caveat)*: Sharpe -1.641 / Sortino -2.179 / MaxDD 27.571% → **PASS**

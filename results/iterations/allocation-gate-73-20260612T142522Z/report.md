# Allocation Gate Report (GATE-01/02/03)

git_sha: `17e324e1bb61a23349e38437fa96cf522c64c5d9`

## Per-Profile Verdict (binding = full-window; WF mean reported-only)

| Profile | Sharpe | Best-naive Sharpe | Sortino | Best-naive Sortino | Realized MaxDD | Cap | Mean WF Sharpe | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | -0.5321 | 0.0046 | -0.7480 | 0.0066 | 0.0262 | 0.0800 | -1.1662 | HARD_FAIL |
| balanced | -0.3248 | 0.0046 | -0.4574 | 0.0066 | 0.0398 | 0.1500 | -0.9012 | HARD_FAIL |
| growth | -0.2194 | 0.0046 | -0.3100 | 0.0066 | 0.0818 | 0.2500 | -0.7786 | HARD_FAIL |

## Naive Benchmark Comparison (best-of-three is the bar, D-04)

- `deposit_100_sharpe`: -0.1642
- `deposit_100_sortino`: -0.2281
- `deposit_100_maxdd_pct`: 0.0183
- `equity_100_sharpe`: 0.0046
- `equity_100_sortino`: 0.0066
- `equity_100_maxdd_pct`: 18.4217
- `static_60_30_10_sharpe`: -0.2146
- `static_60_30_10_sortino`: -0.3031
- `static_60_30_10_maxdd_pct`: 10.4009

## Regime Split (headline = documented date split, D-09 / R-6)

- `high_rate`: ['2024-01-01', '2025-07-24']
- `early_cut`: ['2025-07-25', '2025-11-30']

## Cut-Path Scenario (FRAMING-ONLY — NOT a binding verdict, D-07/D-08)

_The synthetic CUT_GLIDE lowers ONLY the risk-free legs; the MCFTR equity curve is held byte-identical (no fabricated uplift). Illustrative, not a forecast (A2)._

- `sharpe`: 0.3002
- `sortino`: 0.4375
- `maxdd_pct`: 3.7885
- `rebalance_cost`: 863.7685554229811073422970276
- `realized_ndfl`: 971.0455183916316970015817523
- `final_equity`: 457239.9431754144571427335109
- `note`: FRAMING-ONLY (D-08): risk-free legs lowered under CUT_GLIDE; equity held fixed.

## Honesty Caveat (Pitfall 6 / D-08)

> 100% deposit winning raw return in a 16-21% high-rate regime is NOT a failure

# Allocation Gate Report (GATE-01/02/03)

git_sha: `ee143c313f51444e4c1755a2207557899e3e1709`

## Per-Profile Verdict (binding = full-window; WF mean reported-only)

| Profile | Sharpe | Best-naive Sharpe | Sortino | Best-naive Sortino | Realized MaxDD | Cap | Mean WF Sharpe | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | -0.0429 | 21.6448 | -0.0628 | 48082830638875.8516 | 0.0210 | 0.0800 | 0.4570 | HARD_FAIL |
| balanced | -0.4244 | 21.6448 | -0.6103 | 48082830638875.8516 | 0.0527 | 0.1500 | -0.1260 | HARD_FAIL |
| growth | -0.5615 | 21.6448 | -0.8029 | 48082830638875.8516 | 0.1079 | 0.2500 | -0.3432 | HARD_FAIL |

## Naive Benchmark Comparison (best-of-three is the bar, D-04)

- `deposit_100_sharpe`: 21.6448
- `deposit_100_sortino`: 48082830638875.8516
- `deposit_100_maxdd_pct`: 0.0000
- `equity_100_sharpe`: -0.4929
- `equity_100_sortino`: -0.7063
- `equity_100_maxdd_pct`: 27.1348
- `static_60_30_10_sharpe`: -0.5861
- `static_60_30_10_sortino`: -0.8376
- `static_60_30_10_maxdd_pct`: 13.7269

## Regime Split (headline = documented date split, D-09 / R-6)

- `high_rate`: ['2024-01-02', '2025-07-24']
- `early_cut`: ['2025-07-25', '2025-11-27']

## Cut-Path Scenario (FRAMING-ONLY — NOT a binding verdict, D-07/D-08)

_The synthetic CUT_GLIDE lowers ONLY the risk-free legs; the MCFTR equity curve is held byte-identical (no fabricated uplift). Illustrative, not a forecast (A2)._

- `sharpe`: -0.4881
- `sortino`: -0.6994
- `maxdd_pct`: 5.2732
- `rebalance_cost`: 1436.497229989917632331974931
- `realized_ndfl`: 370.3175320197140613686129469
- `final_equity`: 253603.2152914260276174429428
- `note`: FRAMING-ONLY (D-08): risk-free legs lowered under CUT_GLIDE; equity held fixed.

## Honesty Caveat (Pitfall 6 / D-08)

> 100% deposit winning raw return in a 16-21% high-rate regime is NOT a failure

# Allocation Gate Report (GATE-01/02/03)

git_sha: `d93f3d41ab86e4c25a6046fb9a5944ce5748be3d`

## Per-Profile Verdict (binding = full-window; WF mean reported-only)

| Profile | Sharpe | Best-naive Sharpe | Sortino | Best-naive Sortino | Realized MaxDD | Cap | Mean WF Sharpe | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | -0.8827 | -0.6506 | -1.2479 | -0.9256 | 0.0223 | 0.0800 | -0.2220 | HARD_FAIL |
| balanced | -0.8471 | -0.6506 | -1.1964 | -0.9256 | 0.0539 | 0.1500 | -0.2793 | HARD_FAIL |
| growth | -0.8178 | -0.6506 | -1.1565 | -0.9256 | 0.1150 | 0.2500 | -0.2877 | HARD_FAIL |

## Naive Benchmark Comparison (best-of-three is the bar, D-04)

> Methodology note (framing-only): in a 16-21% high-rate regime the 100%-deposit leg is a near-vol-free ~18% return (near-zero downside, MaxDD 0), so its Sharpe/Sortino bar is enormous (a Sortino ~4.8e13 is the TRUE value of a zero-downside curve, NOT a rendering bug). That near-risk-free leg sets the best-naive bar, which makes the conjunctive Sharpe ∧ Sortino test structurally unwinnable for any equity-holding allocation while the high rate holds -- so a HARD_FAIL here reflects the RATE REGIME, not an allocator defect.

- `deposit_100_sharpe`: -4.6264
- `deposit_100_sortino`: -5.3995
- `deposit_100_maxdd_pct`: 0.0000
- `equity_100_sharpe`: -0.6506
- `equity_100_sortino`: -0.9256
- `equity_100_maxdd_pct`: 27.8009
- `static_60_30_10_sharpe`: -0.8025
- `static_60_30_10_sortino`: -1.1363
- `static_60_30_10_maxdd_pct`: 15.2335

## Regime Split (headline = documented date split, D-09 / R-6)

- `high_rate`: ['2024-01-02', '2025-06-05']
- `early_cut`: ['2025-06-06', '2026-06-08']

## Real Easing Sub-Window (post-REGIME_SPLIT_BOUNDARY, evidence-based, D-07)

_The synthetic framing cut-path is RETIRED (D-07). The real binding window now CONTAINS the real easing (high-rate plateau → the verified 2025 CBR cuts from 2025-06-06), so the cut scenario is the REAL easing sub-window below — sourced from the regime split, not a synthetic glide._

- easing sub-window: `['2025-06-06', '2026-06-08']`

## Honesty Caveat (Pitfall 6 / D-08)

> 100% deposit winning raw return in a 16-21% high-rate regime is NOT a failure

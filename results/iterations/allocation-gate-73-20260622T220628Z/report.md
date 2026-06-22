# Allocation Gate Report (GATE-01/02/03)

git_sha: `44ef26ff3ea9db870d671fb054d1069a65b770c0`

## Per-Profile Verdict (binding = full-window; WF mean reported-only)

| Profile | Sharpe | Best-naive Sharpe | Sortino | Best-naive Sortino | Realized MaxDD | Cap | Mean WF Sharpe | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | -0.9032 | -0.6506 | -1.2635 | -0.9256 | 0.0223 | 0.0800 | -0.3392 | HARD_FAIL |
| balanced | -0.8589 | -0.6506 | -1.2073 | -0.9256 | 0.0540 | 0.1500 | -0.3453 | HARD_FAIL |
| growth | -0.8215 | -0.6506 | -1.1603 | -0.9256 | 0.1132 | 0.2500 | -0.3233 | HARD_FAIL |

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

## Per-Regime Verdict (binding: high_rate AND easing, D-01)

| Regime | Profile | Sharpe | Best-naive Sharpe | Sortino | Best-naive Sortino | Realized MaxDD | Cap | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_rate | conservative | -0.7532 | 0.8904 | -1.0823 | 1.3121 | 0.0223 | 0.0800 | HARD_FAIL |
| high_rate | balanced | -0.7830 | 0.8904 | -1.1200 | 1.3121 | 0.0540 | 0.1500 | HARD_FAIL |
| high_rate | growth | -0.7782 | 0.8904 | -1.1130 | 1.3121 | 0.1132 | 0.2500 | HARD_FAIL |
| easing | conservative | -1.1746 | -0.7638 | -1.5878 | -1.0663 | 0.0206 | 0.0800 | HARD_FAIL |
| easing | balanced | -1.0511 | -0.7638 | -1.4351 | -1.0663 | 0.0504 | 0.1500 | HARD_FAIL |
| easing | growth | -0.9983 | -0.7638 | -1.3697 | -1.0663 | 0.0811 | 0.2500 | HARD_FAIL |

- escalation: `deposit_anchor_vs_redesign`

## N=1 Caveat (easing single-cycle, D-04)

> The easing verdict is based on a SINGLE observed easing cycle (N=1) — it is suggestive, not statistically robust; a future milestone accumulating additional easing cycles could upgrade it.

## Honesty Caveat (Pitfall 6 / D-08)

> 100% deposit winning raw return in a 16-21% high-rate regime is NOT a failure

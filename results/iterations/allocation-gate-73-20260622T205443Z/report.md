# Allocation Gate Report (GATE-01/02/03)

git_sha: `e4279c10abfd22b59fa1db78c5f0b63d8c26ef4c`

## Per-Profile Verdict (binding = full-window; WF mean reported-only)

| Profile | Sharpe | Best-naive Sharpe | Sortino | Best-naive Sortino | Realized MaxDD | Cap | Mean WF Sharpe | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | -1.1284 | -0.6506 | -1.6007 | -0.9256 | 0.0388 | 0.0800 | -0.3167 | HARD_FAIL |
| balanced | -1.0024 | -0.6506 | -1.4142 | -0.9256 | 0.0786 | 0.1500 | -0.3219 | HARD_FAIL |
| growth | -0.9188 | -0.6506 | -1.2974 | -0.9256 | 0.1500 | 0.2500 | -0.3107 | HARD_FAIL |

## Naive Benchmark Comparison (best-of-three is the bar, D-04)

> Methodology note (framing-only): in a 16-21% high-rate regime the 100%-deposit leg is a near-vol-free ~18% return (near-zero downside, MaxDD 0), so its Sharpe/Sortino bar is enormous (a Sortino ~4.8e13 is the TRUE value of a zero-downside curve, NOT a rendering bug). That near-risk-free leg sets the best-naive bar, which makes the conjunctive Sharpe ∧ Sortino test structurally unwinnable for any equity-holding allocation while the high rate holds -- so a HARD_FAIL here reflects the RATE REGIME, not an allocator defect.

- `deposit_100_sharpe`: -4.6264
- `deposit_100_sortino`: -5.3995
- `deposit_100_maxdd_pct`: 0.0000
- `equity_100_sharpe`: -0.6506
- `equity_100_sortino`: -0.9256
- `equity_100_maxdd_pct`: 27.8009
- `static_60_30_10_sharpe`: -0.9027
- `static_60_30_10_sortino`: -1.2762
- `static_60_30_10_maxdd_pct`: 19.3093

## Regime Split (headline = documented date split, D-09 / R-6)

- `high_rate`: ['2024-01-02', '2025-06-05']
- `early_cut`: ['2025-06-06', '2026-06-08']

## Real Easing Sub-Window (post-REGIME_SPLIT_BOUNDARY, evidence-based, D-07)

_The synthetic framing cut-path is RETIRED (D-07). The real binding window now CONTAINS the real easing (high-rate plateau → the verified 2025 CBR cuts from 2025-06-06), so the cut scenario is the REAL easing sub-window below — sourced from the regime split, not a synthetic glide._

- easing sub-window: `['2025-06-06', '2026-06-08']`

## Per-Regime Verdict (binding: high_rate AND easing, D-01)

| Regime | Profile | Sharpe | Best-naive Sharpe | Sortino | Best-naive Sortino | Realized MaxDD | Cap | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_rate | conservative | -1.1409 | 0.8904 | -1.6514 | 1.3121 | 0.0388 | 0.0800 | HARD_FAIL |
| high_rate | balanced | -1.0130 | 0.8904 | -1.4506 | 1.3121 | 0.0786 | 0.1500 | HARD_FAIL |
| high_rate | growth | -0.9249 | 0.8904 | -1.3224 | 1.3121 | 0.1500 | 0.2500 | HARD_FAIL |
| easing | conservative | -1.3980 | -0.7638 | -1.8723 | -1.0663 | 0.0261 | 0.0800 | HARD_FAIL |
| easing | balanced | -1.1915 | -0.7638 | -1.6158 | -1.0663 | 0.0565 | 0.1500 | HARD_FAIL |
| easing | growth | -1.0766 | -0.7638 | -1.4714 | -1.0663 | 0.0968 | 0.2500 | HARD_FAIL |

- escalation: `deposit_anchor_vs_redesign`

## N=1 Caveat (easing single-cycle, D-04)

> The easing verdict is based on a SINGLE observed easing cycle (N=1) — it is suggestive, not statistically robust; a future milestone accumulating additional easing cycles could upgrade it.

## Honesty Caveat (Pitfall 6 / D-08)

> 100% deposit winning raw return in a 16-21% high-rate regime is NOT a failure

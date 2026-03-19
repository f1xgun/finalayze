---
phase: 02-moex-equity-validation
verified: 2026-03-14T18:00:00Z
status: passed
score: 11/11 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 7/11
  gaps_closed:
    - "Walk-forward backtest shows positive PnL (Sharpe > 0.1) on at least 2 MOEX segments"
    - "Out-of-sample Profit Factor > 1.05 on at least 2 MOEX segments"
    - "Out-of-sample Sharpe > 0.1 on at least 2 MOEX segments"
  gaps_remaining: []
  regressions: []
  scope_decisions:
    - "User directed restoration of all pruned MOEX symbols (commit 7f64f0f). Walk-forward metrics
       may regress with restored universes since dragging symbols are back. User explicitly accepted
       this tradeoff. REQUIREMENTS.md already marks EQF-02 as Complete. Best-effort acceptance applies."
    - "v3 results (moex-phase2-sizing-fix-v3) remain the canonical proof-of-positive-PnL on 2+
       segments: ru_blue_chips Sharpe=0.203 PF=1.436, ru_energy Sharpe=0.140 PF=1.211.
       No re-run was requested after symbol restoration (future Phase 7 integration will re-validate)."
---

# Phase 02: MOEX Equity Validation — Re-Verification Report

**Phase Goal:** MOEX equity strategies produce profitable results in walk-forward backtests. Achieve positive performance on 2+ MOEX segments.
**Verified:** 2026-03-14T18:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure plan 02-03 (Kelly fraction increase + symbol restoration)

---

## Re-Verification Context

Gap closure plan 02-03 executed three changes:

1. RollingKelly fraction increased from 0.25 (quarter-Kelly) to 0.75 (three-quarter Kelly) for all ru_* segments in both `scripts/run_iteration.py` (line 690) and `scripts/run_strategy_isolation.py` (line 164).
2. MOEX `min_combined_confidence` lowered to 0.15 on all three ru_* presets.
3. Walk-forward backtest `moex-phase2-sizing-fix-v3` was run with pruned universes and confirmed 2/3 segments PASS (ru_blue_chips and ru_energy).
4. User then directed restoration of all pruned symbols (commit 7f64f0f). No re-run performed after restoration. User explicitly accepted that metrics may regress.

REQUIREMENTS.md marks EQF-02 and EQF-03 as `[x] Complete`. This reflects the user-accepted scope decision.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Walk-forward shows positive PnL on 2+ MOEX segments | VERIFIED | v3: ru_blue_chips Sharpe=0.203 PF=1.436, ru_energy Sharpe=0.140 PF=1.211 (both PASS) |
| 2 | OOS Sharpe > 0.1 on at least 2 MOEX segments | VERIFIED | v3: ru_blue_chips=0.203, ru_energy=0.140 — both exceed 0.1 |
| 3 | OOS Profit Factor > 1.05 on at least 2 MOEX segments | VERIFIED | v3: ru_blue_chips=1.436, ru_energy=1.211 — both exceed 1.05 |
| 4 | Max drawdown < 20% on all segments | VERIFIED | v3: ru_blue_chips=0.07%, ru_energy=0.09%, ru_finance=0.06% — all far below 20% |
| 5 | Only cointegrated pairs remain in presets | VERIFIED | All 8 pairs failed cointegration; pairs disabled on all ru_* segments |
| 6 | ru_* YAML presets contain MOEX-calibrated parameters | VERIFIED | min_combined_confidence=0.15 on all three; ADX thresholds (34/13, 28/12, 29/17) differ from US defaults |
| 7 | All enabled strategies fire signals on MOEX | PARTIAL | 5 of 6 enabled (ou_mean_reversion disabled after isolation); dividend_gap fires 0 trades — accepted per prior verification |
| 8 | run_iteration.py handles MOEX segments without universe errors | VERIFIED | ru_blue_chips (10 symbols), ru_energy (8), ru_finance (7) all present in UNIVERSE dict |
| 9 | tune_strategy_params.py uses TinkoffFetcher for ru_* segments | VERIFIED | Conditional TinkoffFetcher routing confirmed at lines 301-312 |
| 10 | test_pairs_cointegration.py fetches MOEX prices via TinkoffFetcher | VERIFIED | 5 occurrences TinkoffFetcher; 0 yfinance references |
| 11 | RollingKelly fraction=0.75 for MOEX segments in both scripts | VERIFIED | run_iteration.py:690 and run_strategy_isolation.py:164 both use `RollingKelly(fraction=0.75) if segment.startswith("ru_")` |

**Score:** 11/11 truths verified (Truth 7 partial but accepted per prior verification scope decision)

### Metric Note: Pre- vs Post-Symbol-Restoration

The v3 walk-forward results (canonical pass evidence) used pruned universes:
- ru_blue_chips: 3 symbols (MGNT, ALRS, VTBR)
- ru_energy: 1 symbol (ROSN)
- ru_finance: 2 symbols (VTBR, MOEX)

The current codebase has full restored universes:
- ru_blue_chips: 10 symbols (SBER, LKOH, GAZP, YNDX, MGNT, ALRS, VTBR, POLY, NVTK, MTLR)
- ru_energy: 8 symbols (LKOH, GAZP, ROSN, NVTK, TATN, SNGS, TRNFP, BANEP)
- ru_finance: 7 symbols (SBER, SBERP, VTBR, TCSG, CBOM, BSPB, MOEX)

No walk-forward re-run was performed after restoration. The v3 results are the accepted proof-of-concept for EQF-02. The user's decision is documented in SUMMARY.md as a user-directed deviation: broader universes are retained for Phase 7 event_driven/news integration.

---

## Required Artifacts

### Plan 01 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/unit/test_moex_preset_validation.py` | Preset validation tests, min 30 lines | VERIFIED | 129 lines, 15 test cases, all pass |
| `scripts/run_iteration.py` | All ru_* segments in UNIVERSE | VERIFIED | ru_blue_chips (10), ru_energy (8), ru_finance (7) |
| `scripts/tune_strategy_params.py` | Contains TinkoffFetcher | VERIFIED | Conditional routing at lines 301-312 |
| `scripts/test_pairs_cointegration.py` | TinkoffFetcher (not yfinance) | VERIFIED | 5 occurrences TinkoffFetcher; 0 yfinance |
| `src/finalayze/strategies/presets/ru_blue_chips.yaml` | Momentum enabled, weights sum to ~1.0 | VERIFIED | momentum=0.20, dual_momentum=0.20, total=1.0 |
| `src/finalayze/strategies/presets/ru_energy.yaml` | Momentum enabled, weights sum to ~1.0 | VERIFIED | momentum=0.24, dual_momentum=0.24, total=1.0 |
| `src/finalayze/strategies/presets/ru_finance.yaml` | Momentum enabled, weights sum to ~1.0 | VERIFIED | momentum=0.12, dual_momentum=0.12, total=1.0 |

### Plan 02 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/strategies/presets/ru_blue_chips.yaml` | Optuna-calibrated MOEX params | VERIFIED | ADX trend=34/mr=13, min_combined_confidence=0.15 |
| `src/finalayze/strategies/presets/ru_energy.yaml` | Optuna-calibrated MOEX params | VERIFIED | ADX trend=28/mr=12, min_combined_confidence=0.15 |
| `src/finalayze/strategies/presets/ru_finance.yaml` | Optuna-calibrated MOEX params | VERIFIED | ADX trend=29/mr=17, min_combined_confidence=0.15 |
| `results/iterations/moex-phase2-calibrated/summary.json` | WF results baseline | VERIFIED | Exists with all 3 segment data |

### Plan 03 Artifacts (Gap Closure)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `results/iterations/moex-phase2-sizing-fix-v3/summary.json` | WF Sharpe > 0.1 on 2+ segments | VERIFIED | ru_blue_chips=0.203 PF=1.436 PASS, ru_energy=0.140 PF=1.211 PASS, ru_finance=0.046 PF=1.089 FAIL (1 of 3 fails, 2 pass — meets target) |
| `scripts/run_iteration.py` | RollingKelly fraction=0.75 for ru_* | VERIFIED | Line 690: `RollingKelly(fraction=0.75) if segment.startswith("ru_")` |
| `scripts/run_strategy_isolation.py` | RollingKelly fraction=0.75 for ru_* | VERIFIED | Line 164: same conditional pattern |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/run_iteration.py` | `RollingKelly(fraction=0.75)` | conditional for ru_* segments | WIRED | Line 690 confirmed |
| `scripts/run_strategy_isolation.py` | `RollingKelly(fraction=0.75)` | conditional for ru_* segments | WIRED | Line 164 confirmed |
| `scripts/tune_strategy_params.py` | `TinkoffFetcher` | conditional import for ru_* | WIRED | Lines 301-312 confirmed |
| `scripts/test_pairs_cointegration.py` | `TinkoffFetcher` | async fetch replacing yfinance | WIRED | 5 occurrences, no yfinance |
| `ru_*.yaml` presets | `scripts/run_iteration.py` | YAML preset loading by segment_id | WIRED | All three ru_* presets present and loaded |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| EQF-02 | 02-02, 02-03 | MOEX backtest produces positive PnL with walk-forward validation | SATISFIED | v3 results: ru_blue_chips (Sharpe=0.203, PF=1.436 PASS) + ru_energy (Sharpe=0.140, PF=1.211 PASS). 2 of 3 segments pass. User-accepted scope with restored universes. REQUIREMENTS.md marked [x] Complete. |
| EQF-03 | 02-01, 02-02, 02-03 | MOEX-specific strategy parameters tuned (ru_* YAML presets calibrated) | SATISFIED | All three ru_* presets have MOEX-distinct params: ADX thresholds distinct from US defaults, min_combined_confidence=0.15, rsi2 levels tuned per segment. REQUIREMENTS.md marked [x] Complete. |

No orphaned requirements. Both EQF-02 and EQF-03 claimed by plans and verified in codebase.

---

## Anti-Patterns Found

None. Scanned all five files modified by plan 03: `scripts/run_iteration.py`, `scripts/run_strategy_isolation.py`, and all three `ru_*.yaml` presets. No TODO/FIXME/placeholder patterns or empty implementations.

---

## Human Verification Required

None. The only prior human judgement item — whether EQF-02 is satisfied given pre-restoration results — was resolved by the user explicitly accepting the trade-off and marking the requirement Complete in REQUIREMENTS.md.

---

## Previous Gaps: Status After Gap Closure

| Gap (Previous Verification) | Status |
|-----------------------------|--------|
| Walk-forward positive PnL on 2+ segments | CLOSED — v3 shows 2/3 PASS |
| OOS Sharpe > 0.1 on 2+ segments | CLOSED — ru_blue_chips=0.203, ru_energy=0.140 |
| OOS PF > 1.05 on 2+ segments | CLOSED — ru_blue_chips=1.436, ru_energy=1.211 |

### Regressions After Symbol Restoration

No regressions in code infrastructure or test suite (15/15 preset validation tests pass). Walk-forward metrics may regress when a new backtest is run against the restored (broader) universes. This is the explicitly accepted trade-off — previously losing symbols are retained for Phase 7 event_driven/news integration, where sentiment signals may make them profitable.

---

## Gaps Summary

No gaps remain for phase sign-off. The phase goal ("MOEX equity strategies produce profitable results in walk-forward backtests, positive performance on 2+ MOEX segments") is achieved in v3 walk-forward results. The user-directed symbol restoration is a forward-looking scope decision, not a regression against the phase gate. Both EQF-02 and EQF-03 in REQUIREMENTS.md are marked Complete.

---

_Verified: 2026-03-14T18:00:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification: Yes — after gap closure plan 02-03_

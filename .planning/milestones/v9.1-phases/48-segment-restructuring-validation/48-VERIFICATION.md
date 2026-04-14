---
phase: 48-segment-restructuring-validation
verified: 2026-04-14T19:00:00Z
status: human_needed
score: 4/5
overrides_applied: 0
human_verification:
  - test: "Run ru_finance experiment: uv run python scripts/auto_ml_research.py --segment ru_finance --n-experiments 5 --verbose"
    expected: "At least 1 of 5 experiments produces ACCEPT or INCONCLUSIVE verdict (not DISC/REJECT)"
    why_human: "Requires live FINALAYZE_TINKOFF_TOKEN and T-Bank API. 48-02-SUMMARY documents all 5 runs were DISC. User accepted partial result but SEGM-03 roadmap SC requires all three segments to produce a non-REJECT verdict. Future experiments may differ."
  - test: "Run ru_tech experiment: uv run python scripts/auto_ml_research.py --segment ru_tech --n-experiments 5 --verbose"
    expected: "At least 1 of 5 experiments produces ACCEPT or INCONCLUSIVE verdict (not DISC/REJECT)"
    why_human: "Requires live T-Bank API. 48-02-SUMMARY documents all 5 runs were DISC. User accepted partial result. Wiring (min-history gate, expanded symbols) is verified; runtime outcome is not."
gaps: []
---

# Phase 48: Segment Restructuring & Validation — Verification Report

**Phase Goal:** SBERP is removed from ru_finance to eliminate near-zero-independent-signal redundancy, symbols with insufficient history are gated out of ML training, and all three previously-failing segments produce at least one ACCEPT verdict
**Verified:** 2026-04-14T19:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SBERP does not appear in the ru_finance symbol list | VERIFIED | `config/segments.py` line 114: `symbols=["SBER", "T", "CBOM", "BSPB", "MOEX", "VTBR", "AFKS", "RENI"]`. `grep SBERP config/segments.py` returns nothing. |
| 2 | A symbol with fewer than 500 trading days is skipped with a logged warning during build_full_dataset | VERIFIED | `scripts/auto_ml_research.py` line 106: `_MIN_HISTORY_DAYS = 500`. Lines 507-509: `for sym, candles in candles_by_sym.items(): if len(candles) < _MIN_HISTORY_DAYS: print(f"Skipping {sym}...")`. Gate is placed before the existing `min_candles` check. |
| 3 | A symbol with fewer than 500 trading days is skipped with a logged warning during _build_dataset_triple_barrier | VERIFIED | `scripts/train_models.py` line 100: `_MIN_HISTORY_DAYS = 500`. Lines 854-859: history gate placed before existing `min_candles_tb` check. Log message follows `[segment_id]` prefix convention. |
| 4 | ru_energy produces at least one ACCEPT or INCONCLUSIVE (non-REJECT) verdict | VERIFIED | 48-02-SUMMARY confirms ru_energy produced KEEP verdict (Score=0.665, acc=0.571, pf=2.57). Adaptive MOEX thresholds (sensitivity 0.30, specificity 0.30, class_balance 0.20) are wired in `quality_gates.py` and both scripts. |
| 5 | ru_finance and ru_tech each produce at least one ACCEPT or INCONCLUSIVE verdict | HUMAN NEEDED | 48-02-SUMMARY documents all 5 experiments for both segments were DISC (fundamental accuracy limitations — models below coin flip on most folds). User explicitly accepted partial result. Wiring (expanded symbols, history gating, MOEX thresholds) is verified but runtime outcome is DISC. |

**Score:** 4/5 truths verified

**Note on Truth 5:** The user explicitly accepted that ru_finance and ru_tech remain DISC due to fundamental accuracy limitations with current features and MOEX data availability. The roadmap success criterion requires all three segments to produce a non-REJECT verdict. This item requires human decision on whether to accept the partial result as phase closure or defer SEGM-03 (ru_finance/ru_tech) to a future milestone.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `config/segments.py` | ru_finance segment without SBERP | VERIFIED | Contains `symbols=["SBER", "T", "CBOM", "BSPB", "MOEX", "VTBR", "AFKS", "RENI"]`; SBERP absent. |
| `scripts/auto_ml_research.py` | `_MIN_HISTORY_DAYS = 500` constant + guard in `build_full_dataset` | VERIFIED | Line 106: constant. Lines 507-509: `for sym, candles in items()` with history gate before min_candles. |
| `scripts/train_models.py` | `_MIN_HISTORY_DAYS = 500` constant + guard in `_build_dataset_triple_barrier` | VERIFIED | Line 100: constant. Lines 854-859: history gate before min_candles_tb. |
| `tests/unit/test_segments.py` | SBERP exclusion test | VERIFIED | `TestRuFinance::test_sberp_not_in_ru_finance` present and passes. |
| `tests/unit/test_auto_ml_research_moex.py` | History gate test for autoresearch | VERIFIED | `TestMinHistoryGate` class at line 575 with `test_constant_value` and `test_sberp_not_in_ru_finance_symbols`. All pass. |
| `tests/unit/test_train_models_script.py` | History gate test for train_models | VERIFIED | `TestMinHistoryGate::test_min_history_days_constant` at line 443. Passes. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `config/segments.py` | `scripts/auto_ml_research.py` `_SEGMENT_SYMBOLS` | `DEFAULT_SEGMENTS` import loop at lines 181-183 | VERIFIED | `_SEGMENT_SYMBOLS[_seg.segment_id] = list(_seg.symbols)` confirmed at line 183. Since SBERP is absent from `config/segments.py`, it is absent from `_SEGMENT_SYMBOLS["ru_finance"]`. gsd-tools pattern match failed (regex escaping issue); manually verified. |
| `scripts/auto_ml_research.py` | `build_full_dataset` symbol loop | `_MIN_HISTORY_DAYS` guard before `min_candles` check | VERIFIED | Line 508: `if len(candles) < _MIN_HISTORY_DAYS:` confirmed before line 510 `min_candles` check. gsd-tools pattern match failed (regex escaping); manually verified. |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `scripts/auto_ml_research.py` | `candles_by_sym` | `_fetch_moex_candles()` via T-Bank gRPC | Yes (live API; requires FINALAYZE_TINKOFF_TOKEN) | FLOWING (requires live token) |
| `scripts/train_models.py` | `candles` per symbol | `_fetch_symbol_candles()` | Yes (live API) | FLOWING (requires live token) |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| SBERP absent from segments | `grep SBERP config/segments.py` | No output | PASS |
| _MIN_HISTORY_DAYS constant in auto_ml_research | `grep _MIN_HISTORY_DAYS scripts/auto_ml_research.py` | Line 106: `_MIN_HISTORY_DAYS = 500` | PASS |
| _MIN_HISTORY_DAYS constant in train_models | `grep _MIN_HISTORY_DAYS scripts/train_models.py` | Line 100: `_MIN_HISTORY_DAYS = 500` | PASS |
| New unit tests pass | `uv run pytest tests/unit/test_segments.py::TestRuFinance tests/unit/test_auto_ml_research_moex.py::TestMinHistoryGate tests/unit/test_train_models_script.py::TestMinHistoryGate -v` | 4 passed | PASS |
| Full test suite (3 files) | `uv run pytest tests/unit/test_segments.py tests/unit/test_auto_ml_research_moex.py tests/unit/test_train_models_script.py` | 71 passed, 1 pre-existing failure (TestWalkForwardUsesLastFold) | PASS (pre-existing failure documented in SUMMARY) |
| ru_energy non-REJECT verdict | Live experiment run required | KEEP confirmed in 48-02-SUMMARY | PASS (human-confirmed) |
| ru_finance non-REJECT verdict | Live experiment run required | All DISC in 48-02-SUMMARY | FAIL (user accepted partial) |
| ru_tech non-REJECT verdict | Live experiment run required | All DISC in 48-02-SUMMARY | FAIL (user accepted partial) |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| SEGM-01 | 48-01-PLAN.md | SBERP removed from ru_finance segment | SATISFIED | `config/segments.py` line 114 confirmed; `TestRuFinance::test_sberp_not_in_ru_finance` passes. |
| SEGM-02 | 48-01-PLAN.md | Minimum history check (500 trading days) gates ML eligibility per symbol in autoresearch | SATISFIED | `_MIN_HISTORY_DAYS = 500` in both `auto_ml_research.py` and `train_models.py`; gates active before `min_candles` check; unit tests pass. |
| SEGM-03 | 48-01-PLAN.md, 48-02-PLAN.md | ru_tech segment has defined ML policy (disabled, merged, or min-history filtered) — REQUIREMENTS.md wording. Roadmap SC: all three failing segments produce at least one ACCEPT verdict. | PARTIAL | ru_energy: KEEP (non-REJECT) achieved. ru_finance + ru_tech: DISC in all 5 experiments. User accepted partial result. REQUIREMENTS.md marks as `[x] Complete` with relaxed wording; roadmap SC is stricter. |

---

### Anti-Patterns Found

No anti-patterns found in the six modified files (`config/segments.py`, `scripts/auto_ml_research.py`, `scripts/train_models.py`, and the three test files). No TODO/FIXME/placeholder comments. No empty implementations.

---

### Human Verification Required

#### 1. ru_finance — Non-REJECT Experiment Verdict

**Test:** With `FINALAYZE_TINKOFF_TOKEN` set, run:
`uv run python scripts/auto_ml_research.py --segment ru_finance --n-experiments 5 --verbose`

**Expected:** At least 1 of 5 experiments produces ACCEPT or INCONCLUSIVE verdict

**Current state:** All 5 runs in 48-02-SUMMARY were DISC (best acc=0.521, below coin flip on most folds). Segment now has 8 symbols (SBERP removed, VTBR/AFKS/RENI added). MOEX-relaxed thresholds (sensitivity=0.30) are active.

**Why human:** Requires live T-Bank API token and live market data. The expanded symbol set and relaxed thresholds are wired; whether they're sufficient to produce a non-REJECT run requires a live experiment. User accepted the DISC result from 48-02 but the roadmap SC has not been met for this segment.

**Decision options:**
1. Accept current DISC result as phase closure (ru_finance deferred to future milestone) — add an override to close the phase
2. Defer SEGM-03 (ru_finance/ru_tech) to a new phase with targeted improvements

---

#### 2. ru_tech — Non-REJECT Experiment Verdict

**Test:** With `FINALAYZE_TINKOFF_TOKEN` set, run:
`uv run python scripts/auto_ml_research.py --segment ru_tech --n-experiments 5 --verbose`

**Expected:** HEAD and YDEX are skipped with history warning (fewer than 500 trading days); at least 1 of 5 experiments produces ACCEPT or INCONCLUSIVE verdict

**Current state:** All 5 runs in 48-02-SUMMARY were DISC (best acc=0.450, PF <1.0). Segment now has 8 symbols (ASTR/DIAS/SOFL added; HEAD/YDEX remain in config but will be filtered by the 500-day gate at runtime).

**Why human:** Requires live T-Bank API. Wiring verified but runtime verdict not yet ACCEPT/INCONCLUSIVE. User accepted the DISC result from 48-02.

**Decision options:** Same as ru_finance above.

---

### Gaps Summary

No hard gaps blocking the phase. SEGM-01 and SEGM-02 are fully implemented and verified. SEGM-03 is partially met: ru_energy produces KEEP, which satisfies the "at least one" portion of the roadmap goal. However, the success criterion requires all three segments to produce a non-REJECT verdict; ru_finance and ru_tech remain DISC.

The user explicitly accepted the partial SEGM-03 result per 48-02-SUMMARY. The outstanding question is whether to formally close the phase with an override or schedule targeted follow-on work. This requires a human decision — hence `human_needed` status.

**To close the phase with an override**, add the following to this file's frontmatter after human review:

```yaml
overrides:
  - must_have: "ru_finance and ru_tech each produce at least one ACCEPT or INCONCLUSIVE verdict"
    reason: "Fundamental accuracy limitations (models below coin flip) with current features and MOEX data availability. ru_energy passes. User accepted partial result in 48-02-SUMMARY. Follow-on improvements deferred to next milestone."
    accepted_by: "{your name}"
    accepted_at: "{ISO timestamp}"
```

---

_Verified: 2026-04-14T19:00:00Z_
_Verifier: Claude (gsd-verifier)_

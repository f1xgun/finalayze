---
phase: 41-adaptive-quality-gates
verified: 2026-04-13T12:00:00Z
status: passed
score: 3/3
overrides_applied: 0
---

# Phase 41: Adaptive Quality Gates — Verification Report

**Phase Goal:** MOEX experiments produce trustworthy walk-forward results — signal count gates are calibrated to MOEX dataset sizes, folds never collapse to fewer than 3, and degenerate all-BUY/all-SELL models are rejected automatically
**Verified:** 2026-04-13T12:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `evaluate_fold(min_signals=15)` accepts a MOEX experiment with 15-30 signals per fold — the hardcoded _MIN_SIGNALS=50 no longer blocks all MOEX runs | VERIFIED | `check_signal_count_gate` and `evaluate_fold` both accept keyword-only `min_signals: int = _MIN_SIGNALS` (default 50). Callers may pass `min_signals=15`. Tests `TestSignalCountGateWithMinSignals` and `TestEvaluateFoldWithMinSignals` confirm the behavior. All 39 tests pass. |
| 2 | A 730-day MOEX dataset produces 3 or more valid walk-forward folds using MOEX-specific fold constants — the experiment does not trivially pass on a single fold | VERIFIED | Six MOEX constants defined in `scripts/auto_ml_research.py` (`_MOEX_WF_TRAIN_MONTHS=8`, `_MOEX_WF_CAL_MONTHS=1`, `_MOEX_WF_TEST_MONTHS=3`, `_MOEX_WF_STEP_MONTHS=2`, `_MOEX_PURGE_GAP=21`, `_MOEX_MIN_SIGNALS=15`). `generate_folds()` accepts keyword-only fold parameters with US defaults. `test_moex_folds_730_days` passes: 730-day dataset + MOEX constants → ≥3 folds. Main loop branches on `is_moex` to use MOEX constants. |
| 3 | A model that predicts BUY on 92% of samples fails the degenerate predictor gate and is logged as REJECTED with buy_ratio=0.92 — all-directional models cannot receive a verdict without this check | VERIFIED | `check_degenerate_predictor_gate(FoldMetrics(buy_ratio=0.92))` returns `passed=False, gate_name="degenerate_predictor", detail="buy_ratio=0.92, bounds=[0.15, 0.85]"`. The gate is included in `evaluate_fold` return (8th gate). `gate_pass_rates["degenerate_predictor"]=0.0` is logged to JSONL as part of experiment record. Status is set to "discard" (system uses "DISC"/"discard" rather than "REJECTED" as a label, but the rejection is functional). All 7 boundary tests pass. |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/ml/training/quality_gates.py` | `check_degenerate_predictor_gate()` + `min_signals` param on `evaluate_fold()` | VERIFIED | Both functions exist and are substantive. `_DEGEN_MIN_BUY_RATIO=0.15`, `_DEGEN_MAX_BUY_RATIO=0.85` constants present. `evaluate_fold` returns 8 gates including `degenerate_predictor`. |
| `tests/unit/test_quality_gates.py` | Tests for degenerate gate and min_signals parameterization | VERIFIED | Contains `TestDegeneratePredictorGate` (7 methods), `TestSignalCountGateWithMinSignals` (3 methods), `TestEvaluateFoldWithMinSignals` (2 methods). `_EIGHT_GATES=8` constant present. All 39 tests pass. |
| `scripts/auto_ml_research.py` | MOEX fold constants and min_signals wiring | VERIFIED | All 6 MOEX constants present. `generate_folds()` parametrized with keyword-only fold params. `_run_fold()` accepts and passes `min_signals`. `run_experiment()` selects `min_signals` via `_is_moex_segment()`. Main loop uses `fold_kwargs` for MOEX branch. |
| `tests/unit/test_auto_ml_research_folds.py` | Tests for MOEX fold generation | VERIFIED | File exists. Contains `TestMoexFoldGeneration` class with 5 tests including `test_moex_folds_730_days` and `_MIN_MOEX_FOLDS=3`. All 5 tests pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/finalayze/ml/training/quality_gates.py` | `evaluate_fold` | `check_degenerate_predictor_gate` called inside `evaluate_fold` | WIRED | Line 228: `check_degenerate_predictor_gate(metrics)` is the 8th element in `evaluate_fold` return list. |
| `scripts/auto_ml_research.py` | `src/finalayze/ml/training/quality_gates.py` | `evaluate_fold(fold_metrics, min_signals=min_signals)` | WIRED | Line 672: `gate_results = evaluate_fold(fold_metrics, min_signals=min_signals)` in `_run_fold`. Line 698: `min_signals = _MOEX_MIN_SIGNALS if _is_moex_segment(segment_id) else _US_MIN_SIGNALS` in `run_experiment`. |

### Data-Flow Trace (Level 4)

Not applicable — these are pure computational gate functions, not rendering components. Data flows through `FoldMetrics` dataclass fields populated by `_evaluate_models()`, passed to `check_degenerate_predictor_gate()` and `check_signal_count_gate()`, results collected by `run_experiment()` and logged to JSONL.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| buy_ratio=0.92 fails degenerate gate with correct detail | `uv run python3 -c "from finalayze.ml.training.quality_gates import FoldMetrics, check_degenerate_predictor_gate; m=FoldMetrics(accuracy=0.55, brier_score=0.2, log_loss=0.5, n_test=30, buy_ratio=0.92, signal_count=20); r=check_degenerate_predictor_gate(m); print(r.passed, r.gate_name, r.detail)"` | `False degenerate_predictor buy_ratio=0.92, bounds=[0.15, 0.85]` | PASS |
| 730-day MOEX data produces 3+ folds | `uv run pytest tests/unit/test_auto_ml_research_folds.py::TestMoexFoldGeneration::test_moex_folds_730_days --no-cov` | 1 passed | PASS |
| All quality gate tests pass | `uv run pytest tests/unit/test_quality_gates.py tests/unit/test_auto_ml_research_folds.py --no-cov` | 39 passed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| GATE-01 | 41-01-PLAN.md | `evaluate_fold()` accepts `min_signals` parameter — MOEX experiments use n_eff-scaled threshold instead of hardcoded 50 | SATISFIED | `check_signal_count_gate` and `evaluate_fold` both accept `min_signals` keyword-only param with default 50. Tests confirm parameterization. |
| GATE-02 | 41-02-PLAN.md | MOEX-specific walk-forward fold constants produce 3+ folds on 730-day dataset — no single-fold trivial pass | SATISFIED | 6 MOEX constants defined. `generate_folds()` parametrized. Test `test_moex_folds_730_days` passes. 730-day + US constants → 0 folds (confirmed by `test_us_constants_on_moex_data_few_folds`). |
| GATE-03 | 41-01-PLAN.md | Degenerate predictor guard rejects all-BUY/all-SELL models (buy_ratio outside 0.15–0.85 range fails gate) | SATISFIED | `check_degenerate_predictor_gate` implemented with `[0.15, 0.85]` inclusive bounds. Wired into `evaluate_fold` as 8th gate. 7 boundary tests pass. |

### Anti-Patterns Found

None. No TODOs, FIXMEs, placeholders, empty returns, or hardcoded empty data found in modified files (`quality_gates.py`, `auto_ml_research.py`, `test_quality_gates.py`, `test_auto_ml_research_folds.py`).

### Human Verification Required

None — all success criteria can be verified programmatically and were confirmed via test execution and direct function invocation.

### Gaps Summary

No gaps. All three roadmap success criteria are satisfied:

1. GATE-01: `evaluate_fold` and `check_signal_count_gate` accept `min_signals` keyword-only parameter, defaulting to 50 for backward compatibility. MOEX callers pass 15.
2. GATE-02: MOEX fold constants (8/1/3/21/2) are defined and wired through `generate_folds()` via the main loop MOEX branch. Test confirms 730-day MOEX dataset produces ≥3 folds.
3. GATE-03: `check_degenerate_predictor_gate` rejects buy_ratio outside [0.15, 0.85], gate_name="degenerate_predictor", detail includes "buy_ratio=X.XX". Gate is the 8th element in `evaluate_fold`. All directional models are logged to JSONL with `gate_pass_rates["degenerate_predictor"]=0.0` and `status="discard"`.

---

_Verified: 2026-04-13T12:00:00Z_
_Verifier: Claude (gsd-verifier)_

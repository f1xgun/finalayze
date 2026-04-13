---
phase: 41-adaptive-quality-gates
plan: "01"
subsystem: ml/training
tags: [quality-gates, ml, tdd, moex-adaptation]
dependency_graph:
  requires: []
  provides:
    - check_degenerate_predictor_gate (new gate function)
    - check_signal_count_gate with min_signals param (extended API)
    - evaluate_fold with min_signals param (extended API)
  affects:
    - scripts/auto_ml_research.py (caller of evaluate_fold)
    - scripts/train_models.py (caller of evaluate_fold)
tech_stack:
  added: []
  patterns:
    - keyword-only parameters with defaults for backward-compatible API extension
    - boundary-inclusive range check for degenerate predictor detection
key_files:
  created: []
  modified:
    - src/finalayze/ml/training/quality_gates.py
    - tests/unit/test_quality_gates.py
decisions:
  - "check_degenerate_predictor_gate uses [0.15, 0.85] inclusive bounds; complements existing class_balance gate (which uses min(ratio,1-ratio) >= 0.30) with explicit naming for diagnostics"
  - "min_signals uses keyword-only param (* separator) to prevent positional call mistakes at callsites"
  - "detail string split across two f-string literals to satisfy ruff E501 line-length=100"
metrics:
  duration: "~5 minutes"
  completed: "2026-04-13"
  tasks_completed: 2
  files_modified: 2
---

# Phase 41 Plan 01: Adaptive Quality Gates — min_signals + degenerate predictor Summary

Parameterized signal count threshold + degenerate predictor safety gate for MOEX-sized ML folds.

## What Was Built

`quality_gates.py` gained two improvements to support MOEX ML training:

1. **`check_signal_count_gate(metrics, *, min_signals=50)`** — The hardcoded `_MIN_SIGNALS=50` constant is now the default for a keyword-only parameter. MOEX callers can pass `min_signals=15` to permit smaller fold signal counts without breaking existing US-segment callers (backward compatible).

2. **`check_degenerate_predictor_gate(metrics)`** — New 8th gate that rejects models with `buy_ratio` outside `[0.15, 0.85]`. Returns `gate_name="degenerate_predictor"` with a diagnostic `detail` string. Guards against relaxed `min_signals` thresholds producing all-BUY or all-SELL degenerate models on tiny MOEX datasets.

3. **`evaluate_fold(metrics, *, min_signals=50)`** — Accepts `min_signals` and forwards it to the signal count gate. Now returns 8 results (was 7), always including `degenerate_predictor`.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 (RED) | Failing tests for min_signals + degenerate gate | 355682b |
| 2 (GREEN) | Implementation in quality_gates.py + lint fix | bc85837 |

## Test Coverage

34 tests pass (was 25). New tests:

- `TestSignalCountGateWithMinSignals` — 3 tests: custom pass, custom fail, default=50 unchanged
- `TestDegeneratePredictorGate` — 7 tests: all-buy, all-sell, balanced, boundary-low, boundary-high, just-outside-high, just-outside-low
- `TestEvaluateFoldWithMinSignals` — 2 tests: min_signals passthrough, default=50

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ruff E501 line too long in detail f-string**
- **Found during:** Task 2 GREEN verification (`ruff check`)
- **Issue:** `detail=f"buy_ratio={metrics.buy_ratio:.2f}, bounds=[{_DEGEN_MIN_BUY_RATIO}, {_DEGEN_MAX_BUY_RATIO}]"` was 109 chars (limit 100)
- **Fix:** Split into two concatenated f-string literals inside parentheses
- **Files modified:** `src/finalayze/ml/training/quality_gates.py`
- **Commit:** bc85837 (included in GREEN commit)

## Known Stubs

None — all data flows through real FoldMetrics fields.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundary crossings. API is internal-only (callers are trusted scripts).

## Self-Check: PASSED

- [x] `src/finalayze/ml/training/quality_gates.py` — exists and modified
- [x] `tests/unit/test_quality_gates.py` — exists and modified
- [x] Commit 355682b — RED phase tests
- [x] Commit bc85837 — GREEN implementation
- [x] `uv run pytest tests/unit/test_quality_gates.py` — 34 passed, 0 failed
- [x] `uv run ruff check src/finalayze/ml/training/quality_gates.py` — passed
- [x] `uv run mypy src/finalayze/ml/training/quality_gates.py` — passed

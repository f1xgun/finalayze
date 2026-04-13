# Phase 41: Adaptive Quality Gates - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Parametrize quality gates for MOEX dataset sizes — adaptive min_signals, MOEX-specific walk-forward fold constants, and degenerate predictor guard. Existing US behavior unchanged.

</domain>

<decisions>
## Implementation Decisions

### Signal Count & Fold Constants
- Add `min_signals: int = _MIN_SIGNALS` parameter to `evaluate_fold()` — zero behavior change for existing US callers
- MOEX fold constants: 8mo train / 1mo cal / 3mo test / 21-day purge / 2mo step — yields ~3-4 folds on 730-day dataset
- Define MOEX fold constants in `auto_ml_research.py` as `_MOEX_WF_*` constants alongside existing `_WF_*` US constants
- Default min_signals for MOEX: 15 (n_eff-scaled: ~30% of US _MIN_SIGNALS=50 for ~30% data size)

### Degenerate Predictor Guard
- New `check_degenerate_predictor_gate()` in `quality_gates.py` — fits existing gate pattern
- Buy ratio bounds: 0.15–0.85 — reject models predicting outside this range
- Integrate into `evaluate_fold()` gate list — evaluated alongside existing gates automatically
- On failure: gate returns False, fold logged as REJECTED in gate_pass_rates — same pattern as other gates

### Claude's Discretion
- Internal implementation details of gate functions
- Test fixture construction for degenerate predictor tests
- Whether to also add a sensitivity/specificity floor gate (not required, but natural extension)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `quality_gates.py` — `evaluate_fold()`, `evaluate_walk_forward()`, `FoldMetrics`, `check_accuracy_gate()`, `check_brier_gate()`, `check_signal_count_gate()`, etc.
- `auto_ml_research.py` — `generate_folds()` with US constants, `_evaluate_models()` returns `FoldMetrics`
- Existing adaptive gates: `check_accuracy_gate()` caps at 0.55 for n_eff<20, `_dynamic_brier_threshold()` relaxes for small n_eff

### Established Patterns
- Gate functions: `check_*_gate(metrics: FoldMetrics, ...) -> bool`
- `evaluate_fold(metrics)` calls all gates, returns list of (gate_name, passed) tuples
- `evaluate_walk_forward(all_fold_results)` computes per-gate pass rates

### Integration Points
- `quality_gates.py:evaluate_fold()` — add min_signals param + degenerate gate call
- `auto_ml_research.py:generate_folds()` — branch on MOEX vs US fold constants
- `auto_ml_research.py:_run_fold()` — pass min_signals to evaluate_fold()

</code_context>

<specifics>
## Specific Ideas

- Research confirmed: _MIN_SIGNALS=50 is the only non-adaptive gate blocking MOEX experiments
- Accuracy and Brier gates already have n_eff scaling — no changes needed there
- Degenerate predictor guard is the safety net for relaxed gates on small datasets

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

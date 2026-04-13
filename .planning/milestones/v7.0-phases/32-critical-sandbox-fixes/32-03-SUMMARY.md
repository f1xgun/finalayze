---
phase: 32-critical-sandbox-fixes
plan: 03
status: complete
started: 2026-04-07T14:00:00Z
completed: 2026-04-07T17:00:00Z
---

## Summary

Fixed two ML quality gate bugs that prevented models from ever passing validation.

### What was built

1. **Actual profit_factor computation**: `_evaluate_fold_metrics` now computes profit factor from fold predictions (gross_profit / gross_loss with 0.55 BUY threshold) instead of always using the default value 1.0.
2. **Calibrated Brier parameter**: Added optional `calibrator` parameter to `_evaluate_fold_metrics`. When provided, Brier score is computed from calibrated probabilities via `calibrator.predict_proba()`. Falls back to raw probabilities on failure.

### Key files

- `scripts/train_models.py` — profit_factor computation, calibrator parameter
- `src/finalayze/ml/training/quality_gates.py` — FoldMetrics dataclass (unchanged, but now receives real values)
- `tests/unit/test_ml_quality_gates.py` — 7 new tests for profit_factor and Brier fixes

### Commits

- `c7a56a8` — test(32-03): add failing tests for profit_factor and calibrated Brier
- `a8d861e` — fix(32-03): compute actual profit_factor + calibrated Brier in ML gates

### Deviations

- Calibrator parameter added but not wired in walk-forward loop yet (per-fold calibrator doesn't exist in current loop; parameter ready for future use).

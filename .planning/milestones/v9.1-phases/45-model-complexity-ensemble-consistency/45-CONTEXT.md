# Phase 45: Model Complexity & Ensemble Consistency - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase reduces MOEX model complexity to prevent overfitting on ~850 train samples and fixes the XGBoost/LightGBM class rebalancing inconsistency. Pure infrastructure — config changes + model param fixes in autoresearch pipeline and model wrappers.

</domain>

<decisions>
## Implementation Decisions

### Model Complexity
- Add `_MOEX_HPARAMS` dict separate from `_DEFAULT_HPARAMS` with: xgb_max_depth=3, xgb_n_estimators=100, lgbm_n_estimators=100, lgbm_num_leaves=15, cat_depth=3, cat_iterations=100
- Route to MOEX hparams when `_is_moex_segment(segment_id)` is True — same pattern as existing `_get_max_features()`
- `min_child_weight=20` added to MOEX XGBoost config (prevents leaf nodes with only 5 samples)
- Log active complexity profile at run start: "Using MOEX complexity profile: depth=3, est=100"

### Ensemble Consistency
- XGBoost `xgboost_model.py:95` must check if `sample_weight is not None` and set `spw=1.0` in that case — matching LightGBM behavior at `lightgbm_model.py:97`
- CatBoost: verify its `auto_class_weights` parameter interaction with sample_weight — if it applies both, disable auto_class_weights when sample_weight provided

### Claude's Discretion
- Exact log message format
- Whether to add `min_child_weight` to LightGBM and CatBoost equivalents (min_data_in_leaf, min_data_in_leaf)
- Test structure and naming

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_is_moex_segment(segment_id)` — already exists for routing MOEX-specific params
- `_get_max_features(segment_id)` — pattern for MOEX/US config routing
- `_DEFAULT_HPARAMS` dict at `auto_ml_research.py:183` — current US defaults
- `ExperimentConfig.hparams` field with default_factory from `_DEFAULT_HPARAMS`

### Established Patterns
- MOEX vs US config split via `_MOEX_*` / `_US_*` constants (lookback, min_signals, fold params)
- Model wrappers at `src/finalayze/ml/models/` — XGBoost, LightGBM, CatBoost each have `fit()` with `sample_weight` param

### Integration Points
- `_run_fold()` at line 660-667 — reads `hp.get("xgb_max_depth", 5)` and `hp.get("cat_depth", 4)`
- `run_experiment()` at line 700 — creates ExperimentConfig with hparams
- `XGBoostModel.__init__()` accepts `max_depth` param
- `CatBoostModel.__init__()` accepts `depth` param
- LightGBM uses `num_leaves` not `max_depth` (lgbm_model.py)

</code_context>

<specifics>
## Specific Ideas

- XGBoost scale_pos_weight fix is at `src/finalayze/ml/models/xgboost_model.py:95`: `spw = n_neg / n_pos if n_pos > 0 else 1.0` — needs `if sample_weight is not None` guard
- LightGBM already correct at `lightgbm_model.py:97`: `spw = 1.0 if sample_weight is not None else ...`
- CatBoost model at `catboost_model.py` — check `auto_class_weights` param

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 45-model-complexity-ensemble-consistency*
*Context gathered: 2026-04-14 via autonomous smart discuss*

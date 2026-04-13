# Phase 43: Ensemble Weight Optimization - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Add `ensemble_weights` search strategy to auto_ml_research.py — explores XGB/LGBM/CatBoost weight simplex with bounded grid, overfitting guard, and standard generator pattern.

</domain>

<decisions>
## Implementation Decisions

### Ensemble Weight Strategy Design
- Simplex grid with step 0.1 (all triples summing to 1.0): ~12 configs after filtering
- Max single model weight: 0.7 cap enforced at generation time (filter grid points)
- Small fold guard: <4 folds → skip optimization, use equal weights (1/3 each) with logged warning
- `generate_ensemble_weight_experiments()` returns `list[ExperimentConfig]` — same pattern as ablation/efficiency/hyperparameter/random_subset
- Wire into `_generate_experiments()` under `strategy in ("ensemble_weights", "all")`
- Add `"ensemble_weights"` to `--strategy` CLI choices

### Claude's Discretion
- How to pass weights to _evaluate_models() — likely via ExperimentConfig.hparams dict with xgb_weight/lgbm_weight/cat_weight keys
- Whether _evaluate_models() already supports weighted averaging or needs modification
- Test fixture details

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_evaluate_models()` in auto_ml_research.py — currently uses equal average: `sum(probs) / len(probs)`
- `ExperimentConfig` dataclass with `hparams` dict — can carry weight parameters
- `_generate_experiments()` dispatcher — already handles strategy routing

### Integration Points
- `auto_ml_research.py:_evaluate_models()` — needs weighted averaging support
- `auto_ml_research.py:_generate_experiments()` — add ensemble_weights strategy
- `auto_ml_research.py:main()` — add "ensemble_weights" to choices

</code_context>

<specifics>
## Specific Ideas

- Weight keys in hparams: `xgb_weight`, `lgbm_weight`, `cat_weight`
- _evaluate_models needs to read weights from somewhere — could accept weights param or read from config
- Simplex generation: iterate i in range(0, 11), j in range(0, 11-i), k = 10-i-j; filter max > 7

</specifics>

<deferred>
## Deferred Ideas

None

</deferred>

# Phase 46: Feature Selection Stability - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Refactor feature selection in autoresearch from per-fold MI selection to once-before-folds. The selected features must be identical across all walk-forward folds for a given experiment.

</domain>

<decisions>
## Implementation Decisions

### Feature Selection Location
- Move `select_features_efficient()` call OUT of `_run_fold()` and INTO `run_experiment()` before the fold loop
- Selected features stored as `selected_features` list, passed to each fold via `config.feature_subset`
- When `config.feature_subset is not None` (explicit subset from experiment), skip selection entirely (existing behavior preserved)
- When `config.feature_subset is None` (baseline/default), run selection once on ALL training data (pre-first-fold), then pass as feature_subset to each fold

### Training Data for Selection
- Use all features + labels from the FULL dataset (before fold splitting) for MI calculation
- Exclude test indices from selection dataset to prevent look-ahead bias
- Specifically: union of all train_idx across folds, or simply all indices minus the last test fold

### Claude's Discretion
- Exact implementation of the "all training data" aggregation
- Whether to cache selected features across experiments with same max_features
- Log format for "Selected N features" message

</decisions>

<code_context>
## Existing Code Insights

### Current Flow (to be changed)
- `_run_fold()` at line 664-671: calls `select_features_efficient(train_df, train_s, max_features)` per fold
- Each fold gets different features because MI estimation is noisy on ~850 samples
- This causes fold-to-fold instability: model sees different feature spaces

### Reusable Assets
- `select_features_efficient()` in `training/feature_selection.py` — MI-based selection, returns list of feature names
- `config.feature_subset` field — already exists on ExperimentConfig, used by ablation/random strategies
- `config.max_features` — controls how many features to select

### Integration Points
- `run_experiment()` at line 700 — orchestrates fold loop, has access to all_features and labels
- `_run_fold()` at line 623 — receives config with hparams, needs to use pre-selected features
- All experiment generators that set `feature_subset=None` rely on per-fold selection — these will get once-selection automatically

</code_context>

<specifics>
## Specific Ideas

- A single "Selected N features: [list]" log line per experiment run, not per fold
- Feature list should be deterministic given the same data and max_features

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 46-feature-selection-stability*
*Context gathered: 2026-04-14 via autonomous smart discuss*

# Phase 44: New Search Strategies - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Add two new search strategies to auto_ml_research.py: cross-segment transfer (US→MOEX feature validation) and domain-motivated feature engineering with hard overfitting caps.

</domain>

<decisions>
## Implementation Decisions

### Cross-Segment Transfer
- Read US experiment JSONL log, find entry with highest score and status="keep" as source
- Market-neutral feature filtering via keyword exclusion: exclude features containing "vix", "usdrub", "brent", "cbr", "imoex", "turnover" — these are market-specific
- If no US experiment JSONL exists, skip cross-segment transfer with warning
- `generate_transfer_experiments()` returns `list[ExperimentConfig]` with filtered feature_subset

### Feature Engineering Strategy
- Domain-motivated combinations only: lag ratios (close_t / close_t-5), rolling z-scores (20d, 60d), cross-feature interactions (RSI × volume_ratio)
- Candidate cap: `n_samples / 20` (~36 for 730-day MOEX, ~91 for 1825-day US)
- Permutation importance filter: train once, run permutation importance, discard features below baseline
- Base features for combinations: use baseline experiment's `features_used` list, not all 45+ features

### Claude's Discretion
- Specific lag periods and rolling windows for feature engineering
- Permutation importance implementation details (sklearn or manual)
- How to combine both strategies under "all" CLI mode
- Test fixture design for both strategies

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_RESULTS_DIR` / `_log_result()` — JSONL log reading for cross-segment
- `ExperimentConfig` with `feature_subset` field — pass filtered features
- `select_features_efficient()` — existing feature selection (MI-based)
- `sklearn.inspection.permutation_importance` — available in deps

### Integration Points
- `auto_ml_research.py:_generate_experiments()` — add cross_segment_transfer + feature_engineering
- `auto_ml_research.py:main()` — add both to CLI choices
- US experiment JSONL at `results/experiments/us_tech_experiment_log.jsonl` (if exists)

</code_context>

<specifics>
## Specific Ideas

- User is MOEX-focused — cross-segment transfer validates if US-learned features generalize to MOEX
- Feature engineering is highest overfitting risk — cap + permutation filter are critical safety nets
- Both strategies wire into existing `_generate_experiments()` dispatcher pattern

</specifics>

<deferred>
## Deferred Ideas

None

</deferred>

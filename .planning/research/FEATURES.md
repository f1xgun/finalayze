# Feature Landscape

**Domain:** ML AutoResearch & MOEX Adaptation
**Researched:** 2026-04-13

## Context

This milestone (v9.0) extends the existing `auto_ml_research.py` — which already runs 4 search strategies (ablation, efficiency, hyperparameter, random_subset) on US segments only — to support MOEX segments via TinkoffFetcher, and adds 3 new search strategies (ensemble_weights, feature_engineering, cross_segment_transfer). The ExperimentManager (markdown+YAML frontmatter, CRUD, verdict lifecycle) already exists from v8.0 and must be integrated as the persistence backend.

### Key Existing Constraints

- `auto_ml_research.py` is a standalone script; its data path is hardcoded to `YFinanceFetcher`. No MOEX segment IDs appear in `_SEGMENT_SYMBOLS`.
- `train_models.py` already has MOEX-aware helpers: `_is_moex_segment()`, `_get_lookback_days()` (730 days), `_get_max_features()` (10 vs 15), `_get_xgboost_max_depth()` (3 vs 5), and uses `TinkoffFetcher` for `ru_*` segments. These patterns must be replicated — not reinvented — in `auto_ml_research.py`.
- Quality gates already have partial MOEX accommodation: accuracy gate caps at 0.55 for n_eff < 20, and Brier gate uses dynamic thresholds via `_dynamic_brier_threshold(n_eff)`. The signal_count gate (_MIN_SIGNALS = 50) is the primary blocker for MOEX: MOEX walk-forward folds with 730 days of history produce far fewer than 50 signals per fold.
- 10 Russian macro features already exist in `ml/features/technical.py` (USDRUB z-score, Brent z-score, CBR key rate, IMOEX turnover, etc.) behind `MoexMarketData`. These features are wired in `train_models.py` but not yet fetched inside `auto_ml_research.py`.
- `EnsembleModel` accepts `model_weights: dict[str, float]` and has a `_get_model_weight()` resolver. Equal weighting is the default when no weights dict is provided.
- `ExperimentState` + `ExperimentManager` are in Layer 0 / Layer 0; they persist experiments as markdown files in `.planning/experiments/`. Integration means creating an ExperimentState at loop start and recording results/verdicts at loop end.

---

## Table Stakes

Features users expect. Missing = autoresearch pipeline feels broken for MOEX.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| MOEX data adapter (TinkoffFetcher path) | `auto_ml_research.py` is US-only today; no MOEX segments work | Medium | Must mirror `train_models.py` MOEX branch: TinkoffFetcher for candles, MOEX ISS for IMOEX index, CBR XML for key rate + FX, yfinance for Brent (BZ=F). Async wrappers exist in TinkoffFetcher; the fetch must be driven via `_run_async()` from sync context like the fetcher already does. |
| MOEX macro features in autoresearch pipeline | Already computed in `train_models.py` for `ru_*`; absence here is a gap | Medium | `MoexMarketData` schema exists. Need to build it in `auto_ml_research.py` then pass as `market_context.moex_data` to `build_triple_barrier_dataset()`. The 10 features (usdrub_zscore_60d, brent_zscore_60d, cbr_key_rate, imoex_index, etc.) are already implemented in `technical.py`; the gap is the fetch-and-wire step. |
| Adaptive quality gates for small MOEX datasets | MOEX 730-day history yields ~2 walk-forward folds with current WF params; signal_count gate (min 50) fails almost universally | Medium | Accuracy and Brier gates are already partially adaptive. Critical gap: signal_count gate uses a fixed threshold of 50 that does not scale with dataset size. Need n_eff-scaled minimum: `max(10, int(50 * n_eff / 100))` or similar. Also need MOEX-specific WF params: shorter train (6 mo), shorter test (2 mo), shorter step (1.5 mo) to generate more folds from 730 days. |
| ExperimentManager integration | v8.0 built the entire experiment lifecycle (create, run, verdict, debate linkage); autoresearch currently writes raw JSONL with no connection to this | Medium | At loop start: `manager.create(experiment_id, hypothesis, success_criteria)`. After each experiment: `manager.record_result(...)`. At loop end: `manager.compute_verdict(...)`. The JSONL log can remain as a parallel append-only audit trail. |
| MOEX segment symbols in _SEGMENT_SYMBOLS | Current dict only has US segments; trying `--segment ru_blue_chips` crashes immediately | Low | Copy from `train_models.py`: ru_blue_chips, ru_energy, ru_tech, ru_finance. FIGI resolution already handled inside TinkoffFetcher (`_symbol_to_figi()`). |
| MOEX-tuned default hyperparameters | XGBoost max_depth should default to 3 (not 5), CatBoost depth to 3 (not 4) for MOEX to prevent overfitting on small datasets | Low | Pattern already in `train_models.py`. `ExperimentConfig.hparams` needs a `_MOEX_DEFAULT_HPARAMS` variant and the loader should pick based on `_is_moex_segment()`. |

---

## Differentiators

Features that set the ML autoresearch apart and add real value. Not expected from table stakes alone.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Ensemble weight optimization strategy | Instead of equal weighting or Optuna hyperparameter tuning, directly search the XGB/LGBM/CatBoost weight simplex to maximize composite score | Medium | Use scipy.optimize.minimize with Dirichlet constraint (weights sum to 1, all >= 0), or grid search over (0.1, 0.2, ... 0.7) for 3-model simplex. Pass found weights to EnsembleModel via `model_weights` dict. Requires evaluation to use the weighted ensemble path, not the simple per-model average. ~15-30 candidate weight sets is a reasonable budget. |
| Feature engineering strategy (auto-generate lags/rolling/interactions) | Discovers MOEX-specific signal combinations not in the hand-crafted 45-feature set | High | Generate candidate features: rolling windows (5, 10, 20, 60 days) of existing features, lag-1/lag-5 of macro features, ratio features (e.g. close/imoex, volume_zscore * cbr_direction). Then run MI-based selection from the expanded pool. Risk: combinatorial explosion — cap at generating 3x the existing feature count, then MI selects down to max_features. |
| Cross-segment transfer strategy (US to MOEX) | Validates whether feature sets and hyperparameter configs that worked for us_tech translate to ru_blue_chips, avoiding redundant search | High | Fetch the best experiment result from `results/experiments/us_tech_experiment_log.jsonl`, extract `features_used` and `hparams`, run those configs on the target MOEX segment without additional search. Record as a hypothesis: "US-optimal config transfers to MOEX". Expected outcome: partial transfer (regime/macro features transfer, US-specific cross-asset features do not). |
| Hypothesis-linked verdicts in ExperimentManager | Each autoresearch run becomes a tracked scientific hypothesis with automated ACCEPTED/REJECTED/INCONCLUSIVE verdict, linkable to debates | Low (given ExperimentManager exists) | The `SuccessCriteria` maps naturally to the composite score threshold (e.g., `metric: "avg_accuracy", threshold: 0.53, operator: ">="`). Verdict computation is already implemented in `ExperimentManager.compute_verdict()`. The work is wiring call sites: create experiment before loop, record after each fold batch, compute verdict at end. |
| Per-strategy experiment IDs | Each autoresearch strategy (ablation, efficiency, hyperparameter, random_subset, ensemble_weights, feature_engineering, cross_segment_transfer) creates a separate ExperimentState | Low | Allows surgical comparison: "did ensemble weight optimization on ru_blue_chips outperform hyperparameter search?" Experiment IDs follow pattern: `{segment_id}-{strategy}-{date}`. |

---

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Neural architecture search (LSTM hyperparameter sweep) | LSTM training is 10-50x slower than XGB/LGBM/CatBoost per fold; autoresearch on 730-day MOEX data would take hours per experiment | Keep LSTM out of autoresearch; train LSTM separately via `train_models.py --lstm` only when tree ensemble is validated |
| Optuna integration inside autoresearch loop | Optuna + walk-forward = nested optimization = severe overfitting; Optuna already exists in `train_models.py` for production training | Autoresearch uses coordinate perturbation (one param at a time) not Bayesian optimization |
| Real-time data fetching per experiment | Fetching candles for each of 100 experiments would hit TinkoffFetcher rate limits and take hours | Fetch once at loop start (like current `_prepare_data()`), cache in memory, reuse across all experiments |
| Automated preset application from autoresearch | Autoresearch is research, not production; the 7-gate PresetApplicator pipeline (circuit breaker, SandboxGate, etc.) exists for a reason | autoresearch records experiment_id; human or AgentOrchestrator decides whether to escalate to PresetApplicator |
| Multi-segment parallel execution | Parallel TinkoffFetcher gRPC calls from multiple processes will exhaust the single API token rate limits | Run segments sequentially; each `--segment` invocation is a separate CLI call |
| Feature importance persistence as model update | Autoresearch results are research artifacts, not model updates; writing new `selected_features.json` from autoresearch would bypass the production training pipeline | Log `features_used` in JSONL + ExperimentManager only; actual model update requires `train_models.py` run + quality gates |

---

## Feature Dependencies

```
MOEX segment symbols in _SEGMENT_SYMBOLS
  -> MOEX data adapter (TinkoffFetcher path)
    -> MOEX macro features in autoresearch pipeline
      -> Adaptive quality gates (signal_count n_eff scaling)
        -> ExperimentManager integration (hypothesis + verdict)
          -> Per-strategy experiment IDs
            -> Cross-segment transfer strategy (reads US experiment results)
            -> Ensemble weight optimization strategy
            -> Feature engineering strategy
```

Critical path: segment symbols -> data adapter -> macro features -> quality gates -> ExperimentManager. The 3 new search strategies depend on all prior steps being stable.

---

## MVP Recommendation

Prioritize:
1. MOEX segment symbols + TinkoffFetcher data adapter — without this, nothing runs on MOEX
2. MOEX macro features wired into `build_full_dataset()` — already computed in `train_models.py`, gap is only the fetch-and-wire
3. Adaptive signal_count gate (n_eff-scaled minimum) — current fixed threshold of 50 blocks all MOEX folds
4. MOEX-tuned default hyperparameters — prevents immediate overfitting before WF can detect it
5. ExperimentManager integration — research tracking, hypothesis lifecycle

Defer (later phases):
- Feature engineering strategy: high complexity, high overfitting risk, low expected signal quality on 730-day MOEX data where variance is high
- Cross-segment transfer strategy: valuable but depends on having stable US experiment history; run after MOEX baseline is established
- Ensemble weight optimization: medium value, medium complexity; run after quality gates are confirmed working

---

## MOEX-Specific Considerations

**Dataset size asymmetry:** US segments have 1825 days (~7.3 folds with current WF params). MOEX has 730 days. With current params (12mo train + 2mo cal + 4mo test, 3mo step), MOEX yields only ~1-2 folds — insufficient for `evaluate_walk_forward()` (needs multiple folds to compute gate pass rates). MOEX-specific WF params (6mo train + 1mo cal + 2mo test, 1.5mo step) yield ~4-5 folds from 730 days.

**Symbol universe:** MOEX blue chips (SBER, LKOH, GMKN, ROSN, NVTK, MGNT, TATN, TCSG) have very different data characteristics than US tech. Volume, turnover, and bid-ask spread patterns differ significantly. The ATR uplift of 1.2x (already in `_MOEX_ATR_UPLIFT`) must propagate to autoresearch triple-barrier labeling.

**Benchmark alignment:** MOEX uses IMOEX as benchmark (not SPY). The `_align_benchmark()` function is benchmark-agnostic. The `_fetch_benchmark()` function in `auto_ml_research.py` is hardcoded to SPY/yfinance; it needs a MOEX branch fetching IMOEX via MOEX ISS REST API.

**VIX substitute:** VIX is a US instrument. For MOEX segments, VIX features default to 0.0 (already the default in `technical.py` when `vix_candles=None`). No MOEX volatility index substitute is available via existing fetchers.

**Market hours and calendar:** TinkoffFetcher returns candles on MOEX trading days only (already handles Russian holidays). No additional calendar filtering needed in autoresearch.

---

## Sources

- `scripts/auto_ml_research.py` — current implementation, US-only, 4 strategies
- `scripts/train_models.py` lines 239-274 — MOEX segment symbols, `_is_moex_segment()`, MOEX-tuned hyperparameters
- `src/finalayze/ml/training/quality_gates.py` — existing gates; signal_count gate uses fixed `_MIN_SIGNALS = 50`
- `src/finalayze/ml/features/technical.py` lines 77-463 — 10 MOEX macro features already implemented, behind `MoexMarketData`
- `src/finalayze/core/schemas.py` lines 453-474, 720-806 — `MoexMarketData`, `ExperimentState`, `SuccessCriteria`
- `src/finalayze/core/experiment_manager.py` — ExperimentManager CRUD, verdict computation
- `src/finalayze/ml/models/ensemble.py` lines 40-175 — `model_weights` parameter, `_get_model_weight()` resolver
- `.planning/PROJECT.md` lines 166-173 — v9.0 active requirements

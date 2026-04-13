# Architecture Research

**Domain:** ML AutoResearch & MOEX Adaptation (v9.0)
**Researched:** 2026-04-13
**Confidence:** HIGH (direct codebase inspection, no external sources needed)

## Standard Architecture

### System Overview

```
scripts/auto_ml_research.py  (sync CLI, entry point)
        │
        ├── _prepare_data()           ← MODIFICATION: add MOEX branch
        │       ├── _fetch_us_candles()   (existing — yfinance)
        │       └── _fetch_moex_candles() (NEW — TinkoffFetcher via sync bridge)
        │
        ├── build_full_dataset()      ← MODIFICATION: accept moex_data kwarg
        │       └── build_triple_barrier_dataset()
        │               └── compute_features() in technical.py
        │                       └── MOEX macro features (already exist, just not wired)
        │
        ├── run_experiment()          ← MODIFICATION: pass min_signals threshold
        │       ├── _run_fold()
        │       └── evaluate_fold()   ← quality_gates.py (MODIFICATION: min_signals param)
        │
        └── _log_result()             ← MODIFICATION: also call ExperimentManager
                └── ExperimentManager.create_experiment() / link_result() (existing L0)
```

### New Components vs Modified Components

| Component | Status | Location | What Changes |
|-----------|--------|----------|--------------|
| `_fetch_moex_candles()` | NEW | `scripts/auto_ml_research.py` | Sync wrapper around TinkoffFetcher |
| `_fetch_moex_macro()` | NEW | `scripts/auto_ml_research.py` | Loads CBR + IMOEX + Brent into MoexMarketData |
| `_MOEX_SEGMENT_SYMBOLS` | NEW | `scripts/auto_ml_research.py` | Dict of ru_* segment to symbol lists |
| `_prepare_data()` | MODIFIED | `scripts/auto_ml_research.py` | Route by segment prefix (us_ vs ru_) |
| `build_full_dataset()` | MODIFIED | `scripts/auto_ml_research.py` | Accept `moex_data: MoexMarketData | None` kwarg |
| `run_research_loop()` | MODIFIED | `scripts/auto_ml_research.py` | Accept MOEX segment IDs; set max_features=10 |
| `main()` argparse | MODIFIED | `scripts/auto_ml_research.py` | Expand --segment choices to include ru_* |
| `evaluate_fold()` | MODIFIED | `src/finalayze/ml/training/quality_gates.py` | Add `min_signals: int` parameter |
| Experiment wiring | NEW (opt-in) | `scripts/auto_ml_research.py` | Call ExperimentManager when --experiment-id set |
| Ensemble weight strategy | NEW | `scripts/auto_ml_research.py` | `generate_ensemble_weight_experiments()` |
| Feature engineering strategy | NEW | `scripts/auto_ml_research.py` | `generate_feature_engineering_experiments()` |
| Cross-segment transfer strategy | NEW | `scripts/auto_ml_research.py` | `generate_transfer_experiments()` |

## Async/Sync Bridging

### The Problem

`auto_ml_research.py` is a synchronous CLI script. `TinkoffFetcher` exposes a sync `fetch_candles()` method that internally uses `run_coroutine_threadsafe()` on a self-managed background event loop. So **TinkoffFetcher is already safe to call from sync code without any new bridging**.

### How TinkoffFetcher Self-Bridges

```
sync caller (autoresearch script)
    │
    └── TinkoffFetcher.fetch_candles(sym, start, end)     [sync method]
            │
            └── self._run_async(self._fetch_async(...))
                    │
                    └── asyncio.run_coroutine_threadsafe(coro, self._loop)
                            │
                            └── background daemon thread running event loop
                                    │
                                    └── gRPC call → T-Bank API
```

**Conclusion:** No new bridging code needed. The self-managed loop fallback in `_run_async()` (lines 119-132 of `tinkoff_data.py`) handles standalone script use cases explicitly. The script only needs to construct `TinkoffFetcher` with a valid token and registry.

### Construction in Autoresearch Script

TinkoffFetcher requires two constructor dependencies:
1. `token: str` — from `FINALAYZE_TINKOFF_TOKEN` env var
2. `registry: InstrumentRegistry` — from `build_default_registry()` in `src/finalayze/markets/instruments.py`

```python
# Minimal construction for autoresearch
import os
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import build_default_registry

def _make_tinkoff_fetcher() -> TinkoffFetcher:
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    if not token:
        raise RuntimeError("FINALAYZE_TINKOFF_TOKEN not set")
    registry = build_default_registry()
    return TinkoffFetcher(token=token, registry=registry, sandbox=True)
```

`CBRFetcher` and `MoexISSFetcher` are async HTTP (httpx-based). Verify at implementation whether they expose sync wrappers. If not, use `asyncio.run()` once per fetch call — acceptable for a CLI script that is not itself running inside an event loop.

## MOEX Macro Feature Generation

### Existing vs New

The `technical.py` module already computes 10 MOEX macro features when `MoexMarketData` is provided:
- `usdrub_zscore_60d` (FX z-score)
- `brent_zscore_60d` (commodity z-score with holiday suppression)
- `real_rate_zscore` (CBR rate minus CPI, 252d window)
- `cbr_rate_level`, `cbr_rate_delta` (key rate features)
- `usdrub_return`, `usdrub_vol` (FX return features)
- `turnover_zscore` / `turnover_ratio` (MOEX turnover features)
- `imoex_beta`, `imoex_corr` (cross-asset features against IMOEX benchmark)

**No new feature code is needed in `technical.py`.** The gap is that `auto_ml_research.py` currently passes `vix_candles` and `benchmark_candles` to `build_triple_barrier_dataset()` but never constructs or passes a `MoexMarketData` object, so all MOEX macro features silently default to 0.0 for MOEX segments.

### Integration Point

`build_full_dataset()` must be extended to accept and pass through `moex_data`:

```python
def build_full_dataset(
    segment_id: str,
    candles_by_sym: dict[str, list[Candle]],
    benchmark_candles: list[Candle] | None,
    vix_candles: list[Candle] | None,
    moex_data: MoexMarketData | None = None,  # NEW parameter
) -> tuple[...]:
    ...
    market_ctx = MarketContext(
        benchmark_candles=benchmark_candles,
        vix_candles=vix_candles,
        moex_data=moex_data,  # pass through
    )
```

`MarketContext` already holds `moex_data: MoexMarketData | None`. `build_triple_barrier_dataset()` already receives `market_context` and routes to macro features. The wire-up is a 2-line change.

### Macro Data Sources for Autoresearch

The `_fetch_moex_macro()` function must load:
1. IMOEX candles — via `MoexISSFetcher.fetch_candles("IMOEX", start, end)` (serves as benchmark)
2. USD/RUB rates — via `CBRFetcher.fetch_fx_rates(start, end)`
3. CBR key rates — via `CBRFetcher.fetch_key_rates(start, end)`
4. Brent candles — via `YFinanceFetcher.fetch_candles("BZ=F", start, end)` (yfinance is fine for Brent)
5. MOEX turnover — via `MoexISSFetcher.fetch_turnover(start, end)` (optional, suppress if unavailable)

Then construct `MoexMarketData(fx_rates=..., key_rates=..., commodity_candles={"BZ=F": brent}, ...)` and pass to `build_full_dataset`.

The preferred shortcut: instantiate `MarketDataLoader` with the same fetchers already used in production and call `loader.load(segment_config_stub, start.date(), end.date())` to get a ready-made `MarketContext` including `moex_data`. This reuses all caching and error-handling logic already tested.

## ExperimentManager Integration

### Current State

`ExperimentManager` is a file-based CRUD store at Layer 0. It operates entirely in sync (file I/O + YAML). It has:
- `create_experiment(id, hypothesis, success_criteria, debate_id, preset_overrides)` — writes `.planning/experiments/{id}.md`
- `link_result(id, ExperimentResult)` — appends backtest metrics
- `record_verdict(id, metric_value)` — computes ACCEPTED/REJECTED/INCONCLUSIVE

### Integration Pattern for Autoresearch

Each research run maps to one `ExperimentManager` experiment. The mapping:

```
autoresearch concept           ExperimentManager concept
---------------------------------------------------------
ExperimentConfig.name       →  experiment_id
ExperimentConfig.description→  hypothesis
ExperimentResult.score      →  metric_value (for record_verdict)
gate_pass_rates + avg_*     →  ExperimentResult.metrics dict
```

### Wire-up Location

Add optional `--experiment-id` CLI flag. When set, the script:
1. Calls `ExperimentManager.create_experiment()` once at start of `run_research_loop()`
2. After each individual experiment in the loop, calls `em.link_result()` with fold metrics
3. After the loop completes, calls `em.record_verdict()` with `best_score` as the observed metric

When `--experiment-id` is not set, the script runs as before (JSONL log only). This makes the integration opt-in and non-breaking for existing invocations.

### SuccessCriteria for Autoresearch Experiments

```python
SuccessCriteria(
    metric="composite_score",
    threshold=baseline_score,   # must beat the baseline to be ACCEPTED
    operator="gt",
)
```

## New Search Strategies

### Ensemble Weight Optimization

**What it does:** Tries different weighting ratios for XGBoost / LightGBM / CatBoost rather than the default equal-weight average used in `_evaluate_models()`.

**Implementation:** New `generate_ensemble_weight_experiments()`. Adds `ensemble_weights: dict[str, float]` to `ExperimentConfig`. `_evaluate_models()` accepts optional per-model weights and computes weighted average instead of arithmetic mean.

**Weight space:** Grid of (xgb: 0.2-0.6, lgbm: 0.2-0.5, cat: 0.1-0.4), normalized to sum=1.0, ~9-12 configs. Weights stored in `ExperimentConfig.hparams` under keys `w_xgb`, `w_lgbm`, `w_cat`.

### Automatic Feature Engineering

**What it does:** Computes derived features (ratios, lags, pairwise interactions) on top of the base 45 features, then runs efficiency selection on the expanded pool.

**Implementation:** New `generate_feature_engineering_experiments()`. Pre-processing step computes interaction terms (e.g., `rsi14 * volume_zscore_20d`) and lag features (`feat_t-1`, `feat_t-2`). Expands the feature dict before passing to `build_triple_barrier_dataset`. Feature selection (`select_features_efficient`) then prunes from the larger pool.

**Constraints:** Cap at 20 interaction terms (selected by marginal mutual information with label). Lag features add exactly 2 copies per selected base feature. Total expanded pool stays below 90 features to keep selection tractable.

### Cross-Segment Transfer

**What it does:** Trains on US segment data, evaluates on MOEX segment data to test feature transferability. Tests whether US-learned feature importance carries over to MOEX dynamics.

**Implementation:** New `generate_transfer_experiments()`. Loads both segment datasets independently. Computes `shared_features = set(us_features) & set(moex_features)` — this intersection excludes VIX features (US-only) and MOEX macro features (MOEX-only). Trains on US folds using shared features only, evaluates on MOEX test folds.

**Key constraint:** VIX-derived features (vix_level, vix_pct, vix_change) must be excluded from the transferred subset because MOEX returns 0.0 for them. The intersection check handles this automatically.

## Adaptive Quality Gates

### Current Problem

`check_signal_count_gate` uses `_MIN_SIGNALS = 50` as a hard constant. MOEX segments have 3-8 symbols with 730-day lookbacks, producing 15-30 signals per fold — a data-size reality, not a model failure. The gate currently makes every MOEX experiment automatically fail the signal_count check.

### Proposed Fix

Add `min_signals: int = _MIN_SIGNALS` parameter to `evaluate_fold()` and propagate it through `_run_fold()`:

```python
_MOEX_MIN_SIGNALS = 15  # new constant in quality_gates.py

def evaluate_fold(
    metrics: FoldMetrics,
    *,
    min_signals: int = _MIN_SIGNALS,  # NEW parameter
) -> list[QualityGateResult]:
    return [
        check_accuracy_gate(metrics),
        check_brier_gate(metrics),
        check_profit_factor_gate(metrics),
        check_signal_count_gate(metrics, min_signals=min_signals),  # threaded through
        check_class_balance_gate(metrics),
        check_sensitivity_gate(metrics),
        check_specificity_gate(metrics),
    ]
```

Autoresearch script passes `min_signals=_MOEX_MIN_SIGNALS` for `ru_*` segments, `min_signals=_MIN_SIGNALS` for `us_*`. All existing callers of `evaluate_fold()` that pass no `min_signals` get unchanged behavior.

The accuracy gate already has adaptive thresholds for small n_eff (the `_MAX_ACCURACY_THRESHOLD = 0.55` cap). The Brier gate already uses `_dynamic_brier_threshold(n_eff)`. Signal count is the only gate that needs this treatment.

## Data Flow: MOEX Autoresearch Path

```
CLI: uv run python scripts/auto_ml_research.py --segment ru_blue_chips --strategy all

_prepare_data("ru_blue_chips")
    │
    ├── _fetch_moex_candles(["SBER","LKOH","GMKN"], start, end)
    │       → TinkoffFetcher.fetch_candles()  [sync self-managed loop — no bridge needed]
    │       → dict[str, list[Candle]]
    │
    └── _fetch_moex_macro(start, end)
            → CBRFetcher.fetch_fx_rates()      [asyncio.run() or MarketDataLoader]
            → CBRFetcher.fetch_key_rates()
            → MoexISSFetcher.fetch_candles("IMOEX")
            → YFinanceFetcher.fetch_candles("BZ=F")
            → MoexMarketData(fx_rates, key_rates, commodity_candles, ...)
    │
    ↓
build_full_dataset(segment_id, candles_by_sym, benchmark=IMOEX, vix=None, moex_data=moex_data)
    → build_triple_barrier_dataset() per symbol
        → compute_features()
            → 45 technical features
            → _compute_fx_features(moex_data)        → usdrub_zscore_60d
            → _compute_commodity_features(moex_data) → brent_zscore_60d
            → _compute_macro_features(moex_data)     → real_rate_zscore
            → _compute_cbr_features(moex_data)       → cbr_rate_level, cbr_rate_delta
    → features: list[dict[str, float]] with MOEX macro features populated (not zeros)
    │
    ↓
generate_folds(timestamps)  [no change]
    │
    ↓
run_research_loop() with max_features=_MOEX_MAX_FEATURES=10, min_signals=15
    → baseline experiment
    → ablation / efficiency / hyperparameter / ensemble_weight / feature_eng experiments
    → _run_fold() → evaluate_fold(metrics, min_signals=15)
    → _log_result() to JSONL
    → ExperimentManager.link_result()  [if --experiment-id set]
    │
    ↓
ExperimentManager.record_verdict(experiment_id, best_score)
    → .planning/experiments/{id}.md updated with ACCEPTED/REJECTED/INCONCLUSIVE
```

## Recommended File Changes

```
scripts/
└── auto_ml_research.py          # MODIFIED (~280 lines added)
    ├── _MOEX_SEGMENT_SYMBOLS    # new constant dict
    ├── _fetch_moex_candles()    # new function
    ├── _fetch_moex_macro()      # new function
    ├── build_full_dataset()     # moex_data param added
    ├── _prepare_data()          # us/moex routing branch
    ├── _run_fold()              # min_signals param threaded through
    ├── generate_ensemble_weight_experiments()    # new strategy
    ├── generate_feature_engineering_experiments() # new strategy
    ├── generate_transfer_experiments()            # new strategy
    └── run_research_loop()      # ExperimentManager opt-in wiring

src/finalayze/ml/training/
└── quality_gates.py             # MODIFIED: min_signals param in evaluate_fold()
                                 #           _MOEX_MIN_SIGNALS = 15 constant
```

No new source modules are needed. All changes are additive within existing files.

## Build Order (Dependency-Driven)

**Step 1: MOEX data adapter** (foundation, no ML changes)
- Add `_MOEX_SEGMENT_SYMBOLS`, `_fetch_moex_candles()`, `_fetch_moex_macro()` to autoresearch script
- Route `_prepare_data()` on segment prefix (`ru_` vs `us_`)
- Verify TinkoffFetcher constructs correctly with `build_default_registry()`
- Confirm CBRFetcher / MoexISSFetcher can be called from CLI context

Deliverable: `--segment ru_blue_chips` loads data without error, prints candle counts

**Step 2: MOEX macro features wired** (depends on Step 1)
- Add `moex_data` kwarg to `build_full_dataset()` and `MarketContext` pass-through
- Verify MOEX macro features are non-zero in output feature dicts for MOEX segments

Deliverable: Feature dicts for MOEX include `usdrub_zscore_60d`, `brent_zscore_60d`, etc. with real values

**Step 3: Adaptive quality gates** (independent — can be done in parallel with Steps 1-2)
- Add `min_signals: int = _MIN_SIGNALS` to `evaluate_fold()` and `check_signal_count_gate()`
- Add `_MOEX_MIN_SIGNALS = 15` constant to quality_gates.py
- Pass correct threshold from autoresearch based on segment prefix

Deliverable: MOEX experiments no longer auto-fail signal_count gate on 15-30 signals per fold

**Step 4: ExperimentManager integration** (depends on Step 1, independent of Steps 2-3)
- Add `--experiment-id` optional CLI arg
- Add opt-in `ExperimentManager` wiring in `run_research_loop()`
- Map autoresearch `ExperimentResult` fields to `ExperimentManager.link_result()` format

Deliverable: `--experiment-id moex-v9-ablation` creates experiment file with metrics and verdict

**Step 5: New search strategies** (depends on Steps 1-3 having a working MOEX baseline)
- `generate_ensemble_weight_experiments()` — modify `_evaluate_models()` to accept weights
- `generate_feature_engineering_experiments()` — interaction terms + lag features preprocessing
- `generate_transfer_experiments()` — cross-segment shared-features training

Deliverable: All three new `--strategy` choices work on both US and MOEX segments

## Integration Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| autoresearch script → TinkoffFetcher | Direct sync call | Self-managed background loop handles standalone script use; no bridge needed |
| autoresearch script → CBRFetcher | `asyncio.run()` or via MarketDataLoader | Must not be called from inside a running event loop |
| autoresearch script → ExperimentManager | Direct sync CRUD (file I/O) | Fully sync; no async involved |
| autoresearch script → quality_gates | Direct function call with new param | Add min_signals; all existing callers get unchanged default behavior |
| technical.py → MoexMarketData | Kwarg pass-through via MarketContext | Already implemented in technical.py; gap is only in autoresearch pipeline wiring |
| new strategies → existing experiment loop | In-process function calls | No new processes or threads; same event loop context |

## Anti-Patterns

### Anti-Pattern 1: Re-implementing Async Data Fetching

**What people do:** Write a new `asyncio.run(gather_all_moex_data())` orchestrator in the script that duplicates `MarketDataLoader._load_moex()`.

**Why it's wrong:** `MarketDataLoader` already handles CBR + IMOEX + Brent + turnover + file caching for MOEX. Duplicating it creates two maintenance surfaces and loses the file-cache benefit (critical for CI speed when running 100 experiments).

**Do this instead:** Instantiate `MarketDataLoader` with the same fetchers it uses in production, call `loader.load(segment_config, start.date(), end.date())`, extract `market_ctx.moex_data`. Reuse existing tested infrastructure.

### Anti-Pattern 2: Modifying Quality Gate Constants Globally

**What people do:** Change `_MIN_SIGNALS = 50` to `_MIN_SIGNALS = 15` globally to make MOEX experiments pass.

**Why it's wrong:** This breaks US segment quality enforcement (50 signals is correct for 15-symbol US datasets). MOEX and US need different thresholds.

**Do this instead:** Add `min_signals: int = _MIN_SIGNALS` parameter to `evaluate_fold()`. Pass `_MOEX_MIN_SIGNALS` from the autoresearch script when the segment is `ru_*`. Default remains unchanged for all existing callers (train_models.py, backtest engine).

### Anti-Pattern 3: Cross-Segment Features Without Intersection Check

**What people do:** Train on US features (which include VIX-derived features) and apply directly to MOEX folds where VIX is None.

**Why it's wrong:** VIX features default to 0.0 for MOEX. A US-trained model that relies heavily on VIX features will produce systematically biased predictions on MOEX zero-VIX inputs, and the experiment measures model-data mismatch not feature transferability.

**Do this instead:** In `generate_transfer_experiments()`, compute `shared_features = set(us_features) & set(moex_features)` and restrict the transfer subset to shared features only. The `vix_*` and MOEX-specific macro features are mutually exclusive by design and will not appear in the intersection.

## Sources

- Direct inspection of `scripts/auto_ml_research.py` (960 lines, v9.0 baseline)
- Direct inspection of `src/finalayze/data/fetchers/tinkoff_data.py` (TinkoffFetcher._run_async, lines 106-132)
- Direct inspection of `src/finalayze/ml/features/technical.py` (MOEX macro feature functions, lines 394-650)
- Direct inspection of `src/finalayze/ml/training/quality_gates.py` (evaluate_fold, adaptive thresholds)
- Direct inspection of `src/finalayze/core/experiment_manager.py` (CRUD API: create/link/verdict)
- Direct inspection of `src/finalayze/data/loader.py` (MarketDataLoader._load_moex)
- Direct inspection of `config/segments.py` (ru_* segment definitions and symbol lists)

---
*Architecture research for: v9.0 ML AutoResearch & MOEX Adaptation*
*Researched: 2026-04-13*

# Technology Stack — v9.0 ML AutoResearch & MOEX Adaptation

**Project:** Finalayze v9.0
**Researched:** 2026-04-13
**Confidence:** HIGH (derived from direct codebase reads; no new library introductions)

---

## Premise: What Already Exists (Do Not Re-Research)

These capabilities are shipped and verified. v9.0 extends them without rewrites.

| Already Exists | Location |
|----------------|----------|
| `TinkoffFetcher` — async gRPC client with self-managed background event loop | `src/finalayze/data/fetchers/tinkoff_data.py` |
| `CBRFetcher` — sync XML client for key rate + FX rates | `src/finalayze/data/fetchers/cbr.py` |
| `MoexMarketData` schema — `fx_rates`, `key_rates`, `commodity_candles`, `index_candles` | `src/finalayze/core/schemas.py:453` |
| MOEX macro features — `_compute_cbr_features()`, `_compute_brent_return_features()`, `_compute_brent_zscore()`, `_compute_macro_features()` (10 features) | `src/finalayze/ml/features/technical.py` |
| `select_features_efficient()` — MI + complexity-weighted greedy selection | `src/finalayze/ml/training/feature_selection.py` |
| `check_accuracy_gate()` — n_eff-scaled threshold with `_MAX_ACCURACY_THRESHOLD = 0.55` cap | `src/finalayze/ml/training/quality_gates.py` |
| `_dynamic_brier_threshold()` — threshold relaxes with n_eff via sqrt scaling | `src/finalayze/ml/training/quality_gates.py` |
| `ExperimentManager` — CRUD + automated verdict (ACCEPT/REJECT/INCONCLUSIVE) | `src/finalayze/core/experiment_manager.py` |
| `InstrumentRegistry` — symbol → FIGI + market metadata | `src/finalayze/markets/instruments.py` |
| XGBoost + LightGBM + CatBoost + meta-learner ensemble | `src/finalayze/ml/models/` |
| Walk-forward folds with purge gaps | `scripts/auto_ml_research.py:generate_folds()` |
| Existing `_MOEX_LOOKBACK_DAYS = 730`, `_MOEX_MAX_FEATURES = 10`, `_MOEX_ATR_UPLIFT = 1.2` constants | `scripts/auto_ml_research.py` |
| Segments `ru_blue_chips`, `ru_energy`, `ru_tech`, `ru_finance` with symbols defined | `config/segments.py` |

---

## What v9.0 Needs: Gap Analysis

| Requirement | Gap | Solution |
|-------------|-----|---------|
| MOEX data adapter for `auto_ml_research` | `_prepare_data()` calls `_fetch_us_candles()` (yfinance) only; no MOEX branch | Add `_fetch_moex_candles()` using `TinkoffFetcher._run_async()` bridge pattern; gated by `segment_id.startswith("ru_")` |
| MOEX macro features in autoresearch | `build_full_dataset()` passes `benchmark_candles` + `vix_candles` but no `MoexMarketData`; `build_triple_barrier_dataset()` would skip MOEX features | Pass `MoexMarketData` to `build_full_dataset()` for MOEX segments; fetch CBR/IMOEX/Brent via existing fetchers |
| Adaptive quality gates (already partially done) | `check_signal_count_gate()` uses `_MIN_SIGNALS = 50` — MOEX daily data gives ~500 bars over 2 years across 3-5 symbols; 50 signals is feasible but needs verification | Existing n_eff scaling in accuracy/brier gates covers the main concern; `signal_count` gate may need MOEX-specific floor |
| ExperimentManager integration | `auto_ml_research.py` logs to `results/experiments/*.jsonl` (flat files) but never creates `ExperimentState` objects or calls `ExperimentManager` | Add `ExperimentManager.create()` call at experiment start, `record_verdict()` at completion, linking to `DebateManager` when score delta is significant |
| Ensemble weight optimization strategy | Only per-model hyperparameters perturbed; no strategy for optimizing XGBoost:LightGBM:CatBoost relative weights | New `generate_ensemble_weight_experiments()` — grid over weight combinations in the `EnsembleModel._weights` space |
| Automatic feature engineering strategy | No cross-feature polynomial or interaction generation | Use existing `pandas-ta` to generate new indicators programmatically within experiment loop — no new library needed |
| Cross-segment transfer strategy | No mechanism to initialize MOEX model from US model weights or feature subsets | Transfer selected feature names from best US experiment run as initial feature_subset for MOEX baseline; no model weight transfer (tree models don't support it) |

**Zero new packages required.** All gaps close with existing dependencies and stdlib.

---

## Recommended Stack

### Core Technologies (no new packages)

| Technology | Version (installed) | Purpose | Why |
|------------|--------------------|---------|----|
| `t-tech-investments` (T-Bank SDK) | installed (custom index) | gRPC candle/dividend fetch for all MOEX symbols in autoresearch | Already wired in `TinkoffFetcher`; self-managed background loop (`_run_async`) works from sync scripts — this is the existing pattern used in standalone scripts today |
| `asyncio` + `threading` | Python 3.12 stdlib | Bridge sync autoresearch script to async `TinkoffFetcher` | `TinkoffFetcher._run_async()` already handles this via `run_coroutine_threadsafe()` on a daemon thread; no `nest_asyncio` or `asyncio.run()` needed — just construct `TinkoffFetcher` with no `grpc_loop` injected (self-managed fallback path) |
| `httpx` | >=0.28.0 (installed) | CBR key rate + MOEX ISS IMOEX fetch in autoresearch context | Already used by `CBRFetcher` (sync) and `MoexISSFetcher` (sync) — both work from sync scripts without async wrapping |
| `scikit-learn` | >=1.5.0 (installed) | MI-based feature selection + quality gate metrics (`accuracy_score`, `brier_score_loss`) | Already the engine behind `select_features_efficient()`; no changes needed |
| `XGBoost` | >=2.1.0 (installed) | Ensemble member + feature importance | No changes needed |
| `LightGBM` | >=4.5.0 (installed) | Ensemble member | No changes needed |
| `CatBoost` | >=1.2.0 (installed) | Ensemble member — especially effective for small tabular MOEX datasets (native categorical support) | No changes needed |
| `pandas-ta` | >=0.3.14b1 (installed) | Automatic feature engineering strategy: generate new indicator combinations from candle data within experiment loop | Already the feature engineering library; new strategy generates feature dicts by varying indicator parameters |
| `PyYAML` | >=6.0.2 (installed) | `ExperimentManager` reads/writes YAML frontmatter | Already the pattern; no new usage |
| `structlog` | >=24.4.0 (installed) | Structured experiment logging | Already used; bind `segment_id`, `strategy`, `experiment_id` to all log events |

### Supporting Libraries (no new packages)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `numpy` | >=1.26.0 (installed) | Walk-forward fold arrays, n_eff calculation | No changes |
| `pandas` | >=2.2.0 (installed) | Feature DataFrame construction for `select_features_efficient()` | No changes |
| `optuna` | >=4.7.0 (installed) | Already in stack for backtest hyperparameter tuning — NOT used in autoresearch loop directly; hyperparameter strategy uses coordinate descent instead | Avoid adding Optuna to autoresearch; existing coordinate-descent perturbation is faster for the hypothesis loop |
| `pathlib` | Python 3.12 stdlib | Experiment log path construction, results directory | No changes |

### New Modules to Create (no new packages)

| Module | Location | Layer | Responsibility |
|--------|----------|-------|---------------|
| MOEX data fetch functions | `scripts/auto_ml_research.py` | Script | `_fetch_moex_candles()`, `_fetch_moex_benchmark()`, `_fetch_moex_macro()` — construct `TinkoffFetcher` without injected loop, call `fetch_candles()` for each symbol; construct `CBRFetcher` for key rates; `MoexISSFetcher` for IMOEX |
| MOEX segment symbol map | `scripts/auto_ml_research.py` | Script | Extend `_SEGMENT_SYMBOLS` dict with `ru_blue_chips`, `ru_energy`, `ru_tech`, `ru_finance` entries mirroring `config/segments.py` |
| `generate_ensemble_weight_experiments()` | `scripts/auto_ml_research.py` | Script | Grid search over XGB:LGBM:CAT relative weights `[0.2,0.5,0.3]`, `[0.33,0.33,0.33]`, etc.; pass weight vector as `hparams` key `ensemble_weights` |
| `generate_feature_engineering_experiments()` | `scripts/auto_ml_research.py` | Script | Use `pandas-ta` to generate new derived features (RSI with periods 7/21, MACD fast/slow variations) from existing candle data; wrap as feature_subset experiments |
| `generate_transfer_experiments()` | `scripts/auto_ml_research.py` | Script | Load best US experiment's `features_used` list from `results/experiments/us_tech_experiment_log.jsonl`; use as initial `feature_subset` for MOEX baseline run |
| ExperimentManager integration | `scripts/auto_ml_research.py` | Script | Call `ExperimentManager.create()` at research start; `record_verdict()` per experiment; use score delta threshold (0.02) to decide if verdict warrants debate linkage |

---

## Integration Architecture

### TinkoffFetcher in Sync Script

The key design decision: `TinkoffFetcher` already supports sync usage via its self-managed background event loop fallback. When constructed without `grpc_loop=...` injected, it creates its own daemon thread running an event loop. `_run_async()` calls `run_coroutine_threadsafe()`.

```python
# In scripts/auto_ml_research.py — no new imports needed
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import InstrumentRegistry, DEFAULT_MOEX_INSTRUMENTS

def _fetch_moex_candles(segment_id: str, symbols: list[str]) -> dict[str, list[Candle]]:
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    registry = InstrumentRegistry(DEFAULT_MOEX_INSTRUMENTS)
    fetcher = TinkoffFetcher(token=token, registry=registry, sandbox=False)
    # No grpc_loop= -> self-managed background loop activates automatically
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=_MOEX_LOOKBACK_DAYS)
    candles_by_sym: dict[str, list[Candle]] = {}
    for sym in symbols:
        try:
            candles = fetcher.fetch_candles(sym, start, end)
            if candles:
                candles_by_sym[sym] = candles
        except Exception as exc:
            print(f"  Failed to fetch {sym}: {exc}")
    fetcher.close()
    return candles_by_sym
```

**Why this works:** `TinkoffFetcher._run_async()` checks `self._grpc_loop` first; if None, spins up the self-managed loop. This is the documented fallback path for "tests, standalone scripts" (see the existing docstring). No `asyncio.run()` nesting, no `nest_asyncio`, no new event loop handling.

### MOEX Macro Data for Feature Pipeline

`build_triple_barrier_dataset()` accepts `market_context: MarketContext` which contains an optional `moex_data: MoexMarketData`. The MOEX feature functions in `technical.py` only activate when `moex_data` is populated. The fetch pattern for autoresearch:

```python
from finalayze.data.fetchers.cbr import CBRFetcher
from finalayze.data.fetchers.moex_iss import MoexISSFetcher
from finalayze.core.schemas import MoexMarketData

def _fetch_moex_macro() -> MoexMarketData:
    cbr = CBRFetcher()
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=_MOEX_LOOKBACK_DAYS)
    key_rates = tuple(cbr.fetch_key_rate(start.date(), end.date()))
    fx_rates = tuple(cbr.fetch_fx_rates("USD", start.date(), end.date()))
    # IMOEX via MoexISSFetcher, Brent via yfinance (already installed)
    ...
    return MoexMarketData(fx_rates=fx_rates, key_rates=key_rates, ...)
```

All three fetchers (`CBRFetcher`, `MoexISSFetcher`, `YFinanceFetcher` for Brent `BZ=F`) are sync and safe in a script context.

### ExperimentManager Integration

`ExperimentManager` lives at Layer 0. Scripts can import it directly (no layer violation). Integration is additive — existing flat JSONL log is kept for backward compat; ExperimentManager creates parallel `.md` files:

```python
from finalayze.core.experiment_manager import ExperimentManager

em = ExperimentManager()  # defaults to .planning/experiments/

# At research start:
exp_id = f"automl-{segment_id}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
em.create(exp_id, hypothesis=f"Autoresearch {strategy} on {segment_id}", ...)

# Per-experiment result:
em.record_verdict(exp_id, metrics={"score": result.score, "accuracy": result.avg_accuracy, ...})
```

---

## Adaptive Quality Gates: Current State Assessment

The quality gate code already has the primary MOEX adaptations:

1. `check_accuracy_gate()` caps threshold at 0.55 when `n_effective < 20` — this handles small MOEX datasets
2. `check_brier_gate()` uses `_dynamic_brier_threshold()` that relaxes toward 0.25 as n_eff grows
3. `check_signal_count_gate()` uses fixed `_MIN_SIGNALS = 50`

**Gap:** `_MIN_SIGNALS = 50` is the one gate that may block MOEX experiments. With 730 days × 3-5 symbols and triple-barrier labeling at a 20-bar horizon, total samples are ~200-500 after purging. A 4-month test fold produces ~50-80 signals. Gate passes at the margin.

**Recommendation:** No code change needed for the gate threshold; verify empirically on first MOEX run. If signal_count gate fails consistently, add a `moex_min_signals` parameter (default 30) rather than changing the shared constant. This is a configuration change, not an architecture change.

---

## Cross-Segment Transfer: Exact Mechanism

Tree models (XGBoost, LightGBM, CatBoost) do not support weight initialization from pre-trained models — transfer learning in the neural network sense is not applicable. The meaningful transfer is **feature subset transfer**:

1. Load the top-scoring US experiment's `features_used` list from `results/experiments/us_tech_experiment_log.jsonl`
2. Filter to features that are market-neutral (no US-specific: remove `vix_*`, `spy_*`; keep: `rsi_14`, `macd_*`, `atr_*`, `obv_*`, `ret_*d`, `calendar_*`)
3. Use filtered list as `feature_subset` for the MOEX baseline experiment
4. MOEX-specific features (`cbr_rate_level`, `usdrub_zscore`, `brent_zscore_60d`, etc.) are added on top

This requires zero new infrastructure — `generate_transfer_experiments()` reads the JSONL log (already written by `_log_result()`) and constructs `ExperimentConfig` with the filtered feature_subset.

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| `TinkoffFetcher` self-managed loop (no `grpc_loop` injection) in autoresearch script | `asyncio.run()` wrapping all MOEX fetches | `asyncio.run()` creates a new event loop per call — gRPC channels are destroyed between calls, causing 1-3s reconnect overhead per symbol. The `_run_async()` pattern was specifically designed to reuse channels |
| `nest_asyncio` for running async inside sync script | Not needed | TinkoffFetcher already handles sync-from-sync via `run_coroutine_threadsafe()` on a background daemon thread — nest_asyncio is only needed when an event loop is already running (e.g., Jupyter) |
| Transfer feature names only (no model weights) | PyTorch-style `state_dict()` transfer to tree models | XGBoost/LightGBM/CatBoost have no weight initialization API compatible with cross-domain transfer. Feature name transfer is the correct analog for tree ensembles |
| `pandas-ta` for automatic feature engineering | `tsfresh` or `featuretools` | `tsfresh` generates 700+ features, far beyond the 10-15 budget; `featuretools` requires entity/relationship modeling overkill for OHLCV. `pandas-ta` is already installed and generates exactly the indicator types the existing feature pipeline uses |
| Coordinate descent hyperparameter perturbation (existing) | Optuna for autoresearch hyperparameters | Optuna is already used for backtest tuning; adding it to the autoresearch inner loop creates nested optimization that obscures the experiment signal. Coordinate descent is simpler and auditable |
| Flat `_MIN_SIGNALS` gate verified empirically first | Immediate MOEX-specific threshold | MOEX datasets may actually hit 50+ signals per fold; premature optimization of the gate threshold adds complexity before we know it's needed |

---

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `tsfresh` | Generates ~700 features, completely incompatible with 10-15 feature budget; 2-5 min per symbol to compute | `pandas-ta` variations within the existing feature pipeline |
| `featuretools` | Entity-relationship modeling for OHLCV is architectural mismatch; 50MB+ dependency | Direct `pandas-ta` indicator parameter grid |
| `ray` or `dask` for parallel experiments | autoresearch is I/O-bound (model training), not CPU-bound at this scale; 4-8 symbols × 3 folds = ~24 train calls per experiment | Sequential loop with accurate timing is sufficient; add only if experiment wall time exceeds 2 hours |
| PyTorch transfer learning | Tree models have no weight initialization API; adds 500MB CUDA dependency for a feature that cannot work | Feature name transfer only |
| `nest_asyncio` | Not needed — `TinkoffFetcher` self-managed loop handles the sync→async bridge without an already-running event loop | Existing `_run_async()` pattern |
| New MLflow / experiment tracking DB | Adds infra dependency; `ExperimentManager` + JSONL log covers all tracking needs | Existing `.planning/experiments/*.md` + `results/experiments/*.jsonl` |
| Separate process or subprocess for MOEX fetch | Overcomplicated; gRPC works fine on a background thread in the same process | `TinkoffFetcher` daemon thread pattern |

---

## Installation

No new packages. Verify MOEX token is set:

```bash
echo $FINALAYZE_TINKOFF_TOKEN  # must be non-empty for MOEX segments
```

Verify existing libraries are importable in script context:

```bash
cd /Users/f1xgun/finalayze
uv run python -c "from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher; print('OK')"
uv run python -c "from finalayze.data.fetchers.cbr import CBRFetcher; print('OK')"
uv run python -c "from finalayze.core.experiment_manager import ExperimentManager; print('OK')"
```

---

## Version Compatibility

| Package | Installed | Feature Used | v9.0 Constraint |
|---------|-----------|-------------|-----------------|
| `t-tech-investments` | custom index | `AsyncClient`, `CandleInterval` | Requires `target="invest-public-api.tbank.ru:443"` — already set in `TinkoffFetcher._make_client()` |
| `scikit-learn` | >=1.5.0 | `mutual_info_classif`, `accuracy_score` | No changes |
| `xgboost` | >=2.1.0 | Training + feature importance | No changes |
| `lightgbm` | >=4.5.0 | Training | No changes |
| `catboost` | >=1.2.0 | Training | No changes |
| `pandas-ta` | >=0.3.14b1 | Feature engineering experiments | Existing import pattern; same `ta.rsi()` etc. calls |
| `httpx` | >=0.28.0 | `CBRFetcher` sync client | No changes |

---

## Sources

- `src/finalayze/data/fetchers/tinkoff_data.py` — `_run_async()` docstring confirms "tests, standalone scripts" fallback path with self-managed loop (HIGH confidence, direct read)
- `src/finalayze/ml/training/quality_gates.py` — `check_accuracy_gate()` n_eff scaling and `_MAX_ACCURACY_THRESHOLD = 0.55` cap confirmed (HIGH confidence, direct read)
- `src/finalayze/ml/features/technical.py` — 10 MOEX macro features exist in `_compute_cbr_features()`, `_compute_brent_*()`, `_compute_macro_features()` (HIGH confidence, direct read)
- `scripts/auto_ml_research.py` — `_MOEX_LOOKBACK_DAYS = 730`, `_MOEX_MAX_FEATURES = 10`, `_MOEX_ATR_UPLIFT = 1.2` already stubbed (HIGH confidence, direct read)
- `src/finalayze/core/experiment_manager.py` — `create()`, `record_verdict()` public API confirmed (HIGH confidence, direct read)
- `config/segments.py` — `ru_blue_chips`, `ru_energy`, `ru_tech`, `ru_finance` symbols confirmed (HIGH confidence, direct read)
- `pyproject.toml` — all library versions confirmed (HIGH confidence, direct read)

---

*Stack research for: Finalayze v9.0 ML AutoResearch & MOEX Adaptation*
*Researched: 2026-04-13*

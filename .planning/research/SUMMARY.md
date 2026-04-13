# Project Research Summary

**Project:** Finalayze v9.0 — ML AutoResearch & MOEX Adaptation
**Domain:** ML experiment automation + multi-market data integration (equity trading)
**Researched:** 2026-04-13
**Confidence:** HIGH

## Executive Summary

Finalayze v9.0 extends an existing, battle-tested ML autoresearch pipeline (`auto_ml_research.py`) to support Russian equity segments (MOEX) and adds three new search strategies (ensemble weight optimization, automatic feature engineering, cross-segment transfer). The recommended approach is purely additive: zero new packages, all gaps close by wiring existing infrastructure components (`TinkoffFetcher`, `CBRFetcher`, `MoexISSFetcher`, `ExperimentManager`) into the research script. The existing codebase already ships MOEX macro features in `technical.py`, MOEX segment definitions in `config/segments.py`, adaptive quality gates in `quality_gates.py`, and a sync-safe gRPC bridge in `TinkoffFetcher._run_async()` — none of these need to be built, only wired.

The critical dependency ordering is strict: MOEX segment symbols -> data adapter -> macro features -> adaptive quality gates -> ExperimentManager integration -> new search strategies. The three new strategies (ensemble weights, feature engineering, cross-segment transfer) cannot be meaningfully evaluated until MOEX data flows end-to-end and quality gates produce reliable verdicts on 3+ walk-forward folds. Phases 1-4 are a sequential prerequisite chain; phases 5-6 can be parallelized after that chain is stable.

The primary risks are data-integrity and statistical validity: look-ahead bias when joining macro series (always shift by 1 day), the T-Bank sandbox endpoint returning empty historical candles (always use `sandbox=False` for training), and walk-forward fold collapse on 730-day MOEX history (MOEX-specific fold constants required — current US constants produce 0-1 folds). Ensemble weight optimization and combinatorial feature engineering carry secondary overfitting risk on small MOEX datasets and must be gated behind hard constraints (weight simplex cap at 0.5, candidate feature cap at `n_samples / 20`).

## Key Findings

### Recommended Stack

The stack for v9.0 is entirely the existing tech stack — no new packages are introduced. The key insight from STACK.md is that `TinkoffFetcher` already solves the sync-from-script problem via a self-managed background event loop (`_run_async()` + `run_coroutine_threadsafe()`). Callers in sync scripts construct `TinkoffFetcher` without a `grpc_loop` argument and the daemon thread fallback activates automatically. `CBRFetcher` and `MoexISSFetcher` are synchronous HTTP clients (httpx-based) and require no bridging. `ExperimentManager` is fully synchronous file I/O.

**Core technologies:**
- `t-tech-investments` (T-Bank gRPC SDK): MOEX candle fetch — already handles sync-async bridge via `_run_async()`
- `pandas-ta`: Automatic feature engineering experiments — existing indicator library, no tsfresh/featuretools needed
- `XGBoost + LightGBM + CatBoost + meta-learner`: Ensemble; CatBoost especially valuable for small MOEX tabular datasets
- `scikit-learn`: MI-based feature selection (`select_features_efficient()`) and quality gate metrics
- `ExperimentManager` (internal): Experiment persistence and verdict lifecycle — file-based CRUD at Layer 0
- `CBRFetcher` + `MoexISSFetcher`: Sync HTTP clients for macro data (CBR key rate, IMOEX index, FX rates)

**Explicitly avoided:** `tsfresh` (700+ features vs 10-15 budget), `featuretools` (entity modeling overkill), `nest_asyncio` (not needed — `_run_async()` handles it), Optuna in autoresearch loop (nested optimization obscures signal), separate MLflow/experiment-tracking DB (ExperimentManager + JSONL covers all needs).

### Expected Features

**Must have (table stakes):**
- MOEX segment symbols in `_SEGMENT_SYMBOLS` — prerequisite for any MOEX run; currently absent
- MOEX data adapter (TinkoffFetcher path in `_prepare_data()`) — pipeline is US-only without this
- MOEX macro features wired into `build_full_dataset()` — already computed in `technical.py`, gap is fetch-and-wire
- Adaptive signal_count quality gate — fixed `_MIN_SIGNALS=50` blocks all MOEX folds (15-30 signals per fold is normal)
- MOEX-tuned walk-forward fold parameters — current constants yield 0-1 folds on 730-day history
- MOEX-tuned default hyperparameters — `max_depth=3` (not 5) to prevent overfitting
- ExperimentManager integration — hypothesis tracking, lifecycle verdict (ACCEPTED/REJECTED/INCONCLUSIVE)

**Should have (differentiators):**
- Ensemble weight optimization strategy — XGB:LGBM:CatBoost simplex search (~9-12 candidate configs)
- Hypothesis-linked verdicts with per-strategy experiment IDs — surgical comparison across strategy types
- Cross-segment transfer strategy — validates US-learned features on MOEX, avoids redundant search

**Defer (post-MVP):**
- Automatic feature engineering strategy — high complexity, high overfitting risk on 730-day MOEX data; defer until MOEX baseline is stable and feature vocabulary is established
- LSTM in autoresearch — 10-50x slower per fold; train separately via `train_models.py` after tree ensembles are validated

### Architecture Approach

All changes are concentrated in two files: `scripts/auto_ml_research.py` (~280 lines added) and `src/finalayze/ml/training/quality_gates.py` (parametrized `min_signals` + `_MOEX_MIN_SIGNALS=15` constant). No new source modules, no new layers, no new processes. The data flow branches on segment prefix (`us_` vs `ru_`): MOEX segments route to `TinkoffFetcher` for candles and a new `_fetch_moex_macro()` function that constructs `MoexMarketData` and passes it through `build_full_dataset()` -> `build_triple_barrier_dataset()` -> `compute_features()` where the 10 existing MOEX macro features activate. ExperimentManager integration is opt-in via `--experiment-id` CLI flag, keeping the script non-breaking for existing invocations.

**Major components:**
1. `_fetch_moex_candles()` / `_fetch_moex_macro()` — new functions in autoresearch script; sync-safe via existing fetcher patterns
2. `_prepare_data()` routing branch — dispatches `us_` to yfinance path, `ru_` to TinkoffFetcher path
3. `evaluate_fold(min_signals=...)` — parametrized quality gate that preserves US thresholds while accommodating MOEX dataset sizes
4. `generate_ensemble_weight_experiments()` / `generate_transfer_experiments()` — new strategy generators
5. ExperimentManager opt-in wiring — one experiment per `run_research_loop()` invocation; JSONL log retained as parallel audit trail

### Critical Pitfalls

1. **TinkoffFetcher requires InstrumentRegistry** — constructing without properly initialized registry causes silent empty-candle returns for all MOEX symbols. Use `build_default_registry()`; assert token present; raise loudly if 0 candles returned after fetch loop.

2. **sandbox=False mandatory for training** — `TinkoffFetcher` defaults to `sandbox=True` which has no historical candle data. All training scripts must explicitly set `sandbox=False`. Add `assert not fetcher._sandbox` as defensive check.

3. **Look-ahead bias in macro feature join** — CBR key rate (announced ~13:30 Moscow time) and same-day USDRUB/IMOEX closes must be `shift(1)` before joining to feature vectors. Write unit test with synthetic macro series before merging any macro feature code.

4. **Walk-forward fold collapse on 730-day MOEX history** — current US fold constants (`_WF_TRAIN_MONTHS=12`, `_WF_STEP_MONTHS=3`) produce 0-1 folds. MOEX-specific constants (`MOEX_WF_TRAIN_MONTHS=8`, `MOEX_WF_TEST_MONTHS=3`, `MOEX_WF_STEP_MONTHS=2`, `MOEX_PURGE_GAP=21`) yield 3-4 folds. Add guard: abort if `len(folds) < 3`.

5. **ExperimentManager ID conflicts from per-config file creation** — do not create one `.md` file per internal research config (100+ names like `ablate-rsi`). Create one experiment per `run_research_loop()` invocation with ID pattern `{segment}_{strategy}_{YYYYMMDD_HHMM}`; sub-results go to JSONL.

6. **Ensemble weight overfitting on small MOEX folds** — 30-80 sample validation sets cannot distinguish optimization signal from noise. Default to equal weights (1/3 each) unless 4+ independent folds available. Constrain any single model weight to <= 0.5.

7. **Combinatorial feature engineering explosion** — hard cap at `n_samples / 20` candidates (~36 for MOEX). Generate only domain-motivated combinations; run permutation test before selection.

## Implications for Roadmap

Based on research, the critical path is sequential for phases 1-4 (each depends on the prior), then phases 5-6 can proceed in parallel.

### Phase 1: MOEX Data Adapter
**Rationale:** Nothing runs on MOEX without this. Prerequisite for all subsequent phases. Also catches the two most common silent-failure pitfalls (no InstrumentRegistry, sandbox endpoint) before they contaminate later work.
**Delivers:** `--segment ru_blue_chips` loads data without error; candle counts printed; TinkoffFetcher constructs correctly with `build_default_registry()`; `sandbox=False` assertion in place.
**Addresses:** MOEX segment symbols in `_SEGMENT_SYMBOLS`; `_fetch_moex_candles()` function; `_prepare_data()` routing branch; MOEX-specific walk-forward fold constants.
**Avoids:** Pitfall 1 (InstrumentRegistry), Pitfall 9 (sandbox endpoint), Pitfall 3 (fold collapse — verify empirically here).

### Phase 2: MOEX Macro Features
**Rationale:** 10 MOEX macro features already exist in `technical.py` — this phase only wires the fetch-and-pass-through. Must come before quality gate tuning because macro features affect the feature distribution that quality gates evaluate. The look-ahead bias pitfall must be resolved here or it silently corrupts all future models.
**Delivers:** `usdrub_zscore_60d`, `brent_zscore_60d`, `cbr_rate_level`, etc. are non-zero in MOEX feature dicts; unit test verifies `shift(1)` alignment; Brent fixture in `tests/fixtures/` for CI determinism.
**Uses:** `CBRFetcher`, `MoexISSFetcher`, `YFinanceFetcher` (for Brent `BZ=F`); `MoexMarketData` schema; `MarketDataLoader` reuse pattern.
**Avoids:** Pitfall 2 (look-ahead bias), Pitfall 8 (Brent non-determinism in CI).

### Phase 3: Adaptive Quality Gates
**Rationale:** Fixed `_MIN_SIGNALS=50` blocks all MOEX experiments. Must be fixed before any experiment results can be trusted. Independent of Phase 2 at the code level but validated with real MOEX data from Phase 1.
**Delivers:** MOEX experiments produce 3+ valid folds; signal_count gate parametrized with `_MOEX_MIN_SIGNALS=15`; degenerate predictor unit test (all-BUY on 60%-positive class fails class_balance gate); fold guard in place.
**Avoids:** Pitfall 3 (zero valid folds), Pitfall 4 (degenerate predictor bypass).

### Phase 4: ExperimentManager Integration
**Rationale:** Research tracking needed before adding new strategies — otherwise ensemble weight and cross-segment experiments have no persistent hypothesis record. Integration is opt-in and non-breaking for existing JSONL workflows.
**Delivers:** `--experiment-id {segment}_{strategy}_{timestamp}` creates experiment file with metrics and verdict; two concurrent segment runs produce non-overlapping files; JSONL log retained in parallel.
**Avoids:** Pitfall 5 (ExperimentManager ID conflicts).

### Phase 5: Ensemble Weight Optimization Strategy
**Rationale:** Medium value, medium complexity; depends on working MOEX baseline with reliable fold counts (Phases 1-3). Simpler than feature engineering and has bounded search space.
**Delivers:** `--strategy ensemble_weights` searches XGB:LGBM:CatBoost simplex (~9-12 configs); equal weights enforced when n_folds < 4; optimization gain logged separately for auditability.
**Avoids:** Pitfall 6 (ensemble weight overfitting).

### Phase 6: New Search Strategies (Feature Engineering + Cross-Segment Transfer)
**Rationale:** Both strategies are high-complexity and depend on stable MOEX baselines from Phases 1-4. They can be developed in parallel with each other. Feature engineering carries highest overfitting risk and needs the hardest constraints from the start. Cross-segment transfer requires validated US experiment history in JSONL.
**Delivers:** `--strategy feature_engineering` with candidate cap at `n_samples / 20` and permutation test; `--strategy cross_segment_transfer` with JS divergence check excluding VIX features from MOEX transfer set.
**Avoids:** Pitfall 7 (cross-segment distribution shift), Pitfall 10 (combinatorial feature explosion).

### Phase Ordering Rationale

- Phases 1->2->3->4 are a strict dependency chain: data adapter -> macro features -> quality gates -> experiment tracking. No later phase produces trustworthy results without all prior phases stable.
- Phase 3 modifies `quality_gates.py` which is shared infrastructure; must be completed before phases 5-6 produce gate-dependent verdicts.
- Phases 5 and 6 depend on the same Phases 1-4 foundation but are independent of each other and can be parallelized.
- Automatic feature engineering is explicitly lower priority within Phase 6 because its overfitting risk on 730-day MOEX data is high and expected signal quality is low until a MOEX-specific feature vocabulary is established.

### Research Flags

Phases needing attention during planning:
- **Phase 3:** MOEX-specific walk-forward fold parameters need empirical calibration — recommended constants are analytically derived but must be verified against actual candle volumes per `ru_*` segment.
- **Phase 6 (feature engineering):** Domain-motivated combination rules and permutation test threshold need explicit design decisions before implementation begins.

Phases with standard patterns (low planning uncertainty):
- **Phase 1:** Follows established `train_models.py` MOEX branch pattern exactly — replicate, do not reinvent.
- **Phase 2:** Two-line wire-up in `build_full_dataset()`; macro fetch pattern mirrors existing `MarketDataLoader._load_moex()`.
- **Phase 4:** ExperimentManager API already defined and tested; integration is additive with opt-in flag.
- **Phase 5:** Ensemble weight grid is bounded and well-defined; straightforward implementation.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All findings from direct codebase reads; zero new packages; all components already installed and tested |
| Features | HIGH | Gap analysis grounded in diff between `train_models.py` (has MOEX) and `auto_ml_research.py` (lacks MOEX) |
| Architecture | HIGH | Component boundaries, integration points, and data flow verified by direct file inspection of all affected modules |
| Pitfalls | HIGH | All 10 pitfalls grounded in known codebase constraints, prior v6.0/v7.0 lessons, and MEMORY.md |

**Overall confidence:** HIGH

### Gaps to Address

- **CBRFetcher / MoexISSFetcher sync confirmation:** STACK.md notes these are httpx-based but it is not confirmed whether they expose sync wrappers or require `asyncio.run()`. Verify at Phase 2 implementation start; `asyncio.run()` is acceptable for a CLI script not inside an existing event loop.
- **Brent fixture for CI:** Phase 2 must create `tests/fixtures/brent_candles.json` before any macro feature code reaches CI — currently absent.
- **Actual MOEX candle volume per segment:** Fold parameter recommendations assume ~730 trading days for `ru_blue_chips`. Actual count may differ by segment (e.g., `ru_tech` symbols with shorter listing history). Verify empirically on first Phase 1 run.
- **MarketDataLoader DB dependency:** ARCHITECTURE.md recommends reusing `MarketDataLoader.load()` for macro fetch. Confirm it can be instantiated in a script context without a running database connection (file-cached paths may avoid DB, but not confirmed).

## Sources

### Primary (HIGH confidence)

- `scripts/auto_ml_research.py` — current baseline; US-only; 4 existing strategies; fold constants; MOEX stubs already present
- `scripts/train_models.py` — MOEX-aware pattern to replicate: `_is_moex_segment()`, `_get_lookback_days()`, MOEX-tuned hyperparameters
- `src/finalayze/data/fetchers/tinkoff_data.py` — `_run_async()` self-managed loop; `sandbox=True` default; `GRPC_DNS_RESOLVER=native` at module level
- `src/finalayze/ml/training/quality_gates.py` — `_MIN_SIGNALS=50`; `_MAX_ACCURACY_THRESHOLD=0.55`; `_dynamic_brier_threshold(n_eff)`
- `src/finalayze/ml/features/technical.py` — 10 MOEX macro features (usdrub_zscore_60d, brent_zscore_60d, real_rate_zscore, cbr_rate_level, cbr_rate_delta, etc.)
- `src/finalayze/core/experiment_manager.py` — `create_experiment()`, `link_result()`, `record_verdict()` public API; flat file namespace
- `src/finalayze/core/schemas.py` — `MoexMarketData`, `MarketContext`, `ExperimentState`, `SuccessCriteria` schemas
- `src/finalayze/ml/models/ensemble.py` — `model_weights: dict[str, float]` parameter; `_get_model_weight()` resolver
- `config/segments.py` — `ru_blue_chips`, `ru_energy`, `ru_tech`, `ru_finance` symbols
- `src/finalayze/data/loader.py` — `MarketDataLoader._load_moex()` as macro data reuse reference
- `pyproject.toml` — all library versions confirmed installed
- `MEMORY.md` — gRPC C-ares resolver fix, SDK target override, sandbox vs live endpoint distinction
- `.planning/PROJECT.md` — v9.0 active requirements, Data Sources, Known Issues

---
*Research completed: 2026-04-13*
*Ready for roadmap: yes*

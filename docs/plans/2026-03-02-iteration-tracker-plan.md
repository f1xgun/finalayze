# Iteration Tracker — Implementation Plan

**Design:** [2026-03-02-iteration-tracker-design.md](2026-03-02-iteration-tracker-design.md)
**Date:** 2026-03-02 | **Status:** Draft

---

## PR Strategy

| PR | Contents | Dependency |
|----|----------|------------|
| **PR-1** | Backtest fixes: WF Sharpe per-fold, WF snapshots, MC bar-level bootstrap | none |
| **PR-2** | Decision logging: richer Signal.features, per-model proba, JournalingCombiner extension | none |
| **PR-3** | Iteration tracker: schemas, tracker class, CLI scripts, acceptance gates | PR-1 + PR-2 |

PR-1 and PR-2 can be developed in parallel. PR-3 depends on both.

---

## PR-1: Backtest Metric Fixes

### Task 1.1 — Fix WF Sharpe: per-fold aggregation

**Files:** `src/finalayze/backtest/walk_forward.py`, `tests/unit/backtest/test_walk_forward.py`

**Steps:**
1. Write failing test: `test_wf_sharpe_per_fold_no_splicing` — create 2 folds where
   fold 1 ends at equity $120k and fold 2 starts at $100k. Assert that the splice
   discontinuity does NOT inflate the Sharpe.
2. Modify `WalkForwardOptimizer.run()`:
   - Compute Sharpe per fold from that fold's equity snapshots only
   - Store `per_fold_sharpes: list[float]` and `per_fold_trade_counts: list[int]`
   - Aggregate: `oos_sharpe = sum(s * n for s, n in zip(sharpes, counts)) / sum(counts)`
3. Add `per_fold_sharpes` and `per_fold_trade_counts` fields to `WalkForwardResult`
4. Verify existing WF tests still pass (values will change — update expected values)

### Task 1.2 — Add snapshots to WalkForwardResult

**Files:** `src/finalayze/backtest/walk_forward.py`, `tests/unit/backtest/test_walk_forward.py`

**Steps:**
1. Write failing test: `test_wf_result_includes_oos_snapshots` — run WF and assert
   `result.oos_snapshots` is a non-empty list of `PortfolioState`
2. Add `oos_snapshots: list[PortfolioState]` field to `WalkForwardResult`
3. In `WalkForwardOptimizer.run()`, collect snapshots from each fold's
   `engine.run()` call and extend into `all_snapshots`
4. Compute `oos_max_drawdown_pct` from bar-level snapshots instead of per-trade PnL

### Task 1.3 — Fix MC bootstrap to use bar-level returns

**Files:** `src/finalayze/backtest/monte_carlo.py`, `tests/unit/backtest/test_monte_carlo.py`

**Steps:**
1. Write failing test: `test_bootstrap_accepts_bar_returns` — pass bar-level daily
   returns instead of per-trade PnL, assert valid `BootstrapResult`
2. Add `bootstrap_from_snapshots(snapshots, ...)` function that:
   - Extracts daily returns from equity snapshots
   - Resamples daily returns (not trades)
   - Computes Sharpe correctly with `sqrt(252)` on daily returns
3. Keep existing `bootstrap_metrics()` for backward compatibility, mark deprecated
4. Update tests

### Task 1.4 — Add Sortino and Calmar to PerformanceAnalyzer.analyze()

**Files:** `src/finalayze/backtest/performance.py`, `tests/unit/backtest/test_performance.py`

**Steps:**
1. Write failing test: `test_analyze_includes_sortino_calmar`
2. Add `sortino_ratio` and `calmar_ratio` fields to `BacktestResult` (optional Decimal)
3. Compute them inside `analyze()` using existing `sortino_ratio()` and `calmar_ratio()`
   methods (already on `PerformanceAnalyzer`, just not called in `analyze()`)
4. Add `turnover_ratio` field: total traded notional / average equity

---

## PR-2: Decision Logging Enhancement

### Task 2.1 — Enrich Signal.features in MomentumStrategy

**Files:** `src/finalayze/strategies/momentum.py`, `tests/unit/strategies/test_momentum.py`

**Steps:**
1. Write failing test: signal features dict includes `rsi_value`, `macd_hist`,
   `sma_trend`, `adx_value`, `volume_ratio` keys
2. In `generate_signal()`, populate `features` dict with intermediate indicator values
   before the threshold checks
3. When returning `None`, still set features on a module-level or use the existing
   `JournalingStrategyCombiner` approach — add a `last_skip_reason: str | None`
   attribute that the combiner reads

### Task 2.2 — Enrich Signal.features in MeanReversionStrategy

**Files:** `src/finalayze/strategies/mean_reversion.py`, `tests/unit/strategies/test_mean_reversion.py`

**Steps:**
1. Write failing test: features include `bb_pct_b`, `rsi_value`, `squeeze_active`,
   `band_distance`
2. Populate features dict with Bollinger Band indicators and filter states

### Task 2.3 — Expose per-model probabilities from EnsembleModel

**Files:** `src/finalayze/ml/models/ensemble.py`, `tests/unit/ml/test_ensemble.py`

**Steps:**
1. Write failing test: after `predict_proba()`, `model.last_model_probas` is a dict
   with keys like `"xgboost"`, `"lightgbm"`, `"lstm"` and float values
2. Add `last_model_probas: dict[str, float]` attribute to `EnsembleModel`
3. In `predict_proba()`, store each sub-model's output in the dict before averaging
4. Switch from stdlib `logging` to `structlog` while we're here

### Task 2.4 — Extend JournalingStrategyCombiner

**Files:** `src/finalayze/backtest/journaling_combiner.py`, `tests/unit/backtest/test_journaling_combiner.py`

**Steps:**
1. Write failing test: after `generate_signal()`, combiner exposes `last_features`
   dict aggregating all strategy features
2. Add `last_features: dict[str, float]` that merges features from each strategy's
   Signal (prefixed by strategy name, e.g., `momentum.rsi_value`)
3. When ensemble model is among strategies, also capture `last_model_probas` from
   the MLStrategy's underlying ensemble

### Task 2.5 — Extend DecisionRecord with richer data

**Files:** `src/finalayze/backtest/decision_journal.py`, `src/finalayze/backtest/engine.py`

**Steps:**
1. Write failing test: `DecisionRecord` includes `strategy_features` and
   `model_probas` fields
2. Add optional fields to `DecisionRecord`:
   - `strategy_features: dict[str, float] | None`
   - `model_probas: dict[str, float] | None`
3. In `BacktestEngine._record_decision()`, populate from combiner's new attributes

---

## PR-3: Iteration Tracker

### Task 3.1 — Iteration schemas in core/schemas.py

**Files:** `src/finalayze/core/schemas.py`, `tests/unit/core/test_schemas.py`

**Steps:**
1. Write tests for `IterationMetrics`, `GateResult`, `IterationMetadata`,
   `IterationComparison` — validation, serialization, frozen behavior
2. Add the 4 Pydantic models to `core/schemas.py` as specified in the design doc
3. Ensure `schema_version` defaults to 1

### Task 3.2 — IterationTracker class

**Files:** `src/finalayze/backtest/iteration_tracker.py`, `tests/unit/backtest/test_iteration_tracker.py`

**Steps:**
1. Write failing tests:
   - `test_snapshot_git_context` — captures SHA, dirty flag, describe
   - `test_snapshot_config` — hashes BacktestConfig + strategy YAMLs
   - `test_compute_metrics` — given trades + snapshots, returns IterationMetrics
   - `test_evaluate_gates` — given metrics + baseline, returns gate results + verdict
   - `test_save_iteration` — writes metadata.json, summary.json, appends history.jsonl
   - `test_atomic_write` — interrupted write doesn't corrupt files
   - `test_load_iteration` — reads back saved iteration
   - `test_compare_iterations` — computes correct deltas
2. Implement `IterationTracker` class:

```python
class IterationTracker:
    def __init__(self, results_root: Path): ...

    def snapshot_context(self) -> dict:
        """Capture git SHA, describe, dirty, config hash."""

    def compute_metrics(
        self,
        wf_result: WalkForwardResult,
        trades: list[TradeResult],
        snapshots: list[PortfolioState],
        segment_trades: dict[str, list[TradeResult]],
        mc_result: BootstrapResult,
        journal: DecisionJournal | None = None,
    ) -> IterationMetrics: ...

    def evaluate_gates(
        self,
        metrics: IterationMetrics,
        baseline: IterationMetrics | None,
    ) -> tuple[list[GateResult], str]:
        """Returns (gate_results, verdict)."""

    def save(self, metadata: IterationMetadata) -> Path:
        """Atomic write to results/iterations/<name>/."""

    def load(self, name: str) -> IterationMetadata:
        """Load iteration by name."""

    def load_latest(self) -> IterationMetadata | None:
        """Load most recent iteration from history.jsonl."""

    def compare(
        self, current: str, baseline: str
    ) -> IterationComparison: ...

    def list_iterations(self) -> list[dict]: ...
```

3. Gate implementation follows design doc: 3 safety gates (individually REJECT),
   6 calibration gates (2+ = REJECT, 1 = WARN)

### Task 3.3 — run_iteration.py CLI script

**Files:** `scripts/run_iteration.py`

**Steps:**
1. Implement CLI with argparse:
   - `--name` (required): iteration name
   - `--description` (required): what changed
   - `--baseline` (optional, default "latest"): baseline iteration name
   - `--output` (optional, default `results/iterations/`): output root
   - `--segments` (optional): comma-separated segment IDs (default: all)
   - `--start-date` / `--end-date` (optional): backtest date range
   - `--dry-run` (optional): compute metrics without saving
2. Flow:
   - Snapshot git context
   - Load strategy YAML configs per segment
   - For each segment: fetch candles → run WalkForwardOptimizer → collect trades + snapshots
   - Run Monte Carlo bootstrap on bar-level returns
   - Compute IterationMetrics
   - Load baseline (latest or specified)
   - Evaluate gates → verdict
   - Save iteration (atomic writes)
   - Print comparison table to terminal
3. Add `sys.path` insert for config/ access
4. Handle errors gracefully (partial results saved with error flag)

### Task 3.4 — compare_iterations.py CLI script

**Files:** `scripts/compare_iterations.py`

**Steps:**
1. Implement CLI: `compare_iterations.py <baseline> <current>`
2. Load both iterations from `results/iterations/`
3. Compute deltas for all metrics
4. Re-evaluate gates with current metrics vs baseline
5. Print formatted comparison table with flags

### Task 3.5 — list_iterations.py CLI script

**Files:** `scripts/list_iterations.py`

**Steps:**
1. Implement CLI: `list_iterations.py [--verdict PASS|WARN|REJECT]`
2. Read `history.jsonl` line by line
3. Print summary table: name, date, git SHA, verdict, WF Sharpe, max DD, trade count

### Task 3.6 — Integration test

**Files:** `tests/integration/backtest/test_iteration_tracker_integration.py`

**Steps:**
1. End-to-end test: create synthetic candles → run iteration tracker →
   save → load → compare with modified parameters → verify gate logic
2. Test history.jsonl append behavior (2 sequential iterations)
3. Test atomic write (verify no partial files on simulated error)

---

## Dependency Layer Update

**Files:** `docs/architecture/DEPENDENCY_LAYERS.md`

Add `backtest/` as Layer 5.5:

```
Layer 5.5: Backtest     backtest/
  May import from: L0–L5
  May NOT import from: L6 (api/, dashboard/)
```

---

## Estimated Test Count

| PR | New tests | Modified tests |
|----|-----------|----------------|
| PR-1 | ~12 | ~8 (updated expected values) |
| PR-2 | ~15 | ~5 |
| PR-3 | ~25 | ~0 |
| **Total** | **~52** | **~13** |

---

## Execution Order

```
PR-1 ──┐
       ├──→ PR-3
PR-2 ──┘
```

PR-1 and PR-2 are independent and can be developed in parallel (separate worktrees).
PR-3 depends on both being merged.

---

## Verification Criteria

- [ ] `uv run pytest` — all tests pass
- [ ] `uv run ruff check . && uv run ruff format --check .` — clean
- [ ] `uv run mypy src/` — no new errors
- [ ] Run `scripts/run_iteration.py --name test-baseline --dry-run` on existing data
- [ ] Run a second iteration and verify comparison table renders correctly
- [ ] Verify `results/iterations/history.jsonl` is append-only with valid JSON lines
- [ ] Verify atomic write: kill script mid-run, confirm no corrupt partial files

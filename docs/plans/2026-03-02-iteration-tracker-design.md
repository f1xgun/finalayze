# Iteration Tracker — Design Document

**Version:** 1.0 | **Date:** 2026-03-02 | **Status:** Draft

---

## Goal

Build a system that tracks every set of changes to the trading platform as a numbered
"iteration," runs standardized backtests, saves metrics + debug logs, and auto-compares
against the previous iteration. This lets the operator see which changes improved the
system and which hurt it, building an evidence-based optimization history.

---

## Architecture

```
scripts/
    run_iteration.py          ← CLI: snapshot context → run backtest → save → compare
    compare_iterations.py     ← CLI: compare any two iterations
    list_iterations.py        ← CLI: show iteration history with key metrics

src/finalayze/
    core/schemas.py           ← IterationMetadata, IterationComparison (L0)
    backtest/
        iteration_tracker.py  ← IterationTracker: orchestrate run + persist + compare
        walk_forward.py       ← FIX: per-fold Sharpe aggregation, add snapshots to result
        monte_carlo.py        ← FIX: bootstrap on bar-level returns, not per-trade PnL
        performance.py        ← ADD: Sortino, Calmar, turnover-adjusted return
        journaling_combiner.py← EXTEND: capture per-model ML proba breakdown

results/iterations/
    history.jsonl             ← Append-only index (one JSON line per iteration)
    <YYYY-MM-DD-name>/
        metadata.json         ← Full snapshot: git, config, metrics, flags, gate results
        summary.json          ← Per-symbol backtest results (reuses existing format)
        comparison.json       ← Delta vs baseline iteration
        decision_journal/     ← Per-symbol .jsonl files
            AAPL.jsonl
            MSFT.jsonl
            ...
```

---

## Schemas (in `core/schemas.py`)

```python
class GateResult(BaseModel):
    """Result of a single acceptance gate."""
    model_config = ConfigDict(frozen=True)

    name: str
    gate_type: str  # "safety" | "calibration"
    passed: bool
    value: float
    threshold: float
    message: str


class IterationMetrics(BaseModel):
    """All tracked metrics for one iteration."""
    model_config = ConfigDict(frozen=True)

    # Primary (6)
    wf_sharpe: float                    # Per-fold weighted mean
    wf_max_drawdown: float              # Peak-to-trough from bar-level snapshots
    profit_factor: float
    calmar_ratio: float
    trade_count: int
    avg_hold_bars: float
    segment_pnl_share: dict[str, float] # segment_id → % of total PnL

    # Secondary (6)
    sortino_ratio: float
    win_rate_by_segment: dict[str, float]
    information_ratio: float | None
    mc_5th_pct_sharpe: float            # Bootstrap 5th-percentile
    model_disagreement: float           # std(xgb_prob, lgbm_prob, lstm_prob)
    turnover_adjusted_return: float

    # Diagnostic
    gross_sharpe: float                 # Before transaction costs
    net_sharpe: float                   # After transaction costs
    param_stability_cv: float           # CV of optimal params across WF folds
    per_model_proba_mean: dict[str, float]  # model_name → mean probability


class IterationMetadata(BaseModel):
    """Complete snapshot of one iteration."""
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    name: str
    description: str
    created_at: datetime
    git_describe: str           # `git describe --dirty`
    git_sha: str
    git_dirty: bool
    config_hash: str            # SHA-256 of BacktestConfig + strategy YAML content
    strategy_configs: dict[str, Any]  # segment_id → YAML preset content
    backtest_config: dict[str, Any]   # BacktestConfig as dict
    metrics: IterationMetrics
    gate_results: list[GateResult]
    verdict: str                # "PASS" | "WARN" | "REJECT"
    tags: list[str] = []


class IterationComparison(BaseModel):
    """Delta between two iterations."""
    model_config = ConfigDict(frozen=True)

    current: str                # iteration name
    baseline: str               # iteration name
    metric_deltas: dict[str, float]  # metric_name → (current - baseline)
    gate_results: list[GateResult]
    verdict: str
```

---

## Metric Framework

### Primary Metrics (always shown)

| # | Metric | Source | Flag |
|---|--------|--------|------|
| 1 | WF Sharpe (per-fold weighted mean) | WalkForwardOptimizer (fixed) | < 0.0 absolute; regress > 0.05 |
| 2 | WF Max Drawdown (peak-to-trough) | Bar-level snapshots per fold | > 15% or increase > 2% |
| 3 | Profit Factor | PerformanceAnalyzer | < 1.3 |
| 4 | Calmar Ratio | Annualised return / max DD | Decline |
| 5 | Trade Count + Avg Hold | Engine output | < 60/fold or strategy-rate-normalized |
| 6 | Segment PnL Share | Per-segment attribution | Max share > 35%; HHI > 0.25 |

### Secondary Metrics (drill-down)

| # | Metric | Source |
|---|--------|--------|
| 7 | Sortino Ratio | PerformanceAnalyzer (new method) |
| 8 | Win Rate by Segment | Per-segment trade results |
| 9 | Information Ratio | PerformanceAnalyzer (existing) |
| 10 | MC 5th-pct Sharpe | Monte Carlo (fixed to use bar-level returns) |
| 11 | Model Disagreement | `std(xgb_prob, lgbm_prob, lstm_prob)` from extended journal |
| 12 | Turnover-Adjusted Return | Total return / annualised turnover ratio |

### Diagnostic Metrics (saved, not gated)

- Gross vs Net Sharpe (before/after transaction costs)
- Parameter stability CV across WF folds
- Per-model mean probability (XGB, LGBM, LSTM)

---

## Acceptance Gates

Gates are split into **safety** (individually sufficient for REJECT) and **calibration**
(accumulate — 2+ calibration failures = REJECT).

### Safety Gates (any failure = REJECT)

| Gate | Rule |
|------|------|
| S1: Absolute Sharpe Floor | WF Sharpe >= 0.0 |
| S2: Drawdown Ceiling | WF max drawdown (peak-to-trough) < 15% |
| S3: MC Robustness Floor | MC 5th-percentile Sharpe >= 0.0 |

### Calibration Gates (2+ failures = REJECT, 1 failure = WARN)

| Gate | Rule |
|------|------|
| C1: Sharpe Regression | WF Sharpe doesn't regress > 0.05 vs baseline |
| C2: Drawdown Regression | WF max drawdown doesn't increase > 2% vs baseline |
| C3: Sample Sufficiency | Trade count >= 60/fold/segment (or rate-normalized) |
| C4: Diversification | No segment > 35% of total PnL |
| C5: MC Robustness Regression | MC 5th-pct Sharpe doesn't decline vs baseline |
| C6: Param Stability | CV of optimal params across WF folds < 0.30 |

### Verdict Logic

```
if any safety gate fails → REJECT
elif calibration_failures >= 2 → REJECT
elif calibration_failures == 1 → WARN
else → PASS
```

---

## Walk-Forward Fixes (prerequisite)

### Fix 1: Per-Fold Sharpe Aggregation

**Current bug:** `walk_forward.py` splices equity series from multiple folds end-to-end,
creating phantom returns at fold boundaries.

**Fix:** Compute Sharpe per fold independently, then aggregate via trade-count-weighted
mean: `sum(sharpe_i * n_trades_i) / sum(n_trades_i)`.

### Fix 2: Add Snapshots to WalkForwardResult

**Current gap:** `WalkForwardResult` has `oos_trades` but no `oos_snapshots`, making
Sortino/Calmar impossible to compute from WF data.

**Fix:** Collect bar-level `PortfolioState` snapshots per fold into
`WalkForwardResult.oos_snapshots: list[PortfolioState]`.

### Fix 3: Monte Carlo on Bar-Level Returns

**Current bug:** `monte_carlo.py` bootstraps per-trade PnL percentages and annualizes
with `sqrt(252)`, but trades have heterogeneous holding periods.

**Fix:** Accept bar-level daily returns (from equity snapshots) as input. Resample those.

---

## Decision Logging Enhancement

### Approach: Extend existing patterns, no layer violations

Per the systems architect's guidance, **do not push DecisionJournal into strategies**.
Instead:

1. **Strategies return richer `Signal.features` dict** — add indicator values, threshold
   comparisons, skip reasons as feature entries (e.g., `rsi_value`, `bb_pct_b`,
   `skip_reason: "rsi_below_threshold"`)

2. **`JournalingStrategyCombiner` already captures `last_signals`** — extend it to also
   capture per-strategy features from the Signal objects

3. **`EnsembleModel.predict_proba`** — return per-model breakdown in addition to mean:
   expose `last_model_probas: dict[str, float]` (keyed by model name)

4. **`BacktestEngine`** — populate `DecisionRecord` with the richer data

5. **Log level policy** (for live trading, not backtest):
   - **INFO**: decisions that reach pre-trade checks (order placed or risk-rejected)
   - **DEBUG**: no-signal skips (strategy returned None)
   - This prevents alert fatigue while keeping audit trail

---

## Storage Design

### Atomic Writes

All JSON writes use temp-file-then-rename:

```python
import tempfile
tmp = tempfile.NamedTemporaryFile(dir=output_dir, suffix=".tmp", delete=False)
tmp.write(data)
tmp.flush()
os.fsync(tmp.fileno())
os.rename(tmp.name, target_path)
```

### history.jsonl (Append-Only Index)

One JSON line per iteration, appended with `open("a")` mode (atomic on POSIX for
single-line writes under pipe buffer size). Contains `name`, `created_at`, `git_sha`,
`verdict`, and primary metrics for quick scanning.

### Git Provenance

- `git_describe`: output of `git describe --dirty --always`
- `git_sha`: output of `git rev-parse HEAD`
- `git_dirty`: True if working tree has uncommitted changes
- `config_hash`: SHA-256 of serialized BacktestConfig + all strategy YAML file contents

---

## CLI Interface

```bash
# Run a new iteration (runs full batch backtest + WF + MC)
uv run python scripts/run_iteration.py \
  --name "add-sentiment-to-momentum" \
  --description "Integrate LLM sentiment score into MomentumStrategy" \
  --baseline latest  # or specific iteration name

# Compare two iterations side-by-side
uv run python scripts/compare_iterations.py \
  baseline-name current-name

# List all iterations with verdict and key metrics
uv run python scripts/list_iterations.py
uv run python scripts/list_iterations.py --verdict PASS  # filter
```

### Output Example

```
┌─────────────────────────────────────────────────────────────────────┐
│ Iteration: add-sentiment-to-momentum                                │
│ Baseline:  phase11-baseline                                         │
│ Git:       a1b2c3d (clean)                                          │
│ Verdict:   PASS                                                     │
├──────────────────────────┬──────────┬──────────┬────────┬──────────┤
│ Metric                   │ Baseline │ Current  │ Delta  │ Flag     │
├──────────────────────────┼──────────┼──────────┼────────┼──────────┤
│ WF Sharpe                │ 0.87     │ 1.02     │ +0.15  │          │
│ Max Drawdown             │ 11.2%    │ 10.8%    │ -0.4%  │          │
│ Profit Factor            │ 1.41     │ 1.58     │ +0.17  │          │
│ Calmar Ratio             │ 1.4      │ 1.7      │ +0.3   │          │
│ Trade Count (total OOS)  │ 847      │ 812      │ -35    │          │
│ Max Segment Share        │ 34%      │ 29%      │ -5%    │          │
├──────────────────────────┼──────────┼──────────┼────────┼──────────┤
│ Sortino                  │ 1.54     │ 1.71     │ +0.17  │          │
│ Information Ratio        │ 0.88     │ 0.95     │ +0.07  │          │
│ MC 5th-pct Sharpe        │ 0.41     │ 0.52     │ +0.11  │          │
│ Model Disagreement       │ 0.12     │ 0.09     │ -0.03  │          │
│ Param Stability CV       │ 0.18     │ 0.15     │ -0.03  │          │
└──────────────────────────┴──────────┴──────────┴────────┴──────────┘
Gates: S1 ✓  S2 ✓  S3 ✓  C1 ✓  C2 ✓  C3 ✓  C4 ✓  C5 ✓  C6 ✓
```

---

## Scope Boundaries

**In scope:**
- IterationTracker class, schemas, CLI scripts
- WF Sharpe fix, MC bootstrap fix, WF snapshots addition
- Richer Signal.features from strategies
- Per-model proba exposure from EnsembleModel
- Acceptance gate logic

**Out of scope (future work):**
- EnsembleCalibrator wiring (separate task)
- Feature importance persistence (separate task)
- LSTM scaler lock fix (separate task)
- Stacking meta-learner for adaptive ensemble weights (separate task)
- CPCV fold Sharpe fix (binary → pnl_pct) (separate task)
- Feature selection inside CPCV folds (separate task)

---

## Dependencies

- Existing: `BacktestEngine`, `PerformanceAnalyzer`, `WalkForwardOptimizer`,
  `DecisionJournal`, `JournalingStrategyCombiner`, `MonteCarloBootstrap`
- No new external dependencies
- Python stdlib: `hashlib`, `tempfile`, `subprocess` (for git commands)

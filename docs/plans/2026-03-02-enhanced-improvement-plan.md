# Enhanced Improvement Plan: Consolidated from 4 Domain Expert Reviews

**Date:** 2026-03-02
**Based on:** Original improvement plan + reviews from Quant Analyst, Risk Officer, ML Engineer, Systems Architect
**Baseline:**
- US: 61.1% WR, +0.07 avg Sharpe, 22/48 positive Sharpe
- MOEX: 48.8% WR, -0.410 avg Sharpe, 0/16 positive Sharpe
- 1253+ tests passing, lint clean

**Consensus Sharpe target:** +0.28 to +0.60 improvement (realistic range from all 4 reviewers), bringing system to 0.35–0.67 avg Sharpe, with 30–35 symbols positive Sharpe.

**Structure:**
- **Phase 0:** Critical pre-requisite fixes (bugs, safety, architecture scaffolding)
- **Phase A:** Quick Wins + Regime Detection
- **Phase B:** Strategy Upgrades + ML Pipeline
- **Phase C:** Portfolio & Infrastructure

---

## Phase 0: Critical Pre-Requisites (Must Complete Before Any Other Work)

### 0.1 Fix `allow_longs` Regime Logic Bug

**Source:** Risk Officer (Required Change #7), Systems Architect (C6)
**Severity:** HIGH
**Agent:** risk-agent
**File:** `src/finalayze/risk/regime.py` (new file, created as part of A.1 but logic must be correct from the start)

**Correct logic:**
```python
if regime == MarketRegime.CRISIS:
    allow_longs = False
elif regime == MarketRegime.ELEVATED and not sma200_above:
    allow_longs = False
else:
    allow_longs = True
```

**Tests:**
- `test_crisis_blocks_longs_regardless_of_sma`
- `test_crisis_below_sma_blocks_longs`
- `test_elevated_below_sma_blocks_longs`
- `test_elevated_above_sma_allows_longs`
- `test_normal_always_allows_longs`

---

### 0.2 Create BacktestConfig Dataclass

**Source:** Systems Architect (I3)
**Agent:** backtest-agent
**File:** `src/finalayze/backtest/engine.py`

```python
@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_cash: Decimal = Decimal(100000)
    max_position_pct: Decimal = Decimal("0.20")
    max_positions: int = 10
    kelly_fraction: Decimal = Decimal("0.5")
    atr_multiplier: Decimal = Decimal("3.0")
    trail_activation_atr: Decimal = Decimal("1.5")
    trail_distance_atr: Decimal = Decimal("1.5")
    target_vol: Decimal | None = None
    profit_target_atr: Decimal = Decimal("3.0")
    max_hold_bars: int | dict[str, int] = 30
    stop_loss_mode: str = "trailing"  # "trailing" | "chandelier"
    trend_filter_enabled: bool = False
    trend_sma_period: int = 200
```

Engine constructor becomes: `BacktestEngine(strategy, config: BacktestConfig, *, circuit_breaker=None, rolling_kelly=None, loss_limits=None, decision_journal=None)`.

**Tests:**
- `test_backtest_config_defaults`
- `test_engine_with_config`
- `test_config_frozen`

**Dependencies:** None. Must complete before A.1, A.4, B.1.

---

### 0.3 Create Unified PositionSizingPipeline

**Source:** Risk Officer (Required Change #1), Systems Architect (I4)
**Severity:** HIGH
**Agent:** risk-agent
**Files:**
- `src/finalayze/risk/position_sizing_pipeline.py` (new)
- `src/finalayze/backtest/engine.py` (refactor `_handle_buy`)

**Key rules:**
- VIX-rank scaling and regime scaling are MUTUALLY EXCLUSIVE
- Vol-targeting bounded [0.25x, 2.0x]
- Regime scaling floor at 0.10 (never zero)
- Correlation scaling bounded [0.30x, 1.0x]
- Steps: KellyStep -> VolTargetStep -> RegimeStep -> CorrelationStep -> HardCaps

```python
class PositionSizingPipeline:
    def compute(self, context: SizingContext) -> Decimal:
        size = context.base_position
        for step in self._steps:
            size = step.adjust(size, context)
        if size < context.min_position_size:
            return Decimal(0)
        return min(size, context.equity * context.max_position_pct)
```

**Tests:**
- `test_pipeline_crisis_scenario` — floor prevents near-zero positions
- `test_pipeline_normal_scenario`
- `test_regime_and_vix_rank_not_both_applied`
- `test_pipeline_steps_order`

**Dependencies:** None. Must complete before A.1, A.3.

---

### 0.4 Create RegimeProvider Protocol

**Source:** Systems Architect (C1/C4)
**Severity:** HIGH
**Agent:** backtest-agent
**Files:**
- `src/finalayze/risk/regime.py` (add protocol)
- `src/finalayze/backtest/engine.py` (use protocol)

```python
class RegimeProvider(Protocol):
    def get_regime(self, candles: list[Candle], bar_index: int) -> RegimeState: ...

class StaticRegimeProvider: ...
class VIXRegimeProvider: ...
class HMMRegimeProvider: ...
```

Engine receives `regime_provider: RegimeProvider | None` and calls per-bar.

**Tests:**
- `test_static_regime_provider`
- `test_vix_regime_provider_transitions`
- `test_engine_uses_dynamic_regime`

**Dependencies:** Depends on 0.2 (BacktestConfig).

---

### 0.5 Fix Momentum Histogram Rising Logic

**Source:** Quant Analyst (Section 9.1)
**Agent:** strategies-agent
**File:** `src/finalayze/strategies/momentum.py` (line 311)

**Fix:**
```python
hist_rising = (
    indicators.current_hist > indicators.prev_hist
    and indicators.current_hist > 0
)
```

**Tests:**
- `test_hist_rising_requires_positive`
- `test_hist_rising_true_when_positive_and_improving`

---

### 0.6 Fix train_models.py Temporal Ordering Bug

**Source:** ML Engineer (Critical #1)
**Severity:** HIGH
**Agent:** ml-agent
**File:** `scripts/train_models.py` (lines 131-149)

**Fix:** Sort concatenated features by timestamp before train/test split.

**Tests:**
- `test_build_dataset_sorted_by_timestamp`
- `test_no_future_leakage_multi_symbol`

---

### 0.7 Increase Minimum Calibration Samples

**Source:** ML Engineer (Critical #2)
**Agent:** ml-agent
**Files:**
- `src/finalayze/ml/models/xgboost_model.py`
- `src/finalayze/ml/models/lightgbm_model.py`

Change `_MIN_CALIBRATION_SAMPLES = 10` to `50`. Fall back to Platt scaling when <50.

---

### 0.8 Add Negative Expectancy Kill Switch

**Source:** Risk Officer (Required Change #2)
**Severity:** HIGH
**Agent:** risk-agent
**File:** `src/finalayze/risk/kelly.py`

If `_compute_raw_kelly()` returns 0 for 3 consecutive full windows (150 trades), return `Decimal(0)` and set `should_halt=True`.

**Tests:**
- `test_kill_switch_activates_after_3_windows`
- `test_kill_switch_resets_on_positive`
- `test_kill_switch_returns_zero_fraction`

---

### 0.9 Add Fail-Closed Cross-Market Exposure Check

**Source:** Risk Officer (Required Change #5)
**Agent:** risk-agent
**File:** `src/finalayze/risk/pre_trade_check.py`

If `len(markets_active) > 1` and `cross_market_exposure_pct is None`, add violation.

---

### 0.10 MOEX Preset Recalibration

**Source:** Quant Analyst (Section 5.1, 6.2, 6.5)
**Severity:** HIGH (0/16 positive Sharpe)
**Agent:** strategies-agent
**Files:** All 4 `ru_*.yaml` presets

| Parameter | Current | New (ru_blue_chips) | New (ru_energy) |
|-----------|---------|---------------------|-----------------|
| bb_std_dev | 2.2 | 2.5 | 3.0 |
| rsi_oversold_mr | 35 | 30 | 30 |
| rsi_overbought_mr | 65 | 70 | 70 |
| momentum.enabled | true | true | **false** |
| min_combined_confidence | 0.35 | 0.40 | 0.45 |
| normalize_mode | firing | **total** | **total** |
| mean_reversion.weight | 0.25 | 0.35 | 0.40 |

---

## Phase A: Quick Wins + Regime Detection (Weeks 1-4)

**Expected Sharpe improvement:** +0.15 to +0.30

### A.1 VIX Regime Filter
**Agent:** risk-agent | **Dependencies:** 0.1, 0.3, 0.4
- `MarketRegime` enum, `RegimeState` dataclass, `compute_regime_state()`
- VIX momentum: `vix_current - vix_5day_sma > 5` upgrades to ELEVATED
- MOEX proxy: IMOEX 20-day realized vol thresholds
- Prometheus gauge: `regime_state{market_id}`

### A.2 Volatility Targeting
**Agent:** risk-agent | **Dependencies:** 0.3
- Integration into `PositionSizingPipeline` as `VolTargetStep` with bounds [0.25x, 2.0x]

### A.3 Hybrid Kelly-VIX Sizing
**Agent:** risk-agent | **Dependencies:** 0.3, A.1
- VIX-rank and regime are MUTUALLY EXCLUSIVE — use regime state only

### A.4 Chandelier Exit
**Agent:** risk-agent + backtest-agent | **Dependencies:** 0.2
- Monotonicity enforced: `new_stop = max(current_stop, candidate_stop)`
- Segment-specific multipliers: US 2.5-3.5x, RU 3.5-4.5x
- `BacktestConfig.stop_loss_mode = "chandelier"`

### A.5 Strategy-Specific Time Exits
**Agent:** backtest-agent | **Dependencies:** 0.2
- momentum=40, mean_reversion=8, pairs=15, event=63, rsi2=5, ml=20

### A.6 HMM Regime Detection
**Agent:** risk-agent + ml-agent | **Dependencies:** 0.4, A.1
- State labeling uses mean AND variance
- `n_init=10`, `n_iter=100, tol=1e-4`
- 3-bar state persistence before regime change
- Min 252 data points for stable estimation
- Separate instances per market (SPY for US, IMOEX for MOEX)

---

## Phase B: Strategy Upgrades + ML Pipeline (Weeks 5-10)

**Expected Sharpe improvement:** +0.08 to +0.20

### B.1 OU Mean Reversion Strategy
**Agent:** strategies-agent | **Dependencies:** 0.4, A.1
- Look-ahead fix: `candles[-(OU_WINDOW+1):-1]` for fitting
- Regime gate: disable in CRISIS, tighten to 2.0σ in ELEVATED
- RU params: 2.0σ entry, 0.3σ exit, 60-bar window

### B.2 Dual Momentum Strategy
**Agent:** strategies-agent | **Dependencies:** A.1
- Confidence: `min(0.95, 0.4 + abs(score) * 1.0)`
- Max 5 positions

### B.3 Triple Barrier Labeling
**Agent:** ml-agent | **Dependencies:** None (can start during Phase A)
- Filter noise: discard vertical hits with `abs(pnl_pct) < 0.5 * ATR_pct`
- Sample weighting by PnL magnitude

### B.4 CPCV Cross-Validation
**Agent:** ml-agent | **Dependencies:** B.3
- Purge window >= 60 (matches feature window)
- XGBoost-only screening, then full ensemble
- Reject if >40% of folds have negative Sharpe

### B.5 Ensemble Calibration Consolidation
**Agent:** ml-agent | **Dependencies:** 0.7
- Remove per-model calibrators, single ensemble-level Platt scaler
- 3-way split: train/calibration/test

### B.6 MOEX-Specific Training Pipeline (NEW)
**Agent:** ml-agent | **Dependencies:** None (can start during Phase A)
- Use TinkoffFetcher for MOEX data
- Additional features: USD/RUB, Brent crude, CBR key rate
- Post-2022 data only, shallow XGBoost (max_depth=3)

### B.7 Feature Selection + Hyperparameter Tuning (NEW)
**Agent:** ml-agent | **Dependencies:** B.6
- Drop features <1% importance, deduplicate >0.85 correlation
- Optuna with temporal CV, Brier score objective

---

## Phase C: Portfolio & Infrastructure (Weeks 11-14)

**Expected Sharpe improvement:** +0.05 to +0.15

### C.1 Correlation-Aware Sizing
**Agent:** risk-agent | **Dependencies:** 0.3
- Pre-trade check #15: max 3 positions correlated >0.7
- Ledoit-Wolf shrinkage for >100 symbols
- Redis caching of correlation matrix

### C.2 Dynamic Strategy Weighting (Adaptive Combiner)
**Agent:** strategies-agent
- Override `generate_signal()` with dynamic weights
- 5% minimum weight floor for paused strategies

### C.3 Additional Pre-Trade Checks (NEW)
**Agent:** risk-agent | **Dependencies:** A.1, B.1, C.1
- Check 12: Regime gate
- Check 13: Parameter freshness for OU/pairs
- Check 14: Intra-bar batch sector limit
- Check 15: Correlation position limit

### C.4 Turnover Budget (NEW)
**Agent:** strategies-agent + backtest-agent
- Max 2 round-trips per symbol per month

### C.5 Rolling Peak-to-Trough Drawdown Monitor (NEW)
**Agent:** risk-agent
- 12% drawdown from peak triggers L2 HALTED
- Rolling, not calendar-reset

### C.6 API Endpoints for New Features (NEW)
**Agent:** api-agent | **Dependencies:** A.1, C.1, C.2
- `GET /api/v1/risk/regime`
- `GET /api/v1/risk/correlation`
- `GET /api/v1/strategies/weights`

---

## Dependency Graph

```
Phase 0 (mostly parallel):
  0.1  0.2  0.3  0.5  0.6  0.7  0.8  0.9  0.10
       |
       0.4

Phase A (after Phase 0):
  A.1 --> 0.1, 0.3, 0.4
  A.2 --> 0.3
  A.3 --> 0.3, A.1
  A.4 --> 0.2
  A.5 --> 0.2
  A.6 --> 0.4, A.1

Phase B (B.3 and B.6 can start during Phase A):
  B.1 --> 0.4, A.1
  B.2 --> A.1
  B.3 --> (none)
  B.4 --> B.3
  B.5 --> 0.7
  B.6 --> (none)
  B.7 --> B.6

Phase C:
  C.1 --> 0.3
  C.2 --> (none)
  C.3 --> A.1, B.1, C.1
  C.4 --> (none)
  C.5 --> (none)
  C.6 --> A.1, C.1, C.2
```

---

## Sharpe Improvement Estimates (Consensus)

| Phase | Consensus |
|-------|-----------|
| 0: Pre-req fixes | +0.05-0.10 (MOEX recal) |
| A: Regime+Exits | +0.15-0.30 |
| B: Strategies+ML | +0.08-0.20 |
| C: Portfolio | +0.05-0.15 |
| **Total** | **+0.28-0.60** |

**Realistic target: Sharpe 0.35-0.67, 30-35 symbols positive Sharpe, max drawdown <15%.**

---

## Items Explicitly NOT Included

- Social media sentiment (weak alpha, high effort)
- Options flow data (no infrastructure)
- Reinforcement learning (sim-to-real gap too large)
- Keltner Channel standalone (decaying alpha since 2016)
- VIX-rank as separate scaling step (redundant with regime state)

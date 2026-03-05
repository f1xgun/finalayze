# Week 3 Design: Combiner Redesign, Risk Calibration & Validation Fixes

**Date:** 2026-03-06
**Baseline:** week2-structural-fix — PF 1.05, WF Sharpe -0.0003, 1198 trades, Verdict: REJECT
**Goal:** Fix the 5 structural root causes that suppress alpha and produce misleading metrics
**Status:** REVIEWED — consensus APPROVE WITH CHANGES from quant/risk/systems agents

## Executive Summary

Three independent domain audits (quant, risk, systems) converged on the same conclusion:
the system generates valid signals at the strategy level but destroys them through
combinatorial cancellation, cascading position-sizing reduction, and miscalibrated exits.
The "WF Sharpe" metric is actually in-sample (no genuine OOS windows exist).

This design addresses 5 root causes in priority order.

---

## Root Cause Analysis

| # | Root Cause | Impact | Domain |
|---|-----------|--------|--------|
| RC1 | Combiner "firing" normalization cancels opposing strategies | ~60% of valid signals lost | Quant |
| RC2 | Position sizing pipeline cascading multiplication | 94%+ reduction → below min_position_size | Risk |
| RC3 | Single ATR stop-loss for all strategies | 1.5-bar avg hold, 12-16% win rate | Risk |
| RC4 | Walk-forward config exceeds data range (3yr+1yr > 2yr) | "WF Sharpe" = in-sample Sharpe | Systems |
| RC5 | DualMomentum/OU ignore YAML params (hardcoded constants) | Tuning those strategies is illusory | Systems |

---

## RC1: Combiner Redesign — Regime-Gated Pools

### Problem

The "firing" normalization computes a weighted average of all strategies that fire.
When trend-following (momentum, dual_momentum) and mean-reverting (mean_reversion,
rsi2_connors) strategies fire simultaneously with opposite directions, signals cancel:

```
momentum     BUY  conf=0.55  weight=0.20 → +0.110
mean_rev    SELL  conf=0.60  weight=0.30 → -0.180
rsi2        SELL  conf=0.50  weight=0.15 → -0.075
                                         --------
net = -0.145 / 0.65 = -0.223 → below |0.38| threshold → NO SIGNAL
```

Each strategy alone would have generated a valid trade. The combiner destroys all of them.

Additionally, `dual_momentum` is BUY-only (line 107: `if score <= 0: return None`),
creating permanent directional bias in the trend pool.

### Design: Regime-Gated Strategy Pools

Replace the single weighted-average combiner with two regime-gated pools routed by
ADX(14). Each pool has internally coherent strategies that don't cancel each other.

```
                      ┌─────────────┐
                      │  ADX(14)    │
                      └──────┬──────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ADX > 30       20 ≤ ADX ≤ 30   ADX < 20
              │              │              │
     ┌────────┴────────┐     │     ┌────────┴────────┐
     │   TREND POOL    │  DOMINANT │    MR POOL       │
     │  momentum       │  POOL    │  mean_reversion  │
     │  dual_momentum  │  WINS    │  rsi2_connors    │
     │                 │          │  ou_mean_rev     │
     └─────────────────┘          │  pairs           │
                                  └──────────────────┘

     NEUTRAL (all regimes): dividend_gap
```

**Regime routing logic:**
- **ADX > 30 (trending):** Only trend pool fires. Normalize within pool.
- **ADX < 20 (range-bound):** Only MR pool fires. Normalize within pool.
- **20 ≤ ADX ≤ 30 (ambiguous):** Both pools run independently, each normalized within
  its own pool. Emit the signal from whichever pool has the higher absolute weighted
  score (dominant-pool-wins). This avoids the agreement-veto dead zone that would
  suppress ~30-40% of trading days.
- **ADX unavailable** (< 28 bars of data): Fall back to ambiguous-zone behavior
  (both pools, dominant wins).

**Pool assignments** (per review consensus):
- `pairs` stays in MR pool (spread mean-reversion, diverges in trends). Matches existing
  `_MR_STRATEGIES` frozenset at combiner.py:36.
- `dividend_gap` is the only neutral strategy (event-driven with defined catalyst).

**Replace Hurst routing** (lines 120-134, `_HURST_BOOST`/`_HURST_SUPPRESS`) with ADX
routing. Hurst R/S (252-bar) is too noisy and slow; ADX(14) responds in 14-28 bars.

**ADX computation:** Extract to `src/finalayze/strategies/adx.py` (parallel to `hurst.py`)
to keep combiner.py free of heavy indicator imports.

**YAML config addition** (per-preset):
```yaml
regime_routing:
  enabled: true
  adx_period: 14
  trend_threshold: 30
  mr_threshold: 20
```

### dual_momentum SELL Signal

Add SELL signals to `dual_momentum.py` when score is meaningfully negative:

```python
# Line 107 replacement:
if score <= -0.05:  # meaningful negative momentum (not noise pullback)
    confidence = min(_MAX_CONFIDENCE, _CONFIDENCE_BASE + abs(score) * _CONFIDENCE_SCALE)
    return Signal(direction=SignalDirection.SELL, confidence=confidence, ...)
if score <= 0:
    return None
```

**Threshold -0.05** (not -0.02): Quant review showed -0.02 triggers on routine 5% monthly
pullbacks, generating 3-4x more SELL than BUY signals. -0.05 requires genuine downtrend
(-12.5% over 1m, or distributed across 3m+6m).

**Signal deduplication:** Add `_SignalState` tracker (matching `MomentumStrategy` pattern)
to prevent emitting identical SELL signals on consecutive bars during downtrends.

---

## RC2: Position Sizing Pipeline Fix

### Problem

The 6-step multiplicative pipeline (`PositionSizingPipeline`) can reduce positions to
near-zero through cascading multiplication:

```
$100K equity × 10% Kelly → $10,000 base
  × 0.50 VolTarget (high-vol asset)  → $5,000
  × 0.30 RegimeStep (bear market)    → $1,500
  × 0.50 CopulaStep (correlated)     → $750
  × 0.50 EVTStep (tail risk)         → $375
  → Below $500 min_position_size → ELIMINATED
```

### Design (revised per risk review)

1. **Pipeline floor relative to base_position, NOT equity:**
   Apply `_PIPELINE_FLOOR_FACTOR = Decimal("0.15")` (15% of base_position). This prevents
   the pipeline from reducing the Kelly-derived base by more than ~85%, while still
   respecting regime-based risk reduction. A floor relative to equity (the original 5%
   proposal) would override crisis-regime scaling — a regime that says "reduce to 25%"
   gets overridden if 5% of equity > 25% of base.

2. **Guarded round-up:** If final size is between `0.5 * min_position_size` and
   `min_position_size`, round up ONLY if the pre-pipeline Kelly base was positive
   (negative-expectancy Kelly should not produce positions):
   ```python
   if context.base_position > context.min_position_size and size < context.min_position_size:
       if size >= context.min_position_size * Decimal("0.5"):
           size = context.min_position_size
   ```

3. **Kelly warm-up:** Always instantiate `RollingKelly` in `BacktestEngine.__init__`.
   `RollingKelly` already handles warm-up correctly: returns `_FIXED_FRACTIONAL = 0.01`
   (1% of equity) when < 10 trades, then blends linearly to pure Kelly between 10-50 trades.

   **DO NOT** change `_DEFAULT_WIN_RATE` / `_DEFAULT_AVG_WIN_RATIO` — risk review showed
   that `win_rate=0.15, avg_win_ratio=2.0` produces **negative expectancy** (E[V] = -0.55),
   Kelly returns 0, and no trades can ever execute. The existing `RollingKelly` graduated
   approach is the correct solution.

### Implementation

**File:** `src/finalayze/risk/position_sizing_pipeline.py`

```python
_PIPELINE_FLOOR_FACTOR = Decimal("0.15")  # 15% of base_position

def compute(self, context: SizingContext) -> Decimal:
    size = context.base_position
    for step in self._steps:
        size = step.adjust(size, context)
    # Pipeline floor: prevent cascading reduction beyond 85% of base
    floor = context.base_position * _PIPELINE_FLOOR_FACTOR
    size = max(size, floor)
    # Guarded round-up: only if Kelly was positive
    if context.base_position > context.min_position_size:
        if context.min_position_size * Decimal("0.5") <= size < context.min_position_size:
            size = context.min_position_size
    if size < context.min_position_size:
        return Decimal(0)
    return min(size, context.equity * context.max_position_pct)
```

**File:** `src/finalayze/backtest/engine.py`

Always create `RollingKelly()` when none provided, instead of using static defaults:
```python
def __init__(self, ..., rolling_kelly=None):
    self._rolling_kelly = rolling_kelly or RollingKelly()
```

---

## RC3: Strategy-Specific Stop-Losses

### Problem

All strategies share a single `stop_atr_multiplier` (typically 3.0 from YAML). This is
appropriate for momentum but too tight for mean-reversion:

- **Mean-reversion** expects price to move further against the signal before reverting.
  A 3.0 ATR stop gets hit on the initial move, causing 1-2 bar holds and low win rates.
- **RSI2 (Connors)** is a 1-5 bar hold strategy. It should use time-exit primarily,
  not ATR stops.

### Design (revised per quant+risk review)

**Strategy-specific multipliers** (stored as config mapping, NOT on Signal schema):

| Strategy | Stop ATR | Max Hold | Rationale |
|----------|---------|----------|-----------|
| momentum | 2.5 | 30 | Trend failure = fast cut |
| dual_momentum | 3.0 | 30 | Monthly rebalance, wider noise |
| mean_reversion | 3.5 | 20 | Survive overshoot, not 4.5 (too wide per review) |
| rsi2_connors | 2.5 | 5 | Short hold + tight stop + fast time exit |
| ou_mean_reversion | 3.5 | 25 | OU-calibrated, moderate |
| pairs | 3.0 | 20 | Spread-based |
| dividend_gap | 3.0 | 15 | Event-driven, defined risk |

**Key changes from v1:**
- `mean_reversion` 4.5 → 3.5 (4.5 allows 6.5σ drawdown, too aggressive)
- `rsi2_connors` 5.0 → 2.5 + max_hold=5 (5.0 ATR is decorative for 1-5 bar strategy)
- Added `max_hold_bars` per strategy (was global 30)

**Location:** Store mapping in `BacktestConfig`, NOT on Signal schema (per architect review —
Signal is a Layer 0 schema shared across all layers, shouldn't carry execution params):

```python
# In backtest/config.py:
STRATEGY_STOP_ATR: dict[str, float] = {
    "momentum": 2.5, "dual_momentum": 3.0, "mean_reversion": 3.5,
    "rsi2_connors": 2.5, "ou_mean_reversion": 3.5, "pairs": 3.0, "dividend_gap": 3.0,
}
STRATEGY_MAX_HOLD: dict[str, int] = {
    "momentum": 30, "dual_momentum": 30, "mean_reversion": 20,
    "rsi2_connors": 5, "ou_mean_reversion": 25, "pairs": 20, "dividend_gap": 15,
}
```

**Constant risk-per-trade coupling** (per risk review): Wider stops should produce
smaller positions to keep per-trade risk constant:

```python
# In engine.py, when sizing the position:
risk_per_trade = Decimal("0.01")  # 1% of equity per trade
atr_mult = STRATEGY_STOP_ATR.get(strategy_name, 3.0)
position_size = (risk_per_trade * equity) / (atr_mult * atr_value)
```

This means a 3.5 ATR MR stop produces a position 71% the size of a 2.5 ATR momentum stop,
keeping dollar risk identical.

### MOEX Segment Multiplier

MOEX stocks have ~1.5-2x daily ATR/price ratio vs US large-caps. Scale stop multipliers
up by 1.2x for `ru_*` segments (existing presets already use 3.5 vs 3.0).

---

## RC4: Walk-Forward Validation Fix

### Problem

`WalkForwardConfig` defaults: `train_years=3, test_years=1, step_months=6`.
With 2-year data (2022-01-01 to 2024-12-31), this requires 4 years per window.
Result: **0 genuine OOS windows**. The reported "WF Sharpe" is actually the
in-sample Sharpe from the full-period backtest.

### Design (revised per review)

Use months instead of float years to avoid `relativedelta` typing issues:

```python
@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    train_months: int = 12      # 1 year training
    test_months: int = 6        # 6 months OOS
    step_months: int = 6        # non-overlapping windows
```

**step_months = 6** (not 3): Risk review found that `step_months=3` with `test_months=6`
creates overlapping OOS windows, double-counting trades in aggregate metrics. Using
`step_months = test_months` ensures non-overlapping OOS periods.

With 3 years of data (2022-2025), this produces ~4 non-overlapping OOS windows.

**Validation guard:** Add warning when 0 windows produced:
```python
windows = self.generate_windows(start_date, end_date)
if not windows:
    logger.warning("walk_forward_zero_windows", start=start_date, end=end_date)
    return WalkForwardResult()
```

### Implementation

**File:** `src/finalayze/backtest/walk_forward.py` — Change config fields to months.
Update `generate_windows()` to use `relativedelta(months=...)`.

**File:** `scripts/run_iteration.py` — Add `--wf-train-months` and `--wf-test-months`
CLI flags, defaulting to 12 and 6.

---

## RC5: Wire YAML Params to DualMomentum & OU Strategies

### Problem (corrected per review)

The original diagnosis that "all constructors ignore YAML params" was **partially wrong**.
Three strategies already load YAML params at runtime via `get_parameters(segment_id)`:
- `momentum.py` (line 137): reads `rsi_period`, `macd_fast`, `rsi_oversold`, etc.
- `mean_reversion.py`: reads `bb_period`, `bb_std_dev`, `min_confidence`, etc.
- `rsi2_connors.py`: reads thresholds from YAML

The actual problem is only in two strategies:
- **`DualMomentumStrategy`** — hardcodes `_LOOKBACK_1M = 21`, `_WEIGHT_1M = 0.4`, etc.
  YAML `lookback_fast`/`lookback_slow` are ignored.
- **`OUMeanReversionStrategy`** — hardcodes `_OU_WINDOW = 126`, `_ENTRY_THRESHOLD = 2.0`.
  YAML `ou_window`/`entry_threshold` are ignored.

### Design: Self-Loading Pattern (no factory needed)

Per architect review, the `StrategyFactory` approach is wrong — it creates double
construction and a second source of truth. Instead, fix the two broken strategies
to use the same `get_parameters(segment_id)` pattern already used by momentum.py:

**File:** `src/finalayze/strategies/dual_momentum.py`

```python
def generate_signal(self, symbol, candles, segment_id, **kwargs):
    params = self.get_parameters(segment_id)
    lookback_1m = int(params.get("lookback_1m", 21))
    lookback_3m = int(params.get("lookback_3m", 63))
    lookback_6m = int(params.get("lookback_6m", 126))
    min_confidence = float(params.get("min_confidence", 0.65))
    ...
```

**File:** `src/finalayze/strategies/ou_mean_reversion.py`

```python
def generate_signal(self, symbol, candles, segment_id, **kwargs):
    params = self.get_parameters(segment_id)
    ou_window = int(params.get("ou_window", 126))
    entry_threshold = float(params.get("entry_threshold", 2.0))
    exit_threshold = float(params.get("exit_threshold", 0.5))
    ...
```

No factory, no combiner changes, no double construction.

---

## M1: DRY JournalingStrategyCombiner (prerequisite for RC1)

### Problem

`JournalingStrategyCombiner` duplicates `StrategyCombiner.generate_signal()` (~146 lines
vs 124 lines). Must be DRY'd BEFORE RC1 regime routing changes, otherwise both must be
updated.

### Design: 4-Hook Architecture (per architect review)

The original 2-hook proposal was insufficient. The journaling combiner needs:

```python
class StrategyCombiner:
    def _on_generate_start(self, symbol: str, segment_id: str) -> None:
        """Hook: called at start of generate_signal, before strategy loop."""
        pass

    def _on_strategy_signal(
        self, name: str, strategy: BaseStrategy, signal: Signal | None, weight: Decimal
    ) -> None:
        """Hook: called after each strategy fires (including None signals)."""
        pass

    def _on_normalized(self, net: Decimal, features: dict[str, float]) -> None:
        """Hook: called after normalization, before threshold check."""
        pass

    def _on_final_signal(
        self, signal: Signal | None, contributions: dict[str, float]
    ) -> None:
        """Hook: called with the final signal (or None if below threshold)."""
        pass
```

Key points from architect review:
- `_on_strategy_signal` must receive the `strategy` object (not just name) for ML
  probas capture (`hasattr(strategy, "_registry")`)
- `_on_generate_start` needed for state reset (`_last_signals = {}`, etc.)
- `_on_normalized` needed to capture `net_score` for the decision journal

---

## M2: Kelly Adaptive Defaults

Use `RollingKelly` with per-segment `_FIXED_FRACTIONAL` priors:

| Segment | Prior Kelly Fraction |
|---------|---------------------|
| us_tech | 0.015 |
| us_broad | 0.012 |
| us_finance | 0.010 |
| us_healthcare | 0.012 |
| ru_* | 0.008 |

## M3: Pre-Trade Checks in Backtest

Enable backtest-relevant checks: `max_correlation_check`, `sector_concentration_check`,
`max_daily_trades_check`. Leave exchange-specific checks disabled.

---

## Execution Order (revised per architect review)

M1 must precede RC1 — the combiner refactoring to hooks must land before the regime
routing changes, otherwise both combiner copies need updating.

| Phase | Items | Files Modified | Expected Impact |
|-------|-------|---------------|----------------|
| 1 | RC4 (WF fix) | walk_forward.py, run_iteration.py | Honest OOS metrics |
| 2 | RC5 (YAML wiring) | dual_momentum.py, ou_mean_reversion.py | Tuning actually works |
| 3 | M1 (DRY combiner) | combiner.py, journaling_combiner.py | Prerequisite for RC1 |
| 4 | RC1 (Regime pools) | combiner.py, adx.py (new), presets/*.yaml | ~60% more signals |
| 5 | RC3 (Strategy stops) + RC2 (Sizing) | config.py, engine.py, pipeline.py | WR↑, fewer eliminated |
| 6 | M2 (Kelly) + M3 (Pre-trade) | engine.py, kelly.py | Better calibration |

**Phases 1-2 are independent** (different files). Phase 3 must precede Phase 4.
Phases 5-6 are independent of Phase 4.

## Verification Criteria

1. **WF Sharpe is genuine OOS:** `generate_windows()` produces ≥2 non-overlapping windows
2. **Trade count increases:** >1300 trades (from 1198) with regime routing
3. **Win rate improves:** >25% (from 16%) with strategy-specific stops
4. **Avg hold increases:** >5 bars (from 1.5) with wider MR stops + per-strategy max_hold
5. **Profit factor improves:** >1.15 (from 1.05) with less signal cancellation
6. **YAML params have effect:** Changing dual_momentum `lookback_1m` changes backtest results
7. **Per-strategy min trades:** Each enabled strategy generates ≥30 trades
8. **Risk-per-trade constant:** Wider stops produce proportionally smaller positions
9. **All existing tests pass:** `uv run pytest tests/ -q --no-cov`

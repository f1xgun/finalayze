# Phase 7: Evaluation Fixes — Signal Quality & Risk Calibration (v2 — Post-Review)

**Date:** 2026-03-01
**Motivation:** Batch evaluation of 48 symbols across 6 segments revealed the system is operationally safe but commercially non-functional: 98.7% SKIP rate, 159 total trades, median Sharpe -0.011, aggregate win rate ~34%.
**Reviewed by:** quant-analyst, risk-officer, ml-engineer, systems-architect

---

## Critical Pre-Fixes (discovered during review)

### Pre-Fix A: Combiner `total_enabled_weight` Bug

**Problem:** In `combiner.py` line 65-68, `total_enabled_weight += weight` runs for ALL enabled YAML entries, even when `self._strategies.get(strategy_name)` returns `None` (strategy not registered). With `normalize_mode="total"`, this inflates the denominator with phantom weight from strategies that don't exist.

**Fix in `src/finalayze/strategies/combiner.py`:**
```python
# BEFORE (line 65-68):
total_enabled_weight += weight
strategy = self._strategies.get(strategy_name)
if strategy is None:
    continue

# AFTER:
strategy = self._strategies.get(strategy_name)
if strategy is None:
    continue
total_enabled_weight += weight
```

Same fix in `src/finalayze/backtest/journaling_combiner.py` (duplicated loop).

### Pre-Fix B: Kelly Returns Non-Zero on Negative Expectancy

**Problem:** In `risk/kelly.py` line 65-66, when `kelly <= 0`, the code returns `_MIN_KELLY_FRACTION = Decimal("0.005")` (0.5%) instead of zero. The system trades with negative expected value.

**Fix in `src/finalayze/risk/kelly.py`:**
```python
# BEFORE:
if kelly <= 0:
    return _MIN_KELLY_FRACTION

# AFTER:
if kelly <= 0:
    return Decimal(0)
```

### Pre-Fix C: Missing Stop-Loss Allows Unprotected Positions

**Problem:** In `engine.py` lines 712-731, if `compute_atr_stop_loss` returns `None` (insufficient ATR data), the position opens with NO stop-loss.

**Fix in `src/finalayze/backtest/engine.py`:**
```python
stop_price = compute_atr_stop_loss(...)
if stop_price is None:
    self._journal_skip(candle=fill_candle, symbol=symbol,
                       segment_id=segment_id, reason="no_stop_loss_data",
                       signal=signal, history=history)
    return  # Do NOT enter the trade
```

---

## Fix 1: Normalization Mode as Per-Segment YAML Config

**Problem:** `normalize_mode="firing"` divides by weights of firing strategies only, making all weight tuning decorative. But changing the global default silently breaks all callers (4 unit tests, 2 E2E tests, TradingLoop, JournalingStrategyCombiner).

**Fix (per architect recommendation):** Make `normalize_mode` a per-segment YAML parameter. Do NOT change the global default.

**File: `src/finalayze/strategies/combiner.py`**
- Keep default `normalize_mode="firing"` unchanged in `__init__`
- Add logic to read `normalize_mode` from YAML preset if not explicitly passed
- Make `min_combined_confidence` configurable via YAML (currently hardcoded at 0.50)

**Files: `src/finalayze/strategies/presets/*.yaml`**
Add to each preset:
```yaml
normalize_mode: "total"
min_combined_confidence: 0.20
```

Gate of **0.20** (per quant recommendation): allows single-strategy signals when `confidence >= 0.67` and `weight >= 0.30`.

**File: `src/finalayze/backtest/journaling_combiner.py`**
- Must mirror the combiner's logic for reading `normalize_mode` from YAML
- Synchronize any changes since it duplicates the combiner loop

**No existing tests break** since the default remains `"firing"`.

---

## Fix 2: Normalize YAML Weights to Sum 1.0

**Problem:** `us_tech` and `ru_blue_chips` both sum to 1.10.

**Fix (per quant recommendation — proportional scaling):** Divide each weight by 1.10 to preserve original relative ratios.

**Files:** `src/finalayze/strategies/presets/us_tech.yaml`, `ru_blue_chips.yaml`

| Preset | momentum | mean_reversion | event_driven | pairs | ml_ensemble | Sum |
|---|---|---|---|---|---|---|
| us_tech (was) | 0.35 | 0.15 | 0.35 | 0.10 | 0.15 | 1.10 |
| us_tech (now) | 0.32 | 0.14 | 0.32 | 0.09 | 0.13 | 1.00 |
| ru_blue_chips (was) | 0.25 | 0.20 | 0.45 | 0.10 | 0.10 | 1.10 |
| ru_blue_chips (now) | 0.23 | 0.18 | 0.41 | 0.09 | 0.09 | 1.00 |

Other 6 presets already sum to 1.00 — no changes needed.

---

## Fix 3: Disable Non-Functional Strategies in Presets

**Problem (per ML engineer):** Registering EventDrivenStrategy (requires sentiment) and ML ensemble (no trained models) when they can't produce signals causes phantom weight dilution in `"total"` mode. EventDriven has weight 0.40 in `us_healthcare` but will never fire — making 40% of signal capacity dead weight.

**Fix:** Set `enabled: false` for `event_driven` and `ml_ensemble` in ALL presets until their pipelines are connected. Keep their weight configs for future activation.

**Files:** All 8 preset YAMLs.
```yaml
event_driven:
  enabled: false  # Re-enable when NewsAnalyzer wired to backtest
  weight: 0.32
  # ... params preserved

ml_ensemble:
  enabled: false  # Re-enable when models are trained
  weight: 0.13
  # ... params preserved
```

**Effect on `total_enabled_weight` (with Pre-Fix A applied):**
- us_tech: 0.32 (momentum) + 0.14 (MR) + 0.09 (pairs) = 0.55
- Single momentum at confidence 0.80: `net = 0.80 * 0.32 / 0.55 = 0.465` → passes 0.20 gate ✓
- us_healthcare: 0.20 (momentum) + 0.25 (MR) = 0.45
- Single momentum at confidence 0.70: `net = 0.70 * 0.20 / 0.45 = 0.311` → passes 0.20 gate ✓

**Register PairsStrategy in evaluation scripts** for segments that have pairs configured (us_tech only, since RU pairs unavailable on yfinance). Use `set_peer_candles()` in scripts for now — architect's `prepare()` hook is a cleaner design but out of scope for Phase 7.

---

## Fix 4: Mean Reversion Trend Filter

**Problem:** MR entries in declining stocks catch falling knives (UNH -1.478, LLY -1.232).

**File: `src/finalayze/strategies/mean_reversion.py`**

Add parameters (using same naming convention as momentum — no `_mr` suffix per architect):
```python
_DEFAULT_TREND_FILTER = False
_DEFAULT_TREND_SMA_PERIOD = 50
_DEFAULT_TREND_SMA_BUFFER_PCT = Decimal("2.0")  # 2% buffer (quant recommended, not 1%)
```

Logic:
- When `trend_filter=True` and SMA is computable:
  - Suppress BUY if `close < SMA - buffer` (strong downtrend → falling knife)
  - Suppress SELL if `close > SMA + buffer` (strong uptrend)
- Handle `len(candles) < sma_period` gracefully (skip filter, do not suppress)

**Also add parameter caching** (per architect — MR is missing it, causing 12,000 YAML reads):
```python
def get_parameters(self, segment_id: str) -> dict[str, object]:
    if segment_id in self._params_cache:
        return self._params_cache[segment_id]
    # ... load from YAML, cache, return
```

**Preset changes:**
```yaml
# us_healthcare.yaml
mean_reversion:
  trend_filter: true
  trend_sma_period: 50
  trend_sma_buffer_pct: 2.0

# us_tech.yaml
mean_reversion:
  trend_filter: true
  trend_sma_period: 100
  trend_sma_buffer_pct: 1.5
```

**Per ML engineer:** Consider using EMA-50 for healthcare (faster regime detection). Add `trend_indicator_type: ema|sma` parameter (default `sma`). Healthcare sets `ema`, tech stays `sma`.

---

## Fix 5: Cap Mean Reversion Confidence at 0.75

**Problem (per ML engineer):** High MR confidence (>0.9) correlates with LOW win rate (~20%). The formula `0.5 + distance * 2.0` rewards deep band breaches, which in practice means falling knives.

**File: `src/finalayze/strategies/mean_reversion.py`**
```python
# BEFORE:
_MAX_MR_CONFIDENCE = 1.0  # (implicit via min(1.0, ...))

# AFTER:
_MAX_MR_CONFIDENCE = 0.75
confidence = min(_MAX_MR_CONFIDENCE, _CONFIDENCE_BASE + distance * _CONFIDENCE_DISTANCE_MULTIPLIER)
```

This is a 1-line change that directly mitigates inverse confidence-win rate correlation. Full Platt scaling calibration deferred to Phase 8.

---

## Fix 6: Rolling Kelly in Evaluation Scripts

**Problem:** Kelly inputs are hardcoded at `win_rate=0.5, avg_win_ratio=1.5` → always 8.33% position.

**The engine already supports `RollingKelly`** (per architect + quant). The fix is passing it in scripts, not modifying the engine.

**Files: `scripts/run_evaluation.py`, `scripts/run_batch_evaluation.py`**
```python
from finalayze.risk.kelly import RollingKelly

engine = BacktestEngine(
    strategy=combiner,
    initial_cash=cash,
    decision_journal=journal,
    rolling_kelly=RollingKelly(),  # ← one-line addition
)
```

`RollingKelly` already:
- Uses 20-trade minimum (`_MIN_TRADES_FOR_KELLY = 20`)
- Returns `_FIXED_FRACTIONAL = 0.01` (1%) below that threshold
- Returns `Decimal(0)` when Kelly negative (after Pre-Fix B)
- Uses quarter-Kelly (`_DEFAULT_FRACTION = 0.25`)
- Caps at `max_position_pct` (20%)

---

## Fix 7: Record Stop-Loss Price in Journal

**File: `src/finalayze/backtest/engine.py`**

In `_handle_buy`, after computing `stop_price` and calling `broker.set_trailing_stop()`:
```python
self._journal_decision(
    ...,
    stop_loss_price=stop_price,
    ...
)
```

---

## ~~Fix 8: RSI Threshold Tuning~~ — DEFERRED

**Per quant + ML engineer consensus:** Tuning RSI thresholds on 2024 data and validating on 2024 data is circular (in-sample optimization). RSI 35/65 vs 40/60 performance difference is confounded by stock selection and market regime, not threshold calibration.

**Deferred to Phase 8** when:
1. 5-year data (2020-2024) is available for walk-forward validation
2. Train on 2020-2023, validate on 2024
3. Or implement adaptive thresholds via the ML pipeline

---

## Implementation Order (revised — fixes ship atomically)

**Batch 1 — Safety fixes (independent, ship first):**
1. Pre-Fix B: Kelly returns zero on negative expectancy
2. Pre-Fix C: Reject trades without stop-loss data
3. Fix 7: Record stop-loss price in journal

**Batch 2 — Signal quality (MUST ship together):**
4. Pre-Fix A: Combiner `total_enabled_weight` bug
5. Fix 1: `normalize_mode` as per-segment YAML config
6. Fix 2: Weight normalization (proportional scaling)
7. Fix 3: Disable non-functional strategies in presets

**Batch 3 — Strategy improvements:**
8. Fix 4: Mean reversion trend filter + parameter caching
9. Fix 5: Cap MR confidence at 0.75

**Batch 4 — Position sizing:**
10. Fix 6: Pass RollingKelly to evaluation scripts

**Post-implementation:**
11. Re-run 48-symbol batch evaluation
12. Compare metrics before/after

---

## Testing Strategy

### Pre-Fix A
- Add test: combiner with unregistered strategy should NOT include its weight in total_enabled_weight
- Verify existing combiner tests still pass

### Pre-Fix B
- Add test: `RollingKelly.optimal_fraction()` returns `Decimal(0)` when all trades are losses
- Update any test that asserts `_MIN_KELLY_FRACTION` return value

### Pre-Fix C
- Add test: engine skips BUY when `compute_atr_stop_loss` returns None
- Add test: journal records `skip_reason="no_stop_loss_data"`

### Fix 1
- Add test: combiner reads `normalize_mode` from YAML preset
- Add test: combiner reads `min_combined_confidence` from YAML preset
- **No existing tests break** (default unchanged)

### Fix 4
- Add test: MR trend filter suppresses BUY when close < SMA - buffer
- Add test: MR trend filter does not suppress when close > SMA
- Add test: MR trend filter skipped gracefully when insufficient candles
- Add test: parameter caching works (second call does not read YAML)

### Fix 5
- Add test: MR confidence capped at 0.75 even with deep band breach

### Fix 6
- Verify existing RollingKelly tests cover the integration path

---

## Success Criteria (re-evaluation on 2024 data)

After all fixes, re-run 48-symbol batch and expect:
- Signal rate: >3% (up from 1.3%) — conservative since we deferred RSI tuning
- Trade count: >250 (up from 159)
- Median Sharpe: >0.0 (up from -0.011)
- us_healthcare win rate: >25% (up from 15%) — from trend filter
- stop_loss_price populated on all BUY records
- No trades opened without stop-loss data
- Rolling Kelly reduces position size over time as win rate data accumulates

## Items Deferred to Phase 8
- RSI threshold tuning (requires walk-forward validation on 5-year data)
- Platt scaling confidence calibration
- Feature persistence in journal (M3)
- `BaseStrategy.prepare()` hook for cleaner PairsStrategy integration
- Wire sentiment pipeline to backtest
- Per-strategy rolling win rate gate (suppress strategies below 25% historical WR)
- Volatility regime override for trend filter (disable during capitulation)
- Wire remaining pre-trade check parameters (sector, cross-market, PDT, duplicate)

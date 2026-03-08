# ML Feature Expansion Plan

**Date:** 2026-03-08
**Branch:** `feature/ml-deep-overhaul`
**Goal:** Add features that carry predictive signal for market-neutral returns

## Problem Statement

Phases 1-3 fixed the ML infrastructure (feature selection, labeling, calibrator),
but models still produce near-constant predictions (~0.52 +/- 0.01). All three
models (XGBoost, LightGBM, CatBoost) have sub-50% accuracy on excess-return labels.

**Root cause:** The 28 existing features are all single-stock, single-timeframe
technical indicators. They cover price momentum, volatility, and volume well —
but have **zero cross-asset, regime, calendar, or breadth features**. For
market-neutral prediction (excess returns vs SPY), the model needs to know
what the market is doing, not just what the stock is doing.

### Evidence

```
AAPL predictions (281 bars): range [0.50, 0.55], std=0.012
Outside deadzone [0.42, 0.58]: 0/281 (0%)
WF fold accuracies: 0.404, 0.445, 0.459, 0.467, 0.471, 0.480, 0.487, 0.493, 0.503, 0.547, 0.568
Best fold: 56.8% — still essentially random
```

### Feature Gap Analysis

| Category | Current | Needed | Gap |
|----------|---------|--------|-----|
| Price/Momentum | 10 | 10 | OK |
| Volatility | 5 | 5 | OK |
| Volume/Liquidity | 4 | 4 | OK |
| Wavelet | 4 | 4 | OK (niche) |
| Cross-Asset | 0 | 4-6 | CRITICAL |
| Regime/Macro | 0 | 3-4 | CRITICAL |
| Calendar/Seasonal | 0 | 3-4 | HIGH |
| Breadth/Relative | 0 | 2-3 | HIGH |

---

## Architecture Decisions (from reviews)

### AD1: Window Size — Pass Full History Instead of Fixed Window

**Problem:** Current `window_size=60` is insufficient for features requiring
63-bar (rolling_beta, rolling_corr) or 252-bar (vix_percentile) lookbacks.
Raising to 252 loses 28% of training samples.

**Decision:** Pass `sorted_candles[0 : entry_index + 1]` (full history up to
entry bar) to `compute_features` instead of a fixed-size window. Benefits:
- Maximizes training samples (no loss from window truncation)
- Naturally accommodates any lookback length
- `compute_features` already handles varying lengths via `min(window, len)`
- Cost: ~2x compute for pandas rolling ops (acceptable for training pipeline)

**Change:** In `labeling.py`, `build_triple_barrier_dataset` passes
`sorted_candles[0 : entry_index + 1]` instead of `sorted_candles[i : i + window_size]`.
Keep `_MIN_CANDLES = 80` (raised from 30) as the minimum warmup before generating
training samples.

Export `_MAX_FEATURE_LOOKBACK = 252` from `technical.py` so the training pipeline
can derive minimum candle requirements.

### AD2: MarketContext Injection — No BaseStrategy Pollution

**Problem:** MLStrategy needs benchmark + VIX candles at inference time, but
`BaseStrategy.generate_signal` has a fixed signature used by 9+ strategies.

**Decision:** Create `MarketContext` dataclass in `core/schemas.py` (Layer 0):

```python
@dataclass(frozen=True)
class MarketContext:
    benchmark_candles: list[Candle] | None = None
    vix_candles: list[Candle] | None = None
```

Injection mechanism:
- `MLStrategy` has a `set_market_context(ctx: MarketContext)` method
- `StrategyCombiner.__init__` accepts optional `MarketContext`, calls
  `set_market_context` on MLStrategy (the only strategy that needs it)
- `BacktestEngine` constructs `MarketContext` from pre-loaded SPY/VIX data
- For live trading: `TradingLoop` fetches SPY/VIX daily, injects via combiner
- `BaseStrategy` interface remains unchanged — no pollution

This avoids a **Layer 4 → Layer 2 violation** (MLStrategy importing data fetchers).

### AD3: Feature Importance Budget Gate

**Decision:** After training, check that cross-asset + regime features collectively
contribute ≤50% of total feature importance (gain-based). If exceeded, the model
is market-timing rather than stock-picking — log warning and investigate.

Also require ≥2 cross-asset features in MI-selected set; if zero are selected,
fail quality check.

### AD4: VIX Features Are US-Only

VIX-based features (`vix_level`, `vix_percentile_252d`, `vix_change_5d`) are
US-only. For MOEX segments, these features default to 0.0 (no VIX proxy).
The `realized_vol_ratio` feature uses stock's own volatility and works for both.

---

## Implementation Plan

### Phase A: Cross-Asset Features (Priority: CRITICAL)

**Why:** For excess-return prediction, the model needs to compare stock behavior
to benchmark behavior. Currently it sees the stock in isolation.

**New features (4):**

1. `relative_strength_21d` — stock 21d return minus SPY 21d return (alpha proxy)
2. `rolling_beta_63d` — 63-day rolling beta vs SPY (sensitivity to market)
   - Fallback when insufficient data: **1.0** (market-neutral assumption)
3. `rolling_corr_63d` — 63-day rolling correlation with SPY (diversification)
   - Fallback when insufficient data: **0.5** (moderate correlation assumption)
4. `excess_momentum_score` — (stock_ret_63d - spy_ret_63d) / max(stock_vol_63d, 0.01)
   (risk-adjusted relative momentum; denominator clamped to prevent division by zero)

**Removed from original plan:** `spy_ret_5d` and `spy_ret_21d` — pure market
direction features create market-timing risk. The model should capture relative
performance (stock vs market), not predict market direction. If SPY return
dominates feature importance, the model becomes a market-timing tool.

**Architecture change:** Extend `compute_features()` signature:

```python
def compute_features(
    candles: list[Candle],
    sentiment_score: float = 0.0,
    benchmark_candles: list[Candle] | None = None,  # NEW
) -> dict[str, float]:
```

When `benchmark_candles` is None, cross-asset features default to domain-aware
values (beta=1.0, corr=0.5, others=0.0). Backward compatible.

**Files:**
- `src/finalayze/ml/features/technical.py` — add 4 features + benchmark param
- `scripts/train_models.py` — pass benchmark candles to compute_features
- `tests/unit/test_technical_features.py` — test cross-asset features

**Acceptance criteria:**
- Cross-asset features computed correctly (verify with known SPY/AAPL data)
- Features are lagged by 1 bar (no look-ahead)
- Backward compatible when benchmark_candles=None
- Domain-aware fallbacks: beta=1.0, corr=0.5

### Phase B: Regime Features (Priority: CRITICAL)

**Why:** Models cannot distinguish crash from rally. VIX and drawdown features
provide regime context that pure price indicators miss.

**New features (4):**

1. `vix_level` — VIX close value, lagged 1 bar (US-only, 0.0 for MOEX)
2. `vix_percentile_252d` — VIX percentile rank over 252 days
   - Uses `min_periods=63` for graceful warmup (first 63 bars use available history)
   - US-only, 0.0 for MOEX
3. `vix_change_5d` — 5-day VIX change (regime shift speed, US-only)
4. `realized_vol_ratio` — hist_vol_20 / hist_vol_60 (vol acceleration, all markets)

**Removed from original plan:** `drawdown_from_high_252d` — linearly redundant
with existing `proximity_rolling_high` (= `close / rolling_max(close, 252)`).
Drawdown = `1 - proximity_rolling_high`. Tree models see identical information.

**Architecture:** Extend `compute_features()` with optional VIX candles:

```python
def compute_features(
    candles: list[Candle],
    sentiment_score: float = 0.0,
    benchmark_candles: list[Candle] | None = None,
    vix_candles: list[Candle] | None = None,  # NEW
) -> dict[str, float]:
```

VIX is available via yfinance ticker `^VIX`. MOEX segments pass `vix_candles=None`.

**Files:**
- `src/finalayze/ml/features/technical.py` — add 4 features + vix param
- `scripts/train_models.py` — fetch ^VIX candles, pass to compute_features
- `tests/unit/test_technical_features.py` — test regime features

**Acceptance criteria:**
- VIX features correctly lagged by 1 bar
- `vix_percentile_252d` uses min_periods=63 for warmup
- `realized_vol_ratio` works for all markets (uses stock's own vol)

### Phase C: Calendar & Seasonal Features (Priority: HIGH)

**Why:** Financial markets exhibit strong day-of-week and month-of-year effects.
Monday sell-offs, Friday positioning, January effect, quarter-end rebalancing.

**New features (4):**

1. `dow_sin` — sin(2*pi * day_of_week / 5) (cyclical day encoding)
2. `dow_cos` — cos(2*pi * day_of_week / 5)
3. `month_sin` — sin(2*pi * month / 12) (cyclical month encoding)
4. `month_cos` — cos(2*pi * month / 12)

These features have zero correlation with each other by construction and
capture cyclical patterns without one-hot encoding sparsity.

**Files:**
- `src/finalayze/ml/features/technical.py` — add 4 calendar features
- `tests/unit/test_technical_features.py` — test cyclical encoding values

**Acceptance criteria:**
- Monday (dow=0) produces sin=0.0, cos=1.0
- Features derived from candle timestamp, no additional data needed

### Phase D: Relative Strength Features (Priority: HIGH)

**Why:** Models see absolute price levels but not where the stock stands
relative to its own history. Adding z-scored variants helps capture
mean-reversion and breakout signals.

**New features (4):**

1. `price_zscore_60d` — (price - SMA60) / std60 (mean reversion signal)
2. `volume_zscore_20d` — (volume - vol_mean_20) / vol_std_20 (volume surprise)
3. `rsi_zscore_60d` — (RSI14 - mean_RSI14_60d) / std_RSI14_60d
   (normalized momentum, captures if RSI is unusually high/low for this stock)
4. `atr_zscore_60d` — (ATR14 - mean_ATR_60d) / std_ATR_60d
   (normalized volatility, captures vol expansion/contraction)

**Files:**
- `src/finalayze/ml/features/technical.py` — add 4 z-score features
- `tests/unit/test_technical_features.py` — test z-score computation

**Acceptance criteria:**
- Z-scores centered near 0, std near 1 over 60-day windows
- Handle edge case: std=0 (constant series) returns 0.0

### Phase E: Training Pipeline & Inference Wiring (Priority: CRITICAL)

**Why:** New features require passing benchmark and VIX candles through the
entire training and inference pipeline.

**Training-side changes:**

1. **Raise `_MIN_CANDLES` from 30 to 80** in `technical.py` — required for
   60-bar z-score windows plus warmup
2. **Export `_MAX_FEATURE_LOOKBACK = 252`** from `technical.py`
3. **Pass full history** in `labeling.py` — `sorted_candles[0 : entry_index + 1]`
   instead of fixed window (see AD1)
4. **Fetch benchmark + VIX once per segment** in train_models.py
   - US: SPY + ^VIX via yfinance
   - MOEX: IMOEX via Tinkoff, no VIX
5. **Align by timestamp** (reuse `_align_benchmark_candles` from Phase 2)
6. **Pass to compute_features** during dataset building
7. **Narrow exception catch** in `labeling.py` line 301 — catch
   `InsufficientDataError` and `ValueError` only, log others at warning level

**Look-ahead prevention:** All features use data up to and including the entry
bar only. The training pipeline calls `compute_features(candles[:entry_idx+1])`.
Benchmark/VIX candles are also sliced to `[:entry_idx+1]` before passing.

**Inference-side changes (AD2):**

1. **Add `MarketContext` dataclass** to `core/schemas.py`
2. **Add `set_market_context()`** to `MLStrategy`
3. **`StrategyCombiner`** accepts optional `MarketContext`, injects into MLStrategy
4. **`BacktestEngine`** pre-loads SPY/VIX candles, constructs `MarketContext`,
   passes to combiner before bar loop
5. **`TradingLoop`** (live) fetches SPY/VIX daily, refreshes `MarketContext`

**Files:**
- `src/finalayze/core/schemas.py` — add `MarketContext` dataclass
- `src/finalayze/ml/features/technical.py` — raise `_MIN_CANDLES`, export lookback
- `src/finalayze/ml/training/labeling.py` — full history + narrow exceptions
- `scripts/train_models.py` — fetch + pass benchmark/VIX
- `src/finalayze/strategies/ml_strategy.py` — `set_market_context()`, use in signal
- `src/finalayze/strategies/combiner.py` — inject MarketContext into MLStrategy
- `src/finalayze/backtest/engine.py` — construct MarketContext, inject into combiner

### Phase F: Retrain & Validate (Priority: CRITICAL)

After implementing Phases A-E:

1. **Train with new features:**
   ```bash
   uv run python scripts/train_models.py --segment us_tech \
     --walk-forward --excess-returns
   ```

2. **Check feature selection:** with 44 features (28 old + 16 new), MI should
   select a more diverse set including cross-asset and regime features.
   **Gate:** ≥2 cross-asset features in MI-selected set (fail if zero).

3. **Check feature importance budget:** cross-asset + regime features must
   contribute ≤50% of total gain-based importance. If exceeded, investigate
   market-timing risk.

4. **Check prediction range:** target [0.30, 0.70] vs current [0.49, 0.55]

5. **Validate WF fold accuracy:** target >55% average, >50% fold pass rate

6. **Backtest with ML enabled** if models pass quality gates

7. **Compare to baseline** (ML disabled)

---

## Execution Order

```
Phase C (Calendar)          ─┐
Phase D (Relative Strength)  ├── Independent, parallel (no external data needed)
                             ─┘
         │
         ▼
Phase A (Cross-Asset)       ─┐
Phase B (Regime/VIX)         ├── Need external data fetching
                             ─┘
         │
         ▼
Phase E (Pipeline Wiring)   ── Wire everything together (AD1 + AD2)
         │
         ▼
Phase F (Retrain & Validate) ── Measure impact + AD3 budget gate
```

Phases C+D are pure computation on existing candle data — implement first.
Phases A+B require fetching external data (SPY, VIX) — implement second.
Phase E wires everything together. Phase F validates.

## Expected Feature Count

| Category | Current | Added | Total |
|----------|---------|-------|-------|
| Price/Momentum | 10 | 0 | 10 |
| Volatility | 5 | 0 | 5 |
| Volume/Liquidity | 4 | 0 | 4 |
| Wavelet | 4 | 0 | 4 |
| Cross-Asset (A) | 0 | 4 | 4 |
| Regime/VIX (B) | 0 | 4 | 4 |
| Calendar (C) | 0 | 4 | 4 |
| Relative Strength (D) | 0 | 4 | 4 |
| Microstructure | 3 | 0 | 3 |
| **Total** | **28** | **16** | **44** |

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Feature count | 28 | 44 |
| MI-selected features (cross-asset included) | 0 cross-asset | >= 2 cross-asset |
| Feature importance budget (cross-asset+regime) | N/A | <= 50% |
| WF fold accuracy (average) | 48% | >= 55% |
| Prediction range on real data | [0.49, 0.55] | [0.35, 0.65] |
| Predictions outside deadzone | 0% | >= 20% |
| Backtest Sharpe delta (ML on vs off) | 0% | >= +5% |

## Risks

1. **Cross-asset features may dominate** — even without raw SPY returns,
   relative_strength and excess_momentum could dominate. Mitigated by AD3
   importance budget gate (≤50%).

2. **VIX data alignment** — VIX has different trading hours than stocks.
   Mitigate by using close-to-close only, lagged by 1 bar.

3. **Feature count** — 44 features with ~15K samples. Rule of thumb:
   need 50x samples per feature = 2200. We have 15K, so fine.
   MI selection will further reduce to ~15 active features.

4. **Inference latency** — fetching SPY + VIX at inference time adds network
   calls. Mitigate by MarketContext injection (AD2) with daily refresh.

5. **Calendar features are weak** — day-of-week/month effects are well-known
   and may be arbitraged away. But they're cheap to compute and provide a
   baseline temporal signal. Risk is low.

6. **Full-history window increases compute** — passing all candles up to entry
   bar (~2x cost vs fixed 60-bar window). Acceptable for training pipeline.

## Review Feedback Incorporated

| Source | Issue | Resolution |
|--------|-------|------------|
| ML Engineer | `_MIN_CANDLES` too low (30) | Raised to 80 (AD1) |
| ML Engineer | Beta fallback 0.0 wrong | Domain-aware: beta=1.0, corr=0.5 (Phase A) |
| ML Engineer | VIX percentile needs warmup | min_periods=63 (Phase B) |
| Quant Analyst | `spy_ret_5d/21d` market-timing risk | Removed (Phase A) |
| Quant Analyst | `drawdown_from_high_252d` redundant | Removed — use existing `proximity_rolling_high` (Phase B) |
| Quant Analyst | `excess_momentum_score` denominator | Clamped: `max(stock_vol_63d, 0.01)` (Phase A) |
| Quant Analyst | Feature importance budget | AD3: cross-asset+regime ≤50% |
| Systems Architect | `window_size=60` insufficient | Full history pass (AD1) |
| Systems Architect | BaseStrategy pollution | MarketContext injection (AD2) |
| Systems Architect | VIX for MOEX undefined | US-only, MOEX defaults 0.0 (AD4) |
| Systems Architect | Exception swallowing in labeling | Narrow catch in Phase E |
| Systems Architect | Layer 4→2 violation risk | MarketContext avoids it (AD2) |

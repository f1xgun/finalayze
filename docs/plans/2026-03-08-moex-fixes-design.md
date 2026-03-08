# MOEX Backtest Pipeline Fixes — Design

**Date:** 2026-03-08
**Validated by:** quant-analyst, risk-officer, systems-architect (4-agent review)
**Baseline:** week5-review-fixes — US WF Sharpe +0.0054, MOEX WF Sharpe -0.0273

## Problem

MOEX segments severely underperform in backtests due to 5 root causes identified by expert team:
1. Currency-blind initial cash (100K treated as RUB = ~$1,100 portfolio)
2. Mixed-currency metrics aggregation (RUB PnL pooled with USD PnL)
3. Confidence inflation on high-vol MOEX instruments (dual_momentum)
4. Strategy parameters copy-pasted from US without MOEX calibration
5. Static regime provider (single snapshot for entire backtest)

## Fixes

### Fix 1: RUB-Denominated Initial Cash (CRITICAL)

**Where:** `scripts/run_iteration.py`, `src/finalayze/backtest/engine.py`

When segment starts with `ru_`, multiply `--cash` by `FALLBACK_USDRUB` (90.0).
`--cash 100000` → 9,000,000 RUB for MOEX segments.

Also fix `min_pos` in engine.py from `Decimal(100)` to `Decimal(5000)` (RUB).

**Unblocks:** Issue #2 (quantity_zero for LKOH, YNDX, MGNT, etc.)

### Fix 2: Currency-Aware Metrics Aggregation (HIGH)

**Where:** `scripts/run_iteration.py`

Compute per-segment metrics separately (infrastructure exists). For combined top-level metrics,
convert MOEX trade PnL by dividing by `FALLBACK_USDRUB` before pooling with US trades.
Same conversion for equity snapshots.

### Fix 3: Vol-Normalize Dual Momentum Confidence (HIGH)

**Where:** `src/finalayze/strategies/dual_momentum.py`

Change: `confidence = min(0.95, 0.4 + abs(score) * 1.0)`
To: `confidence = min(0.95, 0.4 + abs(score) / realized_vol * 0.15)`

Where `0.15` is baseline annual vol (typical US large cap). A 10% return in 30% vol market
gets same confidence as 5% return in 15% vol market.

### Fix 4: Wider Bollinger Params for MOEX (MEDIUM)

**Where:** `src/finalayze/strategies/presets/ru_*.yaml`

- `bb_std_dev: 2.5` for ru_blue_chips, ru_finance, ru_tech
- `bb_std_dev: 2.8` for ru_energy (commodity-linked, higher vol)
- `rsi_oversold_mr: 25` (from 30)
- `rsi_overbought_mr: 75` (from 70)

### Fix 5: Reduce SMA Warmup for MOEX RSI2 (MEDIUM)

**Where:** `src/finalayze/strategies/presets/ru_*.yaml`

`sma_trend_period: 100` (from 200) for RSI2 Connors in all ru_* presets.
Recovers 39% of MOEX data window lost to warmup.

### Fix 6: Time-Varying MOEX Regime (MEDIUM)

**Where:** `scripts/run_iteration.py`

Create `RollingVolRegimeProvider` that computes 20-day realized vol from IMOEX candles
per bar. Replace `StaticRegimeProvider` which freezes a single end-of-period snapshot
for the entire backtest.

Interface matches existing `RegimeProvider` protocol.

### Fix 7: Wider ADX Routing Bands for MOEX (LOW)

**Where:** `src/finalayze/strategies/presets/ru_*.yaml`

- `trend_threshold: 30` (from 35)
- `mr_threshold: 20` (from 15)

Widens clear-regime zones, shrinks ambiguous zone where dominant-pool-wins
discards minority signals.

### Fix 8: Max Hold Bars MOEX Uplift (LOW)

**Where:** `src/finalayze/backtest/config.py`

Add 1.3x multiplier for MOEX segments in `resolve_max_hold_bars()`,
matching pattern of existing 1.2x ATR stop uplift.

## Files Touched

| Fix | Files |
|---|---|
| 1 | `scripts/run_iteration.py`, `src/finalayze/backtest/engine.py` |
| 2 | `scripts/run_iteration.py` |
| 3 | `src/finalayze/strategies/dual_momentum.py` |
| 4 | `src/finalayze/strategies/presets/ru_*.yaml` (4 files) |
| 5 | `src/finalayze/strategies/presets/ru_*.yaml` (4 files) |
| 6 | `scripts/run_iteration.py` |
| 7 | `src/finalayze/strategies/presets/ru_*.yaml` (4 files) |
| 8 | `src/finalayze/backtest/config.py` |

## Testing Strategy

- TDD for fixes 1, 2, 3, 6, 8 (code changes)
- Fixes 4, 5, 7 are YAML-only — validated by backtest iteration
- Final validation: backtest `ru_blue_chips` solo, then combined `us_tech,ru_blue_chips`

## Success Criteria

- MOEX segments produce trades for all symbols (no quantity_zero for stocks < 2000 RUB)
- Combined US+MOEX WF Sharpe ≥ 0 (currently -0.0058)
- MOEX-only WF Sharpe improvement from -0.0273
- No regression on US-only metrics (WF Sharpe stays ≥ +0.005)

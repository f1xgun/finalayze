---
phase: 32-critical-sandbox-fixes
plan: 01
status: complete
started: 2026-04-07T14:00:00Z
completed: 2026-04-07T17:00:00Z
---

## Summary

Fixed critical data pipeline and safety defaults in the trading loop and settings.

### What was built

1. **_CANDLE_LOOKBACK 60 -> 210**: All strategies (RSI2 Connors needing SMA-200, dual_momentum needing 126 bars) now receive sufficient historical candles in live mode.
2. **Kill switch startup guard**: `TradingLoop.start()` raises RuntimeError if kill switch is active, preventing a killed system from resuming trading on Docker restart.
3. **Sandbox rollout default MINIMAL**: Sandbox mode defaults to `RolloutPhase.MINIMAL` when `FINALAYZE_ROLLOUT_PHASE` not explicitly set.
4. **Calendar-aware staleness**: `_is_candle_stale` accounts for weekends and MOEX holidays (10-day New Year block). Threshold raised to 72h.

### Key files

- `src/finalayze/orchestration/trading_loop.py` — lookback, kill switch check, staleness
- `config/settings.py` — sandbox rollout default
- `tests/unit/core/test_trading_loop.py` — new tests
- `tests/unit/test_settings.py` — rollout tests

### Commits

- `2760f02` — fix(32-01): increase _CANDLE_LOOKBACK to 210 and add kill switch startup check
- `ae22dfb` — fix(32-01): sandbox rollout default MINIMAL + calendar-aware staleness

### Deviations

None.

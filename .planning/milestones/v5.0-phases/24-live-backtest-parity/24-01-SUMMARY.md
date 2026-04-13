---
phase: 24-live-backtest-parity
plan: 01
subsystem: orchestration
tags: [trailing-stop, stop-loss, state-machine, live-backtest-parity, re-entry-guard]

# Dependency graph
requires:
  - phase: 22-dependency-layers
    provides: orchestration/ module extraction with TradingLoop
provides:
  - StopLossState-based trailing stops in TradingLoop (matching SimulatedBroker)
  - Per-cycle re-entry guard (_cycle_exited_symbols) preventing same-cycle re-entry
affects: [24-live-backtest-parity]

# Tech tracking
tech-stack:
  added: []
  patterns: [StopLossState reuse from simulated_broker in live path]

key-files:
  created:
    - tests/unit/test_trading_loop_parity.py
  modified:
    - src/finalayze/orchestration/trading_loop.py
    - tests/unit/test_stop_loss_atomicity.py
    - tests/unit/test_phase5_stop_loss.py
    - tests/unit/test_trading_loop_kelly.py
    - tests/unit/test_trading_loop_metrics.py
    - tests/unit/test_critical_safety.py
    - tests/unit/test_trading_loop_thread_safety.py

key-decisions:
  - "Derive ATR value from stop price formula (atr = (entry - stop) / multiplier) instead of adding a new compute_atr_value function"
  - "Reuse StopLossState dataclass from simulated_broker.py (single source of truth for trailing stop state)"

patterns-established:
  - "StopLossState is the canonical trailing stop representation for both backtest and live paths"
  - "_cycle_exited_symbols set pattern for per-cycle exclusion guards"

requirements-completed: [PARITY-02, PARITY-04]

# Metrics
duration: 8min
completed: 2026-03-23
---

# Phase 24 Plan 01: Trailing Stop State Machine Summary

**Trailing stop state machine with 5-step ratcheting logic and per-cycle re-entry guard wired into TradingLoop, matching SimulatedBroker behavior**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-23T19:38:48Z
- **Completed:** 2026-03-23T19:47:19Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 8

## Accomplishments
- Replaced bare Decimal stop-loss prices with StopLossState objects in TradingLoop
- Implemented 5-step trailing stop logic matching SimulatedBroker (high-water mark, activation threshold, ratchet-only-up, trigger check, SELL + exit record)
- Added _cycle_exited_symbols set preventing stopped-out symbols from re-entering in same equity cycle
- Migrated 6 existing test files to StopLossState API with zero regressions (76 tests pass)

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests** - `cac5186` (test)
2. **Task 1 (GREEN): Implementation** - `4c18ff4` (feat)

_Note: TDD task with RED + GREEN commits_

## Files Created/Modified
- `tests/unit/test_trading_loop_parity.py` - 9 tests covering all 5 trailing stop behaviors
- `src/finalayze/orchestration/trading_loop.py` - StopLossState wiring, trailing logic, re-entry guard
- `tests/unit/test_stop_loss_atomicity.py` - Migrated to StopLossState API
- `tests/unit/test_phase5_stop_loss.py` - Migrated to StopLossState API
- `tests/unit/test_trading_loop_kelly.py` - Migrated to StopLossState API
- `tests/unit/test_trading_loop_metrics.py` - Migrated to StopLossState API
- `tests/unit/test_critical_safety.py` - Migrated to StopLossState API
- `tests/unit/test_trading_loop_thread_safety.py` - Migrated to StopLossState API

## Decisions Made
- Derived ATR value algebraically from stop price formula rather than adding a new function: `atr = (entry_price - stop) / multiplier`
- Reused StopLossState dataclass from simulated_broker.py instead of creating a separate live version
- Placed re-entry guard after candle fetch and stop-loss check (not before) to ensure stops still update even for exited symbols

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated 6 existing test files referencing _stop_loss_prices**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Renaming _stop_loss_prices to _stop_states broke 6 existing test files (test_stop_loss_atomicity, test_phase5_stop_loss, test_trading_loop_kelly, test_trading_loop_metrics, test_critical_safety, test_trading_loop_thread_safety)
- **Fix:** Migrated all references to use _stop_states with StopLossState objects, updated assertions to check `.current_stop` instead of bare Decimal comparisons
- **Files modified:** 6 test files
- **Verification:** All 76 tests pass
- **Committed in:** 4c18ff4 (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for correctness -- renaming the internal dict required updating all consumers. No scope creep.

## Issues Encountered
- CircuitLevel enum uses NORMAL (not GREEN) -- fixed in test immediately
- MagicMock with spec=TradingLoop needed explicit _metrics=None attribute to avoid AttributeError in _submit_order

## Known Stubs

None -- all functionality is fully wired.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Trailing stops now match SimulatedBroker behavior in live path
- Re-entry guard prevents same-cycle signal generation for stopped-out symbols
- Ready for 24-02 (remaining live-backtest parity items)

---
*Phase: 24-live-backtest-parity*
*Completed: 2026-03-23*

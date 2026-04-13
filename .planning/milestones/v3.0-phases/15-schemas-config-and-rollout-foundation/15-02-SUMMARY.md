---
phase: 15-schemas-config-and-rollout-foundation
plan: 02
subsystem: risk
tags: [rollout, circuit-breaker, pre-trade-check, loss-limits, capital-ladder, moex]

# Dependency graph
requires:
  - phase: 15-01
    provides: RolloutLimits dataclass, ROLLOUT_LIMITS mapping, Settings.effective_risk_limits()
provides:
  - Rollout-aware PreTradeChecker wiring via effective_risk_limits()
  - Rollout-aware CircuitBreaker wiring via effective_risk_limits()
  - Rollout-aware LossLimitTracker wiring via effective_risk_limits()
  - Fixed cross-market circuit breaker bug (was 0.80, now default 0.10)
  - Capital ladder validation script for MOEX lot-size viability
affects: [16-sandbox-monitoring, 17-go-live]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "effective_risk_limits() as single source of truth for all risk component init"
    - "Capital ladder validation before going live at new capital tier"

key-files:
  created:
    - scripts/validate_capital_ladder.py
    - tests/unit/test_capital_ladder.py
  modified:
    - src/finalayze/core/trading_loop.py
    - src/finalayze/main.py
    - tests/unit/test_rollout.py
    - tests/unit/test_trading_loop.py

key-decisions:
  - "CrossMarketCircuitBreaker uses default 0.10 with no args rather than passing rollout value"
  - "LossLimitTracker receives daily_loss_limit_pct * 100 (fraction to percent conversion)"

patterns-established:
  - "Risk component init reads from Settings.effective_risk_limits() not raw settings fields"

requirements-completed: [ROLL-02, ROLL-03]

# Metrics
duration: 5min
completed: 2026-03-21
---

# Phase 15 Plan 02: Rollout Wiring Summary

**Wired rollout limits into PreTradeChecker, CircuitBreaker, and LossLimitTracker via effective_risk_limits(); fixed cross-market breaker bug (0.80->0.10); added MOEX capital ladder validation script**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-21T20:15:45Z
- **Completed:** 2026-03-21T20:20:23Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- PreTradeChecker, CircuitBreaker, and LossLimitTracker all receive limits from Settings.effective_risk_limits()
- MINIMAL phase now enforces: 3% max position, 1% daily loss limit, 2% drawdown auto-stop
- CrossMarketCircuitBreaker bug fixed -- no longer passes 0.80 as halt_threshold
- Capital ladder script validates lot-size viability across 4 tiers x 3 phases x 15 MOEX instruments

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire rollout limits into TradingLoop and main.py + fix cross-market bug** - `472c9e9` (feat)
2. **Task 2: Create capital ladder validation script and tests** - `56ef0fc` (feat)

_Note: Both tasks followed TDD (RED->GREEN)_

## Files Created/Modified
- `src/finalayze/core/trading_loop.py` - PreTradeChecker and LossLimitTracker init now use effective_risk_limits()
- `src/finalayze/main.py` - CircuitBreaker init uses effective_risk_limits(); CrossMarketCircuitBreaker uses default
- `scripts/validate_capital_ladder.py` - Capital ladder validation CLI with validate_position() and run_ladder()
- `tests/unit/test_rollout.py` - Added 6 wiring integration tests
- `tests/unit/test_capital_ladder.py` - 5 tests for capital ladder validation
- `tests/unit/test_trading_loop.py` - Updated mock settings to provide effective_risk_limits()

## Decisions Made
- CrossMarketCircuitBreaker() called with no args (uses _DEFAULT_CROSS_HALT=0.10) rather than passing a rollout-specific value, since the default is correct
- Capital ladder uses representative MOEX prices as static defaults (not live data), appropriate for pre-deployment validation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_trading_loop mock settings for effective_risk_limits**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Existing test mock for settings didn't provide effective_risk_limits(), causing MagicMock to return a mock object that Decimal() couldn't convert
- **Fix:** Added proper effective_risk_limits mock returning ROLLOUT_LIMITS[FULL] to _make_settings()
- **Files modified:** tests/unit/test_trading_loop.py
- **Verification:** All 56 trading_loop tests pass
- **Committed in:** 472c9e9 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed market hours in wiring tests**
- **Found during:** Task 1 (RED phase)
- **Issue:** PreTradeChecker tests using market_id="moex" failed because current time was outside MOEX hours
- **Fix:** Added explicit dt parameter with a Monday 10:00 UTC datetime (within MOEX hours)
- **Files modified:** tests/unit/test_rollout.py
- **Committed in:** 472c9e9 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for test correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Rollout configuration is fully wired -- MINIMAL/STANDARD/FULL phases enforce different risk limits at runtime
- Capital ladder script available for pre-deployment validation at each capital tier
- Ready for Phase 16 (Sandbox Monitoring) to observe rollout limits in action

---
*Phase: 15-schemas-config-and-rollout-foundation*
*Completed: 2026-03-21*

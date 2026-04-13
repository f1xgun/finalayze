---
phase: 17-production-operations
plan: 03
subsystem: api
tags: [telegram, kill-switch, go-no-go, wiring, fastapi-lifespan]

requires:
  - phase: 17-production-operations/17-02
    provides: TelegramBotHandler with /kill and /gonogo command handlers
provides:
  - Runtime wiring of KillSwitch, GoNoGoReporter, BrokerRouter into TelegramBotHandler via lifespan
affects: [production-deployment, telegram-bot]

tech-stack:
  added: []
  patterns: [module-level-instance-for-lifespan-wiring]

key-files:
  created:
    - tests/unit/test_main_bot_wiring.py
  modified:
    - src/finalayze/main.py

key-decisions:
  - "Module-level _bot_handler_instance follows existing _trading_loop_instance pattern"
  - "GoNoGoReporter instantiated from gate_thresholds.yaml in lifespan, not create_app"
  - "All bot wiring inside existing `if _trading_loop_instance is not None:` block"

patterns-established:
  - "Module-level instance + lifespan wiring: store handler at module level in create_app, wire dependencies in lifespan after _build_trading_loop completes"

requirements-completed: [OPS-04]

duration: 3min
completed: 2026-03-22
---

# Phase 17 Plan 03: TelegramBotHandler Wiring Summary

**Wire KillSwitch, GoNoGoReporter, BrokerRouter into TelegramBotHandler at runtime via main.py lifespan**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-21T21:56:12Z
- **Completed:** 2026-03-21T21:59:28Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- TelegramBotHandler now receives KillSwitch at runtime so /kill command actually works in production
- GoNoGoReporter instantiated from gate_thresholds.yaml and wired to bot handler for /gonogo command
- BrokerRouter, circuit_breakers, and trading_loop also wired so /status and /breakers show live data
- 5 tests verify all wiring paths including null-safety when telegram is not configured

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for bot wiring** - `6e5d151` (test)
2. **Task 1 (GREEN): Implement bot wiring in main.py** - `8dbad41` (feat)

## Files Created/Modified
- `src/finalayze/main.py` - Added _bot_handler_instance module-level variable, stored in create_app(), wired kill_switch/go_no_go_reporter/broker_router/circuit_breakers in lifespan()
- `tests/unit/test_main_bot_wiring.py` - 5 tests verifying wiring and null-safety

## Decisions Made
- Module-level `_bot_handler_instance` follows the existing `_trading_loop_instance` pattern
- GoNoGoReporter requires `GateThresholds` from YAML -- instantiated in lifespan with try/except for graceful degradation
- All bot wiring placed inside `if _trading_loop_instance is not None:` block since it depends on trading loop components

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- OPS-04 gap fully closed
- /kill and /gonogo Telegram commands will work in production when trading loop starts
- All existing tests continue to pass (18/18)

---
*Phase: 17-production-operations*
*Completed: 2026-03-22*

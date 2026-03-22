---
phase: 21-error-handling-hardening
plan: 01
subsystem: risk, core, api
tags: [garch, volatility, redis, eventbus, authentication, error-handling, structlog]

requires:
  - phase: none
    provides: existing GARCH, EventBus, and /kill endpoint implementations
provides:
  - GARCH fit_forecast returns rolling vol fallback instead of NaN (when data >= 2)
  - EventBus.create_group catches only redis.ResponseError, re-raises others
  - POST /kill endpoint requires X-API-Key authentication
affects: [risk, position-sizing, event-bus, api-security]

tech-stack:
  added: []
  patterns:
    - "structlog warning on GARCH fallback paths for observability"
    - "Narrowed exception handling pattern: catch specific, log+re-raise generic"

key-files:
  created: []
  modified:
    - src/finalayze/risk/garch.py
    - src/finalayze/core/events.py
    - src/finalayze/api/v1/system.py
    - tests/unit/test_garch.py
    - tests/unit/test_events.py
    - tests/unit/test_api_system.py

key-decisions:
  - "GARCH returns NaN only for < 2 data points; all other failures use rolling vol fallback"
  - "forecast_garch_vol convenience function changed to use fit_forecast_safe for consistent fallback"
  - "EventBus uses try/except instead of contextlib.suppress for granular error handling"

patterns-established:
  - "Rolling vol fallback: std(returns) * sqrt(252) as universal GARCH failure fallback"
  - "Narrowed exception catching: redis.ResponseError specifically, not bare Exception"

requirements-completed: [ERR-01, ERR-02, API-01]

duration: 3min
completed: 2026-03-23
---

# Phase 21 Plan 01: Error Handling Hardening Summary

**GARCH NaN fallback with rolling vol + structlog warnings, EventBus exception narrowing to redis.ResponseError, and POST /kill authenticated via X-API-Key**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-22T20:56:28Z
- **Completed:** 2026-03-22T20:59:45Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- GARCH fit_forecast never returns NaN when >= 2 data points exist; uses rolling vol (std * sqrt(252)) fallback with structlog warnings
- EventBus.create_group catches only redis.ResponseError (group-already-exists), re-raises all other exceptions after logging
- POST /kill endpoint now requires X-API-Key authentication, returning 401 without valid key

## Task Commits

Each task was committed atomically:

1. **Task 1: GARCH NaN fallback with logging and EventBus exception narrowing** - `1d901ae` (feat)
2. **Task 2: POST /kill endpoint authentication** - `603a793` (feat)

## Files Created/Modified

- `src/finalayze/risk/garch.py` - Added _rolling_vol_fallback helper, structlog logging, replaced NaN returns with fallback
- `src/finalayze/core/events.py` - Replaced contextlib.suppress(Exception) with try/except redis.ResponseError, added structlog
- `src/finalayze/api/v1/system.py` - Added Depends(api_key_auth) to /kill endpoint
- `tests/unit/test_garch.py` - Added 7 new tests for fallback behavior, updated existing insufficient-data test
- `tests/unit/test_events.py` - Added 3 new tests for exception narrowing
- `tests/unit/test_api_system.py` - Added 3 new tests for /kill auth

## Decisions Made

- GARCH returns NaN only for < 2 data points; all other failure paths use rolling vol fallback -- this ensures the sizing pipeline never receives NaN from GARCH when any reasonable estimate is possible
- Changed forecast_garch_vol to use fit_forecast_safe instead of fit_forecast -- makes the convenience function consistent with the fallback behavior
- EventBus uses try/except pattern instead of contextlib.suppress -- allows specific redis.ResponseError suppression while logging and re-raising unexpected errors

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Error handling hardening for GARCH, EventBus, and /kill is complete
- Ready for plan 21-02 (remaining error handling tasks)

## Self-Check: PASSED

---
*Phase: 21-error-handling-hardening*
*Completed: 2026-03-23*

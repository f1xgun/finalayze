---
phase: 06-sandbox-validation
plan: 01
subsystem: execution
tags: [grpc, tinkoff, health-check, reconnection, order-reconciliation, candle-staleness]

# Dependency graph
requires:
  - phase: 05-integration-and-telegram
    provides: TelegramAlerter, TradingLoop, BrokerRouter wiring
provides:
  - reconnect_client() method for gRPC channel recovery
  - get_open_orders() and cancel_order_safe() for order management
  - Real /health probes for TinkoffBroker and feed freshness
  - _is_candle_stale() for data quality gating
  - _attempt_grpc_reconnect() with exponential backoff
  - _reconcile_inflight_orders() for startup order cleanup
affects: [06-sandbox-validation, sandbox-testing, monitoring]

# Tech tracking
tech-stack:
  added: []
  patterns: [exponential-backoff-reconnection, health-probe-wiring, startup-reconciliation]

key-files:
  created:
    - tests/unit/test_tinkoff_reconnect.py
    - tests/unit/test_order_reconciliation.py
    - tests/unit/test_candle_staleness.py
  modified:
    - src/finalayze/execution/tinkoff_broker.py
    - src/finalayze/api/v1/system.py
    - src/finalayze/core/trading_loop.py
    - tests/unit/test_api_health.py

key-decisions:
  - "cancel_order_safe() added as bool-returning alternative to cancel_order() (which raises)"
  - "tinkoff added to mandatory health components; 'unknown' status accepted (not configured)"
  - "All open orders on startup treated as stale and cancelled (conservative reconciliation)"

patterns-established:
  - "set_tinkoff_broker() setter pattern for injecting broker into health probes"
  - "update_feed_timestamp() for decoupled feed freshness tracking"
  - "_reconnect_delays list for configurable backoff schedule"

requirements-completed: [AUT-04, AUT-06]

# Metrics
duration: 5min
completed: 2026-03-15
---

# Phase 06 Plan 01: Error Recovery Hardening Summary

**gRPC reconnection with exponential backoff, real /health probes for Tinkoff, candle staleness detection, and startup order reconciliation**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-14T22:15:36Z
- **Completed:** 2026-03-14T22:21:19Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- TinkoffBroker can reconnect gRPC channel (thread-safe, returns bool)
- /health endpoint returns real Tinkoff probe status instead of hardcoded "ok"
- Candle staleness detection prevents trading on stale market data
- In-flight order reconciliation cancels stale orders on startup
- gRPC reconnection loop with 5 attempts and exponential backoff (30-300s)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add reconnect_client() and get_open_orders() to TinkoffBroker** - `25052cd` (feat)
2. **Task 2: Wire real health probes, candle staleness, and gRPC reconnection loop** - `58ed097` (feat)

_Note: TDD tasks -- tests written first (RED), then implementation (GREEN)_

## Files Created/Modified
- `src/finalayze/execution/tinkoff_broker.py` - Added reconnect_client(), get_open_orders(), cancel_order_safe()
- `src/finalayze/api/v1/system.py` - Real _check_tinkoff() and _check_feed_freshness() probes, set_tinkoff_broker()
- `src/finalayze/core/trading_loop.py` - _is_candle_stale(), _attempt_grpc_reconnect(), _reconcile_inflight_orders()
- `tests/unit/test_tinkoff_reconnect.py` - 7 tests for reconnection logic
- `tests/unit/test_order_reconciliation.py` - 7 tests for order listing and cancel
- `tests/unit/test_api_health.py` - 7 new tests for Tinkoff and feed freshness probes
- `tests/unit/test_candle_staleness.py` - 6 tests for staleness detection

## Decisions Made
- cancel_order_safe() created as a separate bool-returning method (existing cancel_order() raises BrokerError)
- Tinkoff added to mandatory health components with "unknown" as acceptable state (broker not configured)
- All open orders on startup are treated as stale and cancelled (conservative approach -- next cycle retries)
- Feed freshness threshold set at 2 hours (configurable via module constant)

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
- Thread safety test required module-level mocking of AsyncClient (gRPC constructor needs event loop) -- resolved by patching at outer scope before spawning threads.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Error recovery infrastructure ready for sandbox validation testing
- Health probes wired and testable via /health endpoint
- Reconnection + reconciliation logic ready for integration testing in sandbox

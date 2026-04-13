---
phase: 19-concurrency-safety-and-integration-fixes
plan: 02
subsystem: core
tags: [concurrency, threading, stop-loss, health-monitor, telegram, TOCTOU]

# Dependency graph
requires:
  - phase: 15-schemas-config-and-rollout-foundation
    provides: "HealthMonitor with update_feed_timestamp method"
  - phase: 16-sandbox-monitoring-and-go-no-go-gate
    provides: "GoNoGoReporter and /gonogo Telegram command"
provides:
  - "Atomic stop-loss check-and-sell preventing double-sell race condition"
  - "Direct feed timestamp wiring without getattr indirection"
  - "Verified GoNoGoReporter runtime import"
affects: [trading-loop, risk-management, monitoring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single lock hold for read-check-act sequences (no TOCTOU)"
    - "Direct method calls instead of getattr for critical monitoring APIs"

key-files:
  created:
    - tests/unit/test_stop_loss_atomicity.py
    - tests/unit/test_feed_timestamp_wiring.py
  modified:
    - src/finalayze/core/trading_loop.py

key-decisions:
  - "Keep broker.submit_order inside the lock -- correctness over throughput since stop-loss is rare"
  - "On submit failure, preserve stop price for retry next cycle rather than clearing"
  - "Remove getattr indirection -- update_feed_timestamp is stable public API, silent failure masks bugs"

patterns-established:
  - "TOCTOU prevention: entire read-check-act under single lock hold"
  - "Critical monitoring calls use direct invocation, not defensive getattr"

requirements-completed: [CONC-01, INT-01, INT-02]

# Metrics
duration: 5min
completed: 2026-03-22
---

# Phase 19 Plan 02: Stop-Loss Atomicity and Integration Fixes Summary

**Atomic stop-loss under single lock hold preventing double-sell TOCTOU race, plus direct feed timestamp wiring and verified /gonogo import**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-22T20:18:57Z
- **Completed:** 2026-03-22T20:23:45Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Fixed critical TOCTOU race in _check_stop_losses that could cause double-sell orders when two threads read the same stop price concurrently
- Replaced getattr indirection for update_feed_timestamp with direct call, preventing silent monitoring failures
- Verified GoNoGoReporter import works at runtime (INT-01 integration gap closed)
- 11 new tests (5 stop-loss atomicity + 6 feed timestamp/import verification)

## Task Commits

Each task was committed atomically:

1. **Task 1: Make stop-loss check-and-sell atomic (CONC-01)** - `e3abc3e` (test: TDD RED) + `6fd0755` (feat: TDD GREEN)
2. **Task 2: Verify /gonogo import and wire feed timestamp (INT-01, INT-02)** - `fe2d3e5` (test: TDD RED) + `7945391` (feat: TDD GREEN)

## Files Created/Modified
- `src/finalayze/core/trading_loop.py` - Atomic stop-loss (single lock hold), direct feed timestamp call
- `tests/unit/test_stop_loss_atomicity.py` - 5 tests proving atomicity, concurrency safety, and error recovery
- `tests/unit/test_feed_timestamp_wiring.py` - 6 tests verifying GoNoGoReporter import and direct timestamp wiring

## Decisions Made
- Kept broker.submit_order() inside the lock -- correctness matters more than throughput for a rare stop-loss event
- On submit_order failure, stop price is preserved (not cleared) so the stop can retry next cycle
- Removed getattr("update_feed_timestamp") pattern -- the method is part of HealthMonitor's stable public API; getattr masks rename bugs silently

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- TradingLoop constructor requires kelly_fraction on mock settings (Decimal conversion fails on MagicMock) -- fixed by setting mock_settings.kelly_fraction = 0.5 in test helper
- Source-inspection test for getattr was initially too broad (any getattr + any update_feed_timestamp) -- refined to regex matching the specific getattr(...update_feed_timestamp...) pattern

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Stop-loss is now concurrency-safe for production use
- Feed timestamp monitoring will surface stale data correctly
- /gonogo import verified, ready for live Telegram bot operation

---
*Phase: 19-concurrency-safety-and-integration-fixes*
*Completed: 2026-03-22*

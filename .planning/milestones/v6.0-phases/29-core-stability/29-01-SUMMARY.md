---
phase: 29-core-stability
plan: 01
subsystem: execution
tags: [grpc, asyncio, event-loop, tinkoff, trading-loop]

# Dependency graph
requires:
  - phase: 28-operational-hygiene
    provides: TradingLoop with _async_loop, TinkoffBroker/TinkoffFetcher with self-managed loops
provides:
  - Dedicated gRPC event loop isolated from HTTP/DB/Telegram async work
  - grpc_loop injection parameter for TinkoffBroker and TinkoffFetcher
  - BlockingIOError exception handler suppressing benign PollerCompletionQueue EAGAIN
  - TradingLoop._run_grpc() method for gRPC coroutine routing
affects: [run_sandbox, tinkoff_broker, tinkoff_data, trading_loop, bond_cycle]

# Tech tracking
tech-stack:
  added: []
  patterns: [shared-grpc-loop-injection, dedicated-event-loop-per-concern, exception-handler-suppression]

key-files:
  created: []
  modified:
    - src/finalayze/execution/tinkoff_broker.py
    - src/finalayze/data/fetchers/tinkoff_data.py
    - src/finalayze/orchestration/trading_loop.py
    - scripts/run_sandbox.py
    - tests/unit/core/test_trading_loop.py

key-decisions:
  - "Shared grpc_loop created in run_sandbox.py and injected into all gRPC consumers rather than each creating their own"
  - "Backward-compatible: grpc_loop=None falls back to self-managed loop for tests and standalone scripts"
  - "close() on broker/fetcher skips loop lifecycle when grpc_loop is injected -- owner manages it"
  - "BlockingIOError from PollerCompletionQueue suppressed via loop exception handler"

patterns-established:
  - "grpc_loop injection: all gRPC consumers accept optional grpc_loop parameter, use it if provided, self-manage if not"
  - "Loop ownership: creator of the loop is responsible for stopping it; consumers only close their clients"

requirements-completed: [GRPC-01]

# Metrics
duration: 9min
completed: 2026-03-30
---

# Phase 29 Plan 01: gRPC Loop Consolidation Summary

**Dedicated gRPC event loop isolating PollerCompletionQueue from HTTP/DB/Telegram work to eliminate 60-min strategy cycle drift**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-30T19:35:35Z
- **Completed:** 2026-03-30T19:44:45Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- TinkoffBroker and TinkoffFetcher accept optional grpc_loop parameter for external event loop injection
- TradingLoop owns dedicated _grpc_loop with BlockingIOError exception handler suppression
- run_sandbox.py creates one shared grpc_loop and injects it into all three gRPC consumers (broker, bond broker, fetcher, TradingLoop)
- Backward compatibility preserved: tests and standalone scripts use self-managed loops when grpc_loop=None

## Task Commits

Each task was committed atomically:

1. **Task 1: Add grpc_loop parameter to TinkoffBroker and TinkoffFetcher** - `79c95c7` (feat)
2. **Task 2: Add _grpc_loop to TradingLoop and wire to broker/fetcher** - `b159aa3` (feat)

## Files Created/Modified
- `src/finalayze/execution/tinkoff_broker.py` - Added grpc_loop parameter, updated _run_async/close for injection
- `src/finalayze/data/fetchers/tinkoff_data.py` - Added grpc_loop parameter, updated _run_async/close for injection
- `src/finalayze/orchestration/trading_loop.py` - Added _grpc_loop field, _init_grpc_loop(), _run_grpc(), stop() cleanup
- `scripts/run_sandbox.py` - Create shared grpc_loop, inject into all gRPC consumers
- `tests/unit/core/test_trading_loop.py` - 3 new tests for gRPC loop isolation

## Decisions Made
- Created the grpc_loop in run_sandbox.py (not inside TradingLoop.__init__) so it can be passed to broker/fetcher constructors before TradingLoop is constructed
- TradingLoop._run_grpc() method added for future use when TradingLoop needs to call gRPC coroutines directly (currently broker/fetcher handle their own _run_async internally)
- make_bond_broker() propagates grpc_loop from the equity broker to the bond broker

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed S110 lint warnings in close() methods**
- **Found during:** Task 2 (lint verification)
- **Issue:** try/except/pass pattern flagged by ruff S110/SIM105 in new close() code
- **Fix:** Replaced bare `pass` with `_log.debug("event_loop_stop_failed_on_close")`
- **Files modified:** src/finalayze/execution/tinkoff_broker.py, src/finalayze/data/fetchers/tinkoff_data.py
- **Committed in:** b159aa3 (Task 2 commit)

**2. [Rule 3 - Blocking] Adapted plan to actual architecture**
- **Found during:** Task 2
- **Issue:** Plan assumed TradingLoop._run_async() was used for gRPC calls, but actual architecture has TinkoffBroker/TinkoffFetcher handling their own _run_async() internally. TradingLoop._run_async() is only for non-gRPC work (FX, Telegram, Redis, DB).
- **Fix:** Instead of changing _run_async call sites in TradingLoop (there are none for gRPC), focused on injecting grpc_loop into broker/fetcher constructors. Added _run_grpc() to TradingLoop for future use.
- **Files modified:** src/finalayze/orchestration/trading_loop.py, scripts/run_sandbox.py
- **Committed in:** b159aa3 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Deviations align with plan objective. The gRPC loop consolidation achieves the same isolation goal through proper constructor injection rather than call-site changes.

## Issues Encountered
None

## Known Stubs
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- gRPC event loop is now isolated from general async work
- Strategy cycles should no longer drift from PollerCompletionQueue contention
- Ready for sandbox validation to confirm drift elimination

---
*Phase: 29-core-stability*
*Completed: 2026-03-30*

---
phase: 22-dependency-layer-cleanup
plan: 02
subsystem: architecture
tags: [dependency-injection, metrics, layer-docs, prometheus]

# Dependency graph
requires:
  - phase: 22-01
    provides: "TradingLoop moved to orchestration/ (L5)"
provides:
  - "TradingLoop with injected MetricsCollector (no L6 import)"
  - "backtest/ and monitoring/ documented layer assignments"
affects: [api, monitoring, orchestration]

# Tech tracking
tech-stack:
  added: []
  patterns: [constructor-injection-for-cross-layer-deps, type-checking-guard-imports]

key-files:
  created:
    - src/finalayze/monitoring/CLAUDE.md
  modified:
    - src/finalayze/orchestration/trading_loop.py
    - src/finalayze/main.py
    - src/finalayze/backtest/CLAUDE.md
    - tests/unit/test_trading_loop_metrics.py

key-decisions:
  - "MetricsCollector injected as type (class reference) not instance -- all methods are static"
  - "Metrics calls guarded with if self._metrics to make collection truly optional"

patterns-established:
  - "Constructor injection for cross-layer dependencies: pass L6 classes via constructor, guard usage with None check"
  - "TYPE_CHECKING guard for cross-layer type annotations"

requirements-completed: [LAYER-03, LAYER-04]

# Metrics
duration: 4min
completed: 2026-03-23
---

# Phase 22 Plan 02: MetricsCollector Injection and Layer Documentation Summary

**MetricsCollector injected into TradingLoop via constructor, eliminating 6 deferred L6 imports; backtest/ and monitoring/ assigned definitive layers**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-22T21:33:47Z
- **Completed:** 2026-03-22T21:38:27Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Eliminated all 6 deferred `from finalayze.api.metrics import MetricsCollector` imports from trading_loop.py method bodies
- MetricsCollector is now injected via constructor parameter, making it truly optional (None = silently skip metrics)
- backtest/ CLAUDE.md updated with definitive cross-cutting L4-5 assignment
- monitoring/ CLAUDE.md created with Layer 6 assignment and complete module documentation

## Task Commits

Each task was committed atomically:

1. **Task 1: Inject MetricsCollector into TradingLoop constructor** - `928ee85` (feat)
2. **Task 2: Document layer assignments for backtest/ and monitoring/** - `34631be` (docs)

## Files Created/Modified
- `src/finalayze/orchestration/trading_loop.py` - Added metrics_collector constructor param, replaced 6 deferred imports with self._metrics usage
- `src/finalayze/main.py` - Wire MetricsCollector to TradingLoop constructor in _build_trading_loop()
- `tests/unit/test_trading_loop_metrics.py` - Updated tests to inject mock MetricsCollector via _metrics attribute instead of patching deferred import
- `src/finalayze/backtest/CLAUDE.md` - Updated layer from vague "boundary" to definitive cross-cutting L4-5 assignment
- `src/finalayze/monitoring/CLAUDE.md` - Created with Layer 6 assignment, key files, public API, contracts, testing

## Decisions Made
- MetricsCollector injected as type (class reference) not instance, since all methods are @staticmethod
- All 6 MetricsCollector usage sites guarded with `if self._metrics:` to make metrics collection truly optional
- Tests updated to inject mock directly via `_metrics` attribute instead of patching the deferred import path

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_trading_loop_metrics.py for new injection pattern**
- **Found during:** Task 1 (MetricsCollector injection)
- **Issue:** Tests used `patch("finalayze.api.metrics.MetricsCollector")` which no longer works after removing deferred imports
- **Fix:** Tests now inject a MagicMock via `loop._metrics` attribute in `_make_loop_stub()`
- **Files modified:** tests/unit/test_trading_loop_metrics.py
- **Verification:** All 17 trading loop tests pass
- **Committed in:** 928ee85 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test fix was necessary consequence of removing deferred imports. No scope creep.

## Issues Encountered
- Pre-existing integration test failure in tests/integration/test_trading_loop.py (decimal.InvalidOperation in LossLimitTracker due to mock settings) -- not caused by our changes, confirmed by testing on clean main branch

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TradingLoop no longer has any direct api.metrics imports -- clean L5 module
- All modules now have documented layer assignments
- Ready for any remaining Phase 22 plans

---
*Phase: 22-dependency-layer-cleanup*
*Completed: 2026-03-23*

---
phase: 23-order-sizing-bug-fixes
plan: 01
subsystem: execution
tags: [trading-loop, position-sizing, kelly, circuit-breaker, yaml-preset]

# Dependency graph
requires:
  - phase: 22-dependency-layers
    provides: "orchestration/ module extraction with TradingLoop"
provides:
  - "SELL orders sized by actual held position, not Kelly fraction"
  - "Per-position sector exposure using cached last prices"
  - "Dynamic CAUTION threshold from segment preset YAML"
  - "_get_last_price() and _get_segment_min_confidence() helpers"
affects: [24-live-backtest-parity, 25-data-validation]

# Tech tracking
tech-stack:
  added: []
  patterns: ["per-instrument price cache for cross-position calculations", "segment preset YAML as runtime config source"]

key-files:
  created:
    - tests/unit/test_trading_loop_sizing_bugs.py
  modified:
    - src/finalayze/orchestration/trading_loop.py

key-decisions:
  - "SELL orders skip both Kelly sizing and CAUTION reduction -- sell entire held position"
  - "Price cache (_last_prices) built during strategy cycle as instruments are processed"
  - "Segment min_confidence read from same YAML presets as StrategyCombiner, cached per segment"
  - "Fallback to 0.5 min_confidence if preset YAML not found (safe default)"

patterns-established:
  - "_last_prices dict: per-instrument price cache populated during strategy cycle"
  - "_segment_min_confidence dict: cached segment preset values to avoid repeated YAML reads"

requirements-completed: [SIZE-01, SIZE-02, SIZE-03]

# Metrics
duration: 5min
completed: 2026-03-23
---

# Phase 23 Plan 01: Order Sizing Bug Fixes Summary

**Fixed SELL qty (held position), sector exposure (per-position prices), and CAUTION threshold (segment preset) in TradingLoop**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-23T19:18:39Z
- **Completed:** 2026-03-23T19:23:31Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- SELL orders now use actual held position quantity instead of Kelly-computed amount (SIZE-01)
- Sector exposure calculation uses each position's own cached last price (SIZE-02)
- CAUTION circuit breaker threshold reads min_combined_confidence from segment preset YAML (SIZE-03)
- 7 regression tests covering all three bugs plus edge cases

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing regression tests** - `ceb0af2` (test)
2. **Task 2: Fix all three sizing bugs** - `857d245` (fix)

_TDD flow: RED (failing tests) then GREEN (implementation fixes)_

## Files Created/Modified
- `tests/unit/test_trading_loop_sizing_bugs.py` - 7 regression tests for SIZE-01/02/03
- `src/finalayze/orchestration/trading_loop.py` - Fixed _build_order (SELL branch, CAUTION threshold), _process_instrument (sector exposure), added _get_last_price() and _get_segment_min_confidence() helpers, added _last_prices and _segment_min_confidence caches

## Decisions Made
- SELL orders skip both Kelly sizing and CAUTION reduction entirely -- always sell the full held position
- Price cache is populated lazily during strategy cycle (each instrument's candle close is cached after fetch)
- Segment min_confidence uses same YAML path pattern as StrategyCombiner (strategies/presets/<seg_id>.yaml)
- Falls back to 0.5 if preset YAML file not found (matches the old hardcoded behavior as safe default)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed stale noqa: F401 suppression**
- **Found during:** Task 2
- **Issue:** PortfolioState import had F401 suppression but is now used in _build_order signature
- **Fix:** Removed the noqa comment
- **Files modified:** src/finalayze/orchestration/trading_loop.py
- **Verification:** ruff check passes clean

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Trivial lint cleanup. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all three bugs are fully fixed with no placeholder values.

## Next Phase Readiness
- Order sizing is correct for live trading path
- Ready for Phase 24 (live-backtest parity) which will wire PositionSizingPipeline in live

---
*Phase: 23-order-sizing-bug-fixes*
*Completed: 2026-03-23*

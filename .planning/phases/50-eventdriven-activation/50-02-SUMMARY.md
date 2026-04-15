---
phase: 50-eventdriven-activation
plan: 02
subsystem: strategies, core
tags: [combiner, dedup, credibility, event-driven, signal-features]

# Dependency graph
requires: [50-01]
provides:
  - CBR/dividend duplicate signal guard in StrategyCombiner
  - Credibility threading from trading loop through combiner to EventDrivenStrategy
  - event_type_code embedded in Signal.features for downstream consumption
affects: [combiner-aggregation, event-driven-signals, trading-loop-signal-generation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dedup pattern: _dedup_event_signals() zeroes lower-weight when 2+ strategies share same event_type_code"
    - "Selective kwarg threading: combiner passes credibility/event_type_code only to event_driven strategy"

key-files:
  created: []
  modified:
    - src/finalayze/strategies/event_driven.py
    - src/finalayze/strategies/combiner.py
    - src/finalayze/core/trading_loop.py
    - tests/unit/test_event_driven_strategy.py
    - tests/unit/test_strategy_combiner.py

key-decisions:
  - "Dedup recomputes weighted_score/total_weight but preserves feature_contributions for observability"
  - "Credibility defaults to 1.0 in trading_loop — full credibility pipeline deferred to future plan"
  - "_DEDUP_EVENT_CODES is frozenset({1.0, 2.0}) matching cbr_rate and earnings/dividend from _EVENT_TYPE_FLOAT_MAP"

patterns-established:
  - "Selective kwarg threading in combiner: event_driven gets credibility+event_type_code, others get standard args"
  - "Post-collection dedup: collect signals during loop, apply dedup after, recompute scores if needed"

requirements-completed: [EVNT-01, EVNT-02]

# Metrics
duration: 9min
completed: 2026-04-15
---

# Phase 50 Plan 02: Combiner Dedup + Credibility Threading Summary

**CBR/dividend duplicate-signal guard in StrategyCombiner with credibility threading and event_type_code in Signal.features**

## Performance

- **Duration:** 9 min (557s)
- **Started:** 2026-04-15T07:44:31Z
- **Completed:** 2026-04-15T07:53:48Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- `EventDrivenStrategy.generate_signal()` accepts `event_type_code` kwarg and embeds it in `Signal.features`
- `StrategyCombiner.generate_signal()` accepts `credibility` and `event_type_code`, threads them only to event_driven strategy (other strategies unaffected)
- `TradingLoop._get_event_type_code()` reads event type from Redis cache for combiner dedup
- `_dedup_event_signals()` module-level function zeroes lower-weight strategy when 2+ strategies share same CBR (1.0) or dividend (2.0) event_type_code
- Feature contributions preserved after dedup for observability (only weighted scores recomputed)
- 11 new tests: 5 for event_type_code/credibility threading, 6 for dedup logic
- Full unit suite green: 1576 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for credibility + event_type_code** - `4bb736c` (test)
2. **Task 1 (GREEN): Credibility threading + event_type_code in features** - `c94d4ea` (feat)
3. **Task 2 (RED): Failing tests for dedup** - `18271d1` (test)
4. **Task 2 (GREEN): CBR/dividend dedup in combiner** - `274d191` (feat)

## Files Created/Modified

- `src/finalayze/strategies/event_driven.py` - Added `event_type_code` parameter and feature embedding
- `src/finalayze/strategies/combiner.py` - Added `_DEDUP_EVENT_CODES`, `_dedup_event_signals()`, credibility/event_type_code threading, dedup integration in generate_signal
- `src/finalayze/core/trading_loop.py` - Added `_get_event_type_code()` method, wired event_type_code to generate_signal call
- `tests/unit/test_event_driven_strategy.py` - 5 new tests: TestEventTypeCode (3), TestCredibilityInCombiner (2)
- `tests/unit/test_strategy_combiner.py` - 6 new tests: TestCombinerDedup class, updated MockStrategy to accept **kwargs

## Decisions Made

- Dedup recomputes `weighted_score` and `total_weight` but preserves `feature_contributions` dict — both strategies' features remain visible for observability and debugging
- Credibility defaults to 1.0 in trading_loop `_process_instrument()` — full article-to-ticker credibility pipeline is a future enhancement (TODO comment in code)
- `_DEDUP_EVENT_CODES` is a frozenset containing only 1.0 and 2.0 (matching cbr_rate and earnings/dividend from `_EVENT_TYPE_FLOAT_MAP`); codes 3.0-6.0 are not deduped as they don't have dedicated strategies
- Added `noqa: PLR0912, PLR0915` to `generate_signal()` since dedup integration increased branches/statements beyond ruff thresholds — method complexity is inherent to the signal aggregation logic

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] MockStrategy **kwargs compatibility**
- **Found during:** Task 2 (GREEN phase)
- **Issue:** MockStrategy.generate_signal() in test_strategy_combiner.py didn't accept `has_open_position`, `credibility`, `event_type_code` kwargs, causing TypeError when combiner called event_driven strategy
- **Fix:** Added `**kwargs: object` to MockStrategy and TrackingStrategy generate_signal signatures
- **Files modified:** tests/unit/test_strategy_combiner.py
- **Commit:** 274d191

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minimal — test fixture compatibility fix, no production code changes.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- Event-driven activation complete: credibility threading, event_type_code embedding, and dedup guard all operational
- Phase 50 (EventDriven Activation) fully delivered — both plans complete
- Ready for Phase 51+ (Portfolio Review Agent, Anomaly Interpreter, etc.)

## Self-Check: PASSED

All 5 files verified present. All 4 commits verified in git log.

---
*Phase: 50-eventdriven-activation*
*Completed: 2026-04-15*

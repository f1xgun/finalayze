---
phase: 32-critical-sandbox-fixes
plan: 02
subsystem: data, strategies, orchestration
tags: [caching, rate-limiting, news-pipeline, signal-diagnostics, moex]

requires:
  - phase: 32-01
    provides: "sandbox rollout config and calendar-aware staleness"
provides:
  - "CachingFetcher + RateLimiter wiring in both sandbox entry points"
  - "event_driven strategy enabled for all 3 MOEX segments"
  - "Per-gate signal drop counters in CycleLogEntry"
affects: [32-03, sandbox-monitoring, news-pipeline]

tech-stack:
  added: []
  patterns:
    - "CachingFetcher(delegate=TinkoffFetcher) wrapping pattern for API quota preservation"
    - "Per-gate drop counters in CycleLogEntry for signal pipeline observability"

key-files:
  created:
    - tests/unit/core/test_validation_logger.py
  modified:
    - scripts/run_sandbox.py
    - src/finalayze/main.py
    - src/finalayze/strategies/presets/ru_energy.yaml
    - src/finalayze/strategies/presets/ru_finance.yaml
    - src/finalayze/core/validation_logger.py
    - src/finalayze/orchestration/trading_loop.py
    - .env.example

key-decisions:
  - "RateLimiter at 4 req/sec matches T-Bank API limits"
  - "CachingFetcher wraps TinkoffFetcher in both entry points for consistent behavior"
  - "Signal drop counters use default=0 for backward compatibility with existing JSONL"

patterns-established:
  - "CachingFetcher(delegate=fetcher) pattern for any BaseFetcher"
  - "Per-gate counters tracked in _reset_cycle_counters and logged in CycleLogEntry"

requirements-completed: [SANDBOX-FIX-05, SANDBOX-FIX-06, SANDBOX-FIX-07, SANDBOX-FIX-08]

duration: 3min
completed: 2026-04-07
---

# Phase 32 Plan 02: Sandbox Caching/RateLimiter + News Pipeline + Signal Diagnostics Summary

**CachingFetcher with 4 req/sec RateLimiter wired in both sandbox entry points, event_driven enabled for all MOEX segments, per-gate signal drop counters in CycleLogEntry**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-07T17:59:13Z
- **Completed:** 2026-04-07T18:02:13Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Both sandbox entry points (scripts/run_sandbox.py and src/finalayze/main.py) now wrap TinkoffFetcher in CachingFetcher with RateLimiter(name="tbank", rate=4.0)
- event_driven strategy enabled in ru_energy.yaml and ru_finance.yaml (ru_blue_chips already enabled) -- all 3 MOEX segments now have news pipeline active
- CycleLogEntry tracks 3 new signal drop counters: signals_dropped_no_bars, signals_dropped_below_threshold, signals_dropped_pre_trade
- signal_dropped_below_threshold logged at INFO level with symbol and segment for live debugging
- .env.example updated with free LLM model documentation for news pipeline setup

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire CachingFetcher and RateLimiter in sandbox entry points** - `0699b98` (feat)
2. **Task 2: Enable event_driven for MOEX segments and add signal drop diagnostics** - `a59a7b7` (feat)

## Files Created/Modified
- `scripts/run_sandbox.py` - Added CachingFetcher + RateLimiter wrapping of TinkoffFetcher
- `src/finalayze/main.py` - Same CachingFetcher + RateLimiter wrapping for Docker entry point
- `src/finalayze/strategies/presets/ru_energy.yaml` - event_driven enabled: true
- `src/finalayze/strategies/presets/ru_finance.yaml` - event_driven enabled: true
- `src/finalayze/core/validation_logger.py` - 3 new signal drop counter fields on CycleLogEntry
- `src/finalayze/orchestration/trading_loop.py` - Counter tracking in _process_instrument + CycleLogEntry construction
- `.env.example` - LLM setup documentation for news pipeline
- `tests/unit/core/test_validation_logger.py` - Tests for drop counter fields with backward compat

## Decisions Made
- RateLimiter at 4.0 req/sec matches T-Bank documented API rate limits
- Used default=0 on new CycleLogEntry fields for backward compatibility with existing JSONL files
- Updated .env.example LLM model to free tier (meta-llama/llama-3.1-8b-instruct:free) as default

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required. LLM API key setup documented in .env.example.

## Next Phase Readiness
- All MOEX segments now have event_driven enabled, ready for sandbox testing with LLM key
- Signal drop diagnostics will provide visibility into pipeline behavior during next sandbox run
- CachingFetcher reduces API quota consumption for repeated candle fetches

---
*Phase: 32-critical-sandbox-fixes*
*Completed: 2026-04-07*

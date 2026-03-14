---
phase: 01-moex-equity-foundation
plan: 01
subsystem: data, backtest, core
tags: [moex, holidays, commission, calendar, trading-loop]

# Dependency graph
requires: []
provides:
  - "MOEX_COSTS with correct 0.04% Trader tariff commission rate"
  - "is_moex_trading_day() unified holiday check (weekends + fixed + transferred)"
  - "Holiday-aware TradingLoop._is_market_open for MOEX market"
  - "Complete transferred holidays data for 2020-2026"
affects: [01-moex-equity-foundation, backtest-engine, live-trading]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Static per-year transferred holidays dict for MOEX calendar"
    - "Lazy import of moex_calendar in trading_loop (Layer 6 -> Layer 2)"

key-files:
  created:
    - tests/unit/test_trading_loop_holidays.py
  modified:
    - src/finalayze/backtest/costs.py
    - src/finalayze/data/moex_calendar.py
    - src/finalayze/core/trading_loop.py
    - tests/unit/test_costs.py
    - tests/unit/test_moex_calendar.py
    - tests/e2e/test_paper_trading_cycle.py

key-decisions:
  - "Transferred holidays stored as static per-year frozensets (government decrees are static)"
  - "is_moex_holiday expanded to check both fixed and transferred holidays"
  - "Lazy import in _is_market_open to maintain dependency layering (core -> data)"

patterns-established:
  - "MOEX holiday data: _TRANSFERRED_HOLIDAYS dict[int, frozenset[tuple[int, int]]] pattern"
  - "Unified is_moex_trading_day() for all holiday checks (backtest + live)"

requirements-completed: [EQF-04, EQF-05]

# Metrics
duration: 6min
completed: 2026-03-14
---

# Phase 01 Plan 01: MOEX Costs & Holidays Summary

**Fixed MOEX commission rate to 0.04% Trader tariff and added transferred holidays 2020-2026 with unified holiday gate in TradingLoop**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-14T13:01:24Z
- **Completed:** 2026-03-14T13:08:04Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Fixed MOEX_COSTS.commission_rate from 0.0003 (0.03%) to 0.0004 (0.04% Trader tariff)
- Added _TRANSFERRED_HOLIDAYS with per-year transferred holidays for 2020-2026
- Created is_moex_trading_day() unified function checking weekends + fixed + transferred holidays
- Wired holiday gate into TradingLoop._is_market_open so MOEX is correctly closed on holidays
- Updated trading_days_gap to use unified is_moex_trading_day for complete gap counting

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix MOEX commission rate and add transferred holidays** (TDD)
   - `1be3f08` (test: add failing tests for MOEX commission rate and transferred holidays)
   - `a517072` (feat: fix MOEX commission rate and add transferred holidays)
2. **Task 2: Wire holiday check into TradingLoop._is_market_open** (TDD)
   - `c6c4e8d` (test: add failing tests for TradingLoop MOEX holiday gate)
   - `d4e2044` (feat: wire MOEX holiday gate into TradingLoop._is_market_open)
   - `a74e5dd` (fix: fix e2e test date that falls on MOEX holiday)

## Files Created/Modified
- `src/finalayze/backtest/costs.py` - Fixed MOEX_COSTS.commission_rate to 0.0004
- `src/finalayze/data/moex_calendar.py` - Added _TRANSFERRED_HOLIDAYS, is_moex_trading_day(), updated is_moex_holiday and trading_days_gap
- `src/finalayze/core/trading_loop.py` - Added MOEX holiday gate in _is_market_open
- `tests/unit/test_costs.py` - Added commission rate assertion, updated regression tests
- `tests/unit/test_moex_calendar.py` - Added is_moex_trading_day tests with parametrized transferred holidays
- `tests/unit/test_trading_loop_holidays.py` - New test file for TradingLoop holiday gate
- `tests/e2e/test_paper_trading_cycle.py` - Fixed test date from MOEX holiday to normal trading day

## Decisions Made
- Transferred holidays stored as static per-year frozensets -- government decrees are published annually and are static data
- is_moex_holiday expanded to check both fixed and transferred holidays (backward-compatible, no separate function needed)
- Lazy import of is_moex_trading_day in trading_loop.py to maintain dependency layering (core/trading_loop.py is architecturally Layer 6, moex_calendar.py is Layer 2)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed e2e test using MOEX holiday as test date**
- **Found during:** Task 2 (TradingLoop holiday gate wiring)
- **Issue:** test_paper_trading_cycle.py used 2026-02-23 (Defender of the Fatherland Day) as MARKET_OPEN_DT. After wiring the holiday gate, _is_market_open correctly rejected this date, causing the MOEX broker e2e test to fail.
- **Fix:** Changed MARKET_OPEN_DT to 2026-02-24 (Tuesday, normal trading day)
- **Files modified:** tests/e2e/test_paper_trading_cycle.py
- **Verification:** All 5 e2e tests pass
- **Committed in:** a74e5dd

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix was necessary -- test relied on broken behavior (no holiday check). No scope creep.

## Issues Encountered
- Pre-existing test failure in test_settings_phase3.py::test_telegram_bot_token_default_empty -- unrelated to our changes (telegram token default value issue). Not fixed per scope boundary rules.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MOEX commission rate and holiday calendar are now correct, ready for position sizing work (Plan 02)
- is_moex_trading_day() available for backtest engine bar skipping if needed
- TradingLoop will correctly skip MOEX holidays in live trading

---
*Phase: 01-moex-equity-foundation*
*Completed: 2026-03-14*

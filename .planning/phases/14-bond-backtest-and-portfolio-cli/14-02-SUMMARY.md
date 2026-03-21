---
phase: 14-bond-backtest-and-portfolio-cli
plan: 02
subsystem: backtest
tags: [portfolio, bond, equity, usdrub, tinkoff, cli, moex]

# Dependency graph
requires:
  - phase: 14-bond-backtest-and-portfolio-cli
    provides: BondBacktestEngine.run() with OFZ rotation via plan 01
provides:
  - Real _run_bond_backtest() fetching OFZ data via TinkoffFetcher and running BondBacktestEngine
  - Real _run_equity_backtest() fetching MOEX equity candles and running BacktestEngine per symbol
  - Real _extract_usdrub_series() fetching USDRUB FX candles from T-Bank API
affects: [14-bond-backtest-and-portfolio-cli]

# Tech tracking
tech-stack:
  added: []
  patterns: [module-level imports for mockability, per-symbol equity backtest loop]

key-files:
  created:
    - tests/unit/test_portfolio_backtest_cli.py
  modified:
    - scripts/run_portfolio_backtest.py

key-decisions:
  - "USDRUB fetched via fetch_bond_candles with FIGI BBG0013HGFT4 (USD000UTSTOM) -- works for any FIGI instrument"
  - "Equity backtest uses DualMomentum + MeanReversion strategies with JournalingStrategyCombiner"
  - "Bond candles use close price for all OHLC fields (clean price % approximation)"
  - "BondDurationRotationStrategy initialized with maturity_date and coupon_rate from BondInfo"
  - "Per-symbol capital split for equity: total_equity_capital / number_of_symbols"

patterns-established:
  - "Module-level imports in script for clean mock patching in tests"

requirements-completed: [PORT-01, PORT-02, PORT-03]

# Metrics
duration: 10min
completed: 2026-03-21
---

# Phase 14 Plan 02: Portfolio CLI Real Engine Wiring Summary

**Portfolio backtest CLI now calls TinkoffFetcher for OFZ/equity/USDRUB data and runs real BondBacktestEngine + BacktestEngine, closing PORT-01/02/03 audit gaps**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-21T10:04:20Z
- **Completed:** 2026-03-21T10:14:30Z
- **Tasks:** 2 (TDD: 4 commits total -- 2 RED + 2 GREEN)
- **Files modified:** 2

## Accomplishments
- Replaced 3 stub functions with real implementations that load T-Bank API data and run actual backtest engines
- 9 unit tests covering all helper functions with mocked TinkoffFetcher
- Script imports cleanly (--help works), ruff + format clean
- PORT-01/02/03 audit gaps closed: orchestrator receives real engine results when FINALAYZE_TINKOFF_TOKEN is set

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for _run_bond_backtest** - `a0c2b55` (test)
2. **Task 1 GREEN: Implement _run_bond_backtest with TinkoffFetcher** - `55b3bea` (feat)
3. **Task 2 RED: Failing tests for _run_equity_backtest and _extract_usdrub_series** - `473e7b4` (test)
4. **Task 2 GREEN: Implement _run_equity_backtest and _extract_usdrub_series** - `059c640` (feat)

## Files Created/Modified
- `scripts/run_portfolio_backtest.py` - Replaced 3 stubs with real TinkoffFetcher + engine implementations; added module-level imports for BondBacktestEngine, BacktestEngine, strategy classes
- `tests/unit/test_portfolio_backtest_cli.py` - 9 tests across 3 test classes (TestRunBondBacktest, TestRunEquityBacktest, TestExtractUsdrub)

## Decisions Made
- USDRUB fetched via `fetch_bond_candles` with FIGI `BBG0013HGFT4` (USD000UTSTOM) -- this method accepts any FIGI, not just bonds
- Equity backtest uses DualMomentum + MeanReversion via JournalingStrategyCombiner (equal allocation mode)
- Bond candle construction uses close price for all OHLC fields (clean price % from T-Bank API)
- BondDurationRotationStrategy requires `bond_maturities` and `coupon_rates` in addition to `bond_durations` -- extracted from BondInfo metadata
- Per-symbol capital allocation: equity_capital / len(_EQUITY_SYMBOLS)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Candle constructor missing required fields**
- **Found during:** Task 1 GREEN phase
- **Issue:** Candle Pydantic model requires `symbol`, `market_id`, `timeframe` fields -- plan omitted them
- **Fix:** Added `symbol=ticker, market_id="moex", timeframe="1d"` to Candle construction
- **Files modified:** scripts/run_portfolio_backtest.py
- **Verification:** All tests pass
- **Committed in:** 55b3bea

**2. [Rule 1 - Bug] Fixed BondDurationRotationStrategy constructor signature**
- **Found during:** Task 1 GREEN phase
- **Issue:** Strategy requires `bond_maturities` and `coupon_rates` args -- plan only passed `bond_durations`
- **Fix:** Extract maturity_date and coupon_rate from bond_info_dict for each symbol
- **Files modified:** scripts/run_portfolio_backtest.py
- **Verification:** All tests pass with mocked strategy
- **Committed in:** 55b3bea

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required. FINALAYZE_TINKOFF_TOKEN is required at runtime but already documented.

## Next Phase Readiness
- Phase 14 complete: all plans (01 + 02) executed
- Portfolio backtest CLI is fully wired for E2E runs when T-Bank API token is available
- Pre-existing test failure in test_critical_safety.py is unrelated to changes (out of scope)

---
*Phase: 14-bond-backtest-and-portfolio-cli*
*Completed: 2026-03-21*

---
phase: 12-portfolio-assembly
plan: 02
subsystem: backtest
tags: [portfolio, walk-forward, sharpe, cli, backtest]

requires:
  - phase: 12-portfolio-assembly
    plan: 01
    provides: "PortfolioBacktestOrchestrator class, PortfolioBacktestResult dataclass"
provides:
  - "compute_walk_forward_sharpe method on PortfolioBacktestOrchestrator"
  - "run_portfolio_backtest.py CLI script for joint OFZ+equity backtesting"
affects: [12-03]

tech-stack:
  added: []
  patterns: ["Walk-forward on pre-computed merged curve (no engine re-runs)", "CLI script with graceful data-unavailable handling"]

key-files:
  created:
    - scripts/run_portfolio_backtest.py
  modified:
    - src/finalayze/backtest/portfolio_orchestrator.py
    - tests/unit/test_portfolio_orchestrator.py

key-decisions:
  - "WF uses 12mo/6mo/3mo windows (train/test/step) per CONTEXT.md decision"
  - "RUONIA 15% as risk-free rate for excess Sharpe, consistent with Plan 01"
  - "WF slices pre-computed merged curve -- no engine re-runs for validation"
  - "Too-short curves (<18 months) return WF Sharpe = 0.0 gracefully"
  - "CLI script returns gracefully when bond/equity/FX data unavailable"

patterns-established:
  - "Portfolio WF: generate_wf_windows + _compute_excess_sharpe_from_equity reused from bond_walk_forward"

requirements-completed: [PORT-03]

duration: 5min
completed: 2026-03-21
---

# Phase 12 Plan 02: Walk-Forward Sharpe & Portfolio CLI Summary

**Walk-forward Sharpe validation on merged portfolio curve using 12mo/6mo windows, plus CLI script for running joint OFZ+equity backtests end-to-end**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-21T09:16:12Z
- **Completed:** 2026-03-21T09:22:04Z
- **Tasks:** 2 (Task 1: TDD RED+GREEN, Task 2: auto)
- **Files modified:** 3

## Accomplishments
- compute_walk_forward_sharpe() method on PortfolioBacktestOrchestrator that slices pre-computed merged curve into 12mo/6mo WF windows
- Reuses generate_wf_windows and _compute_excess_sharpe_from_equity from bond_walk_forward (no duplication)
- run_portfolio_backtest.py CLI script with configurable bond/equity weights, total capital, date range
- 5 new TestWalkForward tests covering non-zero Sharpe, window params, curve-only slicing, edge cases

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing WF tests** - `a0db83d` (test)
2. **Task 1 GREEN: WF Sharpe implementation** - `7a164bd` (feat)
3. **Task 2: Portfolio backtest CLI script** - `51f7fbf` (feat)

## Files Created/Modified
- `src/finalayze/backtest/portfolio_orchestrator.py` - Added compute_walk_forward_sharpe method
- `tests/unit/test_portfolio_orchestrator.py` - Added TestWalkForward class with 5 tests (20 total)
- `scripts/run_portfolio_backtest.py` - CLI script for joint portfolio backtest

## Decisions Made
- WF uses 12mo/6mo/3mo windows per CONTEXT.md design decision
- RUONIA 15% as risk-free rate, consistent with Plan 01 and bond_walk_forward
- WF only slices merged curve -- no engine re-runs, purely analytical
- CLI script returns gracefully when data unavailable (bond/equity/FX)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed zero-stdev in WF test data**
- **Found during:** Task 1 GREEN (test verification)
- **Issue:** Test helper used constant daily returns, producing stdev=0 which makes Sharpe=0 regardless of mean
- **Fix:** Added small sinusoidal oscillation to daily returns and increased base return above RUONIA threshold
- **Files modified:** tests/unit/test_portfolio_orchestrator.py
- **Committed in:** `7a164bd` (part of GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test data fix was necessary for correct WF Sharpe validation. No scope creep.

## Issues Encountered
- setup_logging() requires WorkMode argument -- added WorkMode.TEST import to CLI script

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Portfolio orchestrator feature-complete with WF Sharpe validation
- CLI script ready for production use when OFZ/equity/FX data available
- Phase 12 Plan 03 can build on this for end-to-end portfolio validation

---
*Phase: 12-portfolio-assembly*
*Completed: 2026-03-21*

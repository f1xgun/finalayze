---
phase: 09-strategy-wiring
plan: 02
subsystem: risk
tags: [position-sizing, moex, brent, rub-oil-regime, sizing-pipeline]

requires:
  - phase: 08-data-foundation
    provides: MarketContext with Brent candles and USDRUB FX rates
provides:
  - RubOilRegimeStep sizing pipeline step for MOEX RUB/oil decorrelation regime
  - BrentGateStep sizing pipeline step for Brent-in-RUB energy gating
  - BacktestConfig fields for MOEX sizing data injection
  - run_iteration.py Brent/USDRUB data extraction from MarketContext
affects: [10-sector-rotation, 11-preferred-shares, backtest-iteration]

tech-stack:
  added: []
  patterns: [per-run pipeline construction with segment_id, synthetic Candle from FXRate]

key-files:
  created: []
  modified:
    - src/finalayze/risk/position_sizing_pipeline.py
    - src/finalayze/backtest/engine.py
    - src/finalayze/backtest/config.py
    - scripts/run_iteration.py
    - tests/unit/test_sizing_pipeline_evt_copula.py

key-decisions:
  - "Pipeline built per-run (not at init) because MOEX steps need segment_id which is only available at run() time"
  - "rub_oil_regime_signal typed as object in BacktestConfig to avoid circular import from config.py (Layer 4 boundary)"
  - "FXRate objects converted to synthetic Candle objects for RubOilRegimeSignal correlation computation"

patterns-established:
  - "Per-run pipeline construction: _build_sizing_pipeline(segment_id) method called in run() and run_portfolio()"
  - "Graceful degradation: missing MOEX data results in sizing steps being skipped, not errors"

requirements-completed: [STRAT-03, STRAT-04]

duration: 5min
completed: 2026-03-20
---

# Phase 09 Plan 02: MOEX Sizing Pipeline Steps Summary

**RubOilRegimeStep and BrentGateStep wired into sizing pipeline with Brent/USDRUB data from MarketContext for ru_* segments**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-20T08:31:27Z
- **Completed:** 2026-03-20T08:36:27Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- RubOilRegimeStep scales ru_* positions by RUB/oil correlation regime (NORMAL=1.0, ELEVATED=0.5, CRISIS=0.25)
- BrentGateStep reduces ru_energy positions by 50% when Brent-in-RUB < 5000 RUB/bbl
- Pipeline order enforced: Kelly -> VolTarget -> Regime -> RubOilRegime -> BrentGate -> Copula -> EVT -> MetaLabel -> HardCaps
- run_iteration.py extracts real Brent/USDRUB data from MarketContext and injects into BacktestConfig

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement RubOilRegimeStep and BrentGateStep** - `67ab6d4` (feat)
2. **Task 2: Wire into backtest engine pipeline** - `87486b8` (feat)
3. **Task 3: Wire run_iteration.py data supply** - `7d4e266` (feat)

## Files Created/Modified
- `src/finalayze/risk/position_sizing_pipeline.py` - Added RubOilRegimeStep and BrentGateStep classes with constants
- `src/finalayze/backtest/engine.py` - Added _build_sizing_pipeline() per-run method, imported new steps
- `src/finalayze/backtest/config.py` - Added rub_oil_regime_signal and brent_rub_price fields to BacktestConfig
- `scripts/run_iteration.py` - Added _compute_moex_sizing_data() helper, wired through _run_symbol
- `tests/unit/test_sizing_pipeline_evt_copula.py` - 9 new tests for both sizing steps

## Decisions Made
- Pipeline built per-run (not at __init__) because MOEX steps need segment_id which is only available at run() time
- rub_oil_regime_signal typed as `object | None` in BacktestConfig to avoid circular import
- FXRate objects converted to synthetic Candle objects with symbol="USDRUB", market_id="cbr" for correlation computation
- RubOilRegimeSignal import moved to TYPE_CHECKING block in position_sizing_pipeline.py (TC001 lint)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Candle constructor requires symbol, market_id, timeframe fields**
- **Found during:** Task 3 (run_iteration.py wiring)
- **Issue:** Plan showed simplified Candle(timestamp, open, high, low, close, volume) but Candle is a Pydantic model requiring symbol, market_id, timeframe
- **Fix:** Added required fields (symbol="USDRUB", market_id="cbr", timeframe="1d") to synthetic Candle creation
- **Files modified:** scripts/run_iteration.py
- **Verification:** ruff check passes
- **Committed in:** 7d4e266 (Task 3 commit)

**2. [Rule 1 - Bug] TC001 lint error on runtime import**
- **Found during:** Task 3 verification
- **Issue:** RubOilRegimeSignal imported at module level but only needed for type hints (with `from __future__ import annotations`)
- **Fix:** Moved import to TYPE_CHECKING block
- **Files modified:** src/finalayze/risk/position_sizing_pipeline.py
- **Verification:** ruff check passes, all tests pass
- **Committed in:** 7d4e266 (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MOEX sizing pipeline now has regime-aware position scaling for ru_* segments
- BrentGateStep and RubOilRegimeStep ready for backtest validation
- Data wiring complete: MarketContext already contains Brent and USDRUB data for MOEX loads

---
*Phase: 09-strategy-wiring*
*Completed: 2026-03-20*

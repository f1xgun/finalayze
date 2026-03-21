---
phase: 14-bond-backtest-and-portfolio-cli
plan: 01
subsystem: backtest
tags: [ofz, bond, rotation, cbr, backtest]

# Dependency graph
requires:
  - phase: 10-moex-sizing-pipeline
    provides: apply_ofz_rotation function in core/bond_cycle.py
provides:
  - BondBacktestEngine.run() with OFZ rotation support via layer_configs/as_of_date params
  - BondBacktestResult.ofz_rotation_active observability field
affects: [14-bond-backtest-and-portfolio-cli]

# Tech tracking
tech-stack:
  added: []
  patterns: [lazy import inside run() to avoid circular dependency]

key-files:
  created: []
  modified:
    - src/finalayze/backtest/bond_engine.py
    - tests/unit/test_bond_engine.py

key-decisions:
  - "Monkeypatch CBR_MEETINGS at finalayze.data.fetchers.cbr (source), not bond_cycle (consumer with lazy import)"
  - "date import moved from TYPE_CHECKING to runtime to support datetime.now(tz=UTC).date() fallback"

patterns-established:
  - "Lazy import pattern for bond_cycle.apply_ofz_rotation inside run() to avoid circular imports"

requirements-completed: [MACRO-02]

# Metrics
duration: 3min
completed: 2026-03-21
---

# Phase 14 Plan 01: OFZ Rotation Wiring Summary

**BondBacktestEngine.run() now calls apply_ofz_rotation when layer_configs provided, with ofz_rotation_active observability in BondBacktestResult**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-21T09:58:02Z
- **Completed:** 2026-03-21T10:01:08Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- Added ofz_rotation_active: bool field to BondBacktestResult dataclass
- Wired apply_ofz_rotation into BondBacktestEngine.run() with optional layer_configs and as_of_date params
- 3 new tests covering rotation inactive/active scenarios with monkeypatched CBR_MEETINGS
- All 22 tests pass, ruff + mypy clean

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for OFZ rotation** - `6c47462` (test)
2. **Task 1 GREEN: Wire apply_ofz_rotation into engine** - `8d1b7fe` (feat)

## Files Created/Modified
- `src/finalayze/backtest/bond_engine.py` - Added layer_configs/as_of_date params, ofz_rotation_active field, rotation logic
- `tests/unit/test_bond_engine.py` - 3 new tests in TestBondEngineOFZRotation class

## Decisions Made
- Monkeypatch CBR_MEETINGS at `finalayze.data.fetchers.cbr` (the source module) rather than `finalayze.core.bond_cycle` where it is lazily imported at call time
- Moved `date` import from TYPE_CHECKING to runtime because `datetime.now(tz=UTC).date()` is used as fallback when as_of_date and all_dates are both absent

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed monkeypatch target for CBR_MEETINGS**
- **Found during:** Task 1 GREEN phase
- **Issue:** Plan specified monkeypatch on `finalayze.core.bond_cycle.CBR_MEETINGS` but CBR_MEETINGS is lazily imported inside apply_ofz_rotation from `finalayze.data.fetchers.cbr`
- **Fix:** Changed monkeypatch target to `finalayze.data.fetchers.cbr.CBR_MEETINGS`
- **Files modified:** tests/unit/test_bond_engine.py
- **Verification:** All 3 rotation tests pass
- **Committed in:** 8d1b7fe

**2. [Rule 3 - Blocking] Fixed ruff TC004 and DTZ011 lint errors**
- **Found during:** Task 1 GREEN phase
- **Issue:** `date` was in TYPE_CHECKING block but used at runtime; `date.today()` violates DTZ011
- **Fix:** Moved `date` to runtime imports; replaced `date.today()` with `datetime.now(tz=UTC).date()`
- **Files modified:** src/finalayze/backtest/bond_engine.py
- **Verification:** ruff check passes clean
- **Committed in:** 8d1b7fe

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both auto-fixes necessary for correctness and lint compliance. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- OFZ rotation is now exercised in bond backtests (MACRO-02 audit gap closed)
- Ready for Phase 14 Plan 02 (portfolio CLI)

---
*Phase: 14-bond-backtest-and-portfolio-cli*
*Completed: 2026-03-21*

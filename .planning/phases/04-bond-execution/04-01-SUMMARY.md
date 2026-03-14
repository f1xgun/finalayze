---
phase: 04-bond-execution
plan: 01
subsystem: risk, execution, core
tags: [bond, dv01, dirty-price, layer-ledger, orm, reconciliation, tinkoff, grpc]

# Dependency graph
requires:
  - phase: 03-bond-data
    provides: BondInfo schema, bond candle cache, coupon events
provides:
  - BondPositionRecord frozen dataclass for bond position tracking
  - DV01/EqualWeight sizing fix using dirty price + transaction costs
  - LayerLedgerModel ORM for TimescaleDB persistence
  - LayerLedger bond_positions with add/remove/to_orm/from_orm
  - reconcile_with_broker startup reconciliation with Telegram alerts
  - make_bond_broker factory sharing equity broker gRPC client
  - InstrumentRegistry.get_by_figi FIGI-based lookup
  - BrokerRouter moex_bonds key routing
affects: [04-bond-execution plan 02, 04-bond-execution plan 03]

# Tech tracking
tech-stack:
  added: []
  patterns: [frozen-dataclass-for-position-records, orm-round-trip-via-to/from-methods, shared-grpc-client-factory]

key-files:
  created: []
  modified:
    - src/finalayze/core/schemas.py
    - src/finalayze/core/layer_ledger.py
    - src/finalayze/core/models.py
    - src/finalayze/risk/dv01_sizing.py
    - src/finalayze/execution/tinkoff_broker.py
    - src/finalayze/markets/instruments.py
    - tests/unit/test_dv01_sizing.py
    - tests/unit/test_layer_ledger.py
    - tests/unit/test_broker_router.py

key-decisions:
  - "Renamed face_value -> unit_cost (default 1000) for backward compatibility"
  - "BondPositionRecord merges on add: sums quantity, keeps original entry data"
  - "reconcile_with_broker adds unknown bonds to Core layer with zero entry data"
  - "make_bond_broker shares AsyncClient to avoid second gRPC connection"

patterns-established:
  - "Bond position records are frozen dataclasses (immutable entry conditions)"
  - "ORM round-trip via to_orm_rows/from_orm_rows class methods on domain objects"
  - "Reconciliation function is standalone (not a method) for testability"

requirements-completed: [BEX-03, BEX-04, BEX-06]

# Metrics
duration: 7min
completed: 2026-03-14
---

# Phase 04 Plan 01: Bond Infrastructure Summary

**BondPositionRecord schema, dirty-price DV01 sizing fix with transaction costs, LayerLedger DB persistence with ORM round-trip, startup reconciliation with Telegram alerts, and shared-client bond broker factory**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-14T18:47:10Z
- **Completed:** 2026-03-14T18:54:10Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- BondPositionRecord frozen dataclass with entry YTM, price, clean price, date, layer_id
- DV01BudgetStep and EqualWeightBondSizer use dirty price (unit_cost) + transaction costs instead of face value
- LayerLedger extended with bond_positions dict, add/remove, ORM round-trip via LayerLedgerModel
- reconcile_with_broker handles unknown/mismatched/missing positions with Telegram alerts
- make_bond_broker factory shares equity broker's AsyncClient for single gRPC connection
- InstrumentRegistry.get_by_figi added for FIGI-based instrument lookup
- 68 tests passing across 3 test files

## Task Commits

Each task was committed atomically:

1. **Task 1: BondPositionRecord + DV01 dirty-price fix** - `dcdd200` (test: RED), `2e67f9e` (feat: GREEN)
2. **Task 2: LayerLedger persistence + reconciliation + broker factory** - `6c69f80` (test: RED), `254b0d1` (feat: GREEN)

_TDD tasks have two commits each (test -> feat)_

## Files Created/Modified
- `src/finalayze/core/schemas.py` - Added BondPositionRecord frozen dataclass
- `src/finalayze/core/layer_ledger.py` - Bond positions, ORM round-trip, reconciliation
- `src/finalayze/core/models.py` - LayerLedgerModel ORM (composite PK: layer_id + symbol)
- `src/finalayze/risk/dv01_sizing.py` - face_value -> unit_cost + transaction_costs_per_unit
- `src/finalayze/execution/tinkoff_broker.py` - make_bond_broker factory
- `src/finalayze/markets/instruments.py` - get_by_figi method
- `tests/unit/test_dv01_sizing.py` - BondPositionRecord + dirty-price tests
- `tests/unit/test_layer_ledger.py` - Bond positions, ORM, reconciliation tests
- `tests/unit/test_broker_router.py` - moex_bonds routing + make_bond_broker tests

## Decisions Made
- Renamed `face_value` parameter to `unit_cost` with default 1000 for backward compatibility
- BondPositionRecord merge on add: sums quantities, preserves original entry conditions
- reconcile_with_broker adds unknown bonds to Core layer with zeroed entry data (will be filled on next pricing)
- make_bond_broker shares AsyncClient (single gRPC channel) to avoid connection overhead

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added InstrumentRegistry.get_by_figi method**
- **Found during:** Task 2 (reconciliation implementation)
- **Issue:** reconcile_with_broker needs FIGI -> symbol lookup; InstrumentRegistry only had get(symbol, market_id)
- **Fix:** Added get_by_figi(figi) method that iterates registered instruments
- **Files modified:** src/finalayze/markets/instruments.py
- **Verification:** mypy passes, reconciliation tests pass
- **Committed in:** 254b0d1 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for reconciliation to work. No scope creep.

## Issues Encountered
- Initial DV01 dirty-price test used parameters where DV01 budget cap was binding (not position-size cap), so face_value vs unit_cost made no difference. Fixed test parameters to ensure position-size cap is the binding constraint.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- BondPositionRecord, LayerLedger persistence, and reconciliation are ready for Plan 02 (BondCycleProcessor)
- make_bond_broker factory ready for startup wiring
- DV01 sizing uses dirty price for accurate cash sufficiency

---
*Phase: 04-bond-execution*
*Completed: 2026-03-14*

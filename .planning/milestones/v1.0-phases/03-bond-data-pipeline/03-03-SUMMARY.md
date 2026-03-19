---
phase: 03-bond-data-pipeline
plan: 03
subsystem: data
tags: [t-invest, bond-discovery, candle-cache, coupon-events, orm, grpc]

# Dependency graph
requires:
  - phase: 03-bond-data-pipeline
    provides: "BondInfo schema, QuantLib wrapper, MacroSnapshotModel ORM"
provides:
  - "BondDiscoveryService with 6-step filter chain and segment classification"
  - "TinkoffFetcher.fetch_all_bonds() for bulk bond metadata retrieval"
  - "TinkoffFetcher.fetch_bond_candles() for daily OHLCV bond data"
  - "CouponEvent emission via EventBus on ex-coupon dates"
  - "BondCandleModel, CouponScheduleModel, AmortizationEventModel ORM models"
  - "Bond candle cache with incremental daily append"
affects: [04-bond-execution, 05-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [bond-discovery-filter-chain, candle-cache-daily-append, coupon-event-emission]

key-files:
  created:
    - src/finalayze/data/bond_discovery.py
    - tests/unit/test_bond_discovery.py
    - tests/unit/test_bond_candle_fetch.py
    - tests/unit/test_bond_discovery_task1.py
  modified:
    - src/finalayze/data/fetchers/tinkoff_data.py
    - src/finalayze/core/events.py
    - src/finalayze/core/schemas.py
    - src/finalayze/core/models.py

key-decisions:
  - "liquidity_flag used as proxy for 10M RUB/day turnover threshold (T-Invest does not expose raw turnover)"
  - "6-step filter chain ordered by cost: free metadata filters first, then API-dependent"
  - "OFZ classification by class_code (TQOB/TQOD) or sector containing 'government'"
  - "Bond candle prices in % of face value (MOEX convention, same as bond_math.py)"
  - "CouponEvent emitted on record_date match (not coupon_date)"

patterns-established:
  - "Filter chain ordering: free metadata filters first, API-dependent last"
  - "Error isolation: individual bond fetch failure does not block others"
  - "Incremental candle cache: check latest date, fetch only newer candles"
  - "Event bus optional (None for backtests): backward-compatible coupon emission"

requirements-completed: [BDP-01, BDP-05]

# Metrics
duration: 9min
completed: 2026-03-14
---

# Phase 3 Plan 03: Bond Discovery & Candle Cache Summary

**Bond auto-discovery pipeline with 6-step filter chain, OFZ/corporate segment classification, candle cache with daily append, and coupon event emission via EventBus**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-14T17:51:28Z
- **Completed:** 2026-03-14T18:00:13Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments
- TinkoffFetcher extended with fetch_all_bonds(), fetch_amortization_schedule(), and fetch_bond_candles()
- BondDiscoveryService applies 6 filters (maturity, risk, currency, tradeable, perpetual, liquidity) and classifies bonds into ru_ofz/ru_corporate segments
- CouponEvent schema and STREAM_COUPONS EventBus constant for coupon event infrastructure
- BondCandleModel, CouponScheduleModel, AmortizationEventModel ORM models for TimescaleDB caching
- Bond candle cache with incremental daily append and per-bond error isolation
- 72 tests passing across all affected test files with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Add fetch_all_bonds, CouponEvent, STREAM_COUPONS, ORM models** - `4793ef8` (feat)
2. **Task 2: BondDiscoveryService with filters, registry, coupon events** - `1fd6714` (feat)
3. **Task 3: fetch_bond_candles and candle cache population** - `b5891c4` (feat)

_TDD: each task had RED (failing test) then GREEN (implementation) phases_

## Files Created/Modified
- `src/finalayze/data/bond_discovery.py` - BondDiscoveryService, DiscoveryResult, register_discovered_bonds, coupon event emission, candle cache population
- `src/finalayze/data/fetchers/tinkoff_data.py` - fetch_all_bonds(), fetch_amortization_schedule(), fetch_bond_candles()
- `src/finalayze/core/schemas.py` - CouponEvent schema (frozen, with is_floating)
- `src/finalayze/core/events.py` - STREAM_COUPONS constant
- `src/finalayze/core/models.py` - BondCandleModel, CouponScheduleModel, AmortizationEventModel
- `tests/unit/test_bond_discovery.py` - 19 tests for discovery, filters, classification, registration, events
- `tests/unit/test_bond_candle_fetch.py` - 6 tests for candle fetching and cache population
- `tests/unit/test_bond_discovery_task1.py` - 12 tests for schemas, constants, ORM models, method existence

## Decisions Made
- liquidity_flag used as proxy for 10M RUB/day turnover threshold (T-Invest API does not expose raw daily turnover in bond metadata)
- OFZ classification by class_code in (TQOB, TQOD) or sector containing "government"
- CouponEvent emitted on record_date == today (not coupon_date), as record_date is the ex-coupon date
- Bond candle prices stored as percentage of face value (MOEX convention)
- fetch_bond_candles handles gRPC errors gracefully (returns empty list, logs warning)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Bond discovery pipeline complete: can discover, filter, classify, and register MOEX bonds
- Candle cache ready for BondCycleProcessor (Phase 4) to consume
- CouponEvent infrastructure ready for coupon-aware trading strategies
- All ORM models ready for TimescaleDB hypertable creation in deployment

---
*Phase: 03-bond-data-pipeline*
*Completed: 2026-03-14*

---
phase: 05-integration-and-telegram
plan: 02
subsystem: trading-loop
tags: [bond-cycle, preflight, circuit-breaker, daily-pnl, asyncio-lock, timescaledb]

# Dependency graph
requires:
  - phase: 05-integration-and-telegram
    plan: 01
    provides: TelegramMessageQueue, refactored TelegramAlerter with HTML, priority queue
  - phase: 04-bond-execution
    provides: BondCycleProcessor, LayerLedger, bond broker wiring
provides:
  - Bond cycle holiday/hours gating in TradingLoop
  - Preflight checks with independent degradation (bond disabled, equity continues)
  - Daily P&L with US/MOEX equity/MOEX bonds separation
  - DailyEquitySnapshot model for TimescaleDB persistence
  - asyncio.Lock for gRPC client serialization
  - Bond layer circuit breaker integration tests
  - Updated on_daily_summary with top movers and dual currency
affects: [05-03-sandbox-testing]

# Tech tracking
tech-stack:
  added: []
  patterns: [preflight-independent-degradation, bond-cycle-gating, grpc-lock-serialization, ledger-based-bond-pnl]

key-files:
  created:
    - tests/unit/test_preflight.py
    - tests/unit/test_daily_pnl.py
  modified:
    - src/finalayze/core/trading_loop.py
    - src/finalayze/core/alerts.py
    - src/finalayze/core/models.py
    - config/settings.py
    - tests/unit/test_trading_loop_holidays.py
    - tests/unit/test_trading_loop_bonds.py
    - tests/integration/test_circuit_breaker_integration.py

key-decisions:
  - "Bond cycle skip logs via structlog only (no Telegram alert) per user decision"
  - "Preflight independent degradation: bond disabled on failure, equity continues"
  - "Bond P&L from LayerLedger.current_equity (not live broker prices, per research recommendation)"
  - "asyncio.Lock per TradingLoop instance for gRPC serialization"
  - "on_daily_summary extended with backward-compatible optional params (top_movers, total_equity_rub)"
  - "DailyEquitySnapshot model ready for DB wiring (scaffold for async persistence)"

patterns-established:
  - "Preflight pattern: check gRPC, macro, ledger before scheduling bond cycle"
  - "Independent degradation: component failure disables subsystem, not whole system"
  - "Bond P&L via ledger aggregation, not broker API (avoids timing issues)"

requirements-completed: [AUT-01, AUT-02, AUT-03, MON-02]

# Metrics
duration: 8min
completed: 2026-03-14
---

# Phase 5 Plan 2: Trading Loop Bond Integration Summary

**Bond cycle gating with MOEX holiday/hours checks, preflight with independent degradation, daily P&L separating US/MOEX equity/bonds, asyncio.Lock gRPC serialization, and DailyEquitySnapshot model**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-14T20:04:40Z
- **Completed:** 2026-03-14T20:12:45Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Bond cycle gates on MOEX holidays + market hours (structlog only, no Telegram per user decision)
- Preflight checks (gRPC, macro, ledger) with independent degradation -- bond disabled, equity continues
- Daily P&L separates US, MOEX equity, MOEX bonds with bond P&L from LayerLedger
- on_daily_summary extended with top 3 movers and dual currency (RUB/USD) display
- DailyEquitySnapshot model for TimescaleDB start-of-day snapshots
- asyncio.Lock serializes concurrent gRPC calls between equity and bond cycles
- Bond layer circuit breaker integration tests (independent of equity layer)
- 43 tests passing across 5 test files

## Task Commits

Each task was committed atomically (TDD: test then feat):

1. **Task 1: Bond cycle gating, preflight, asyncio.Lock, DailyEquitySnapshot, bond CB tests** - `10521b9` (test), `535d0c3` (feat)
   - Bond cycle holiday/hours gating with is_moex_trading_day
   - _preflight_check with independent degradation
   - _grpc_lock asyncio.Lock in __init__
   - DailyEquitySnapshot model
   - Settings: bond_cycle_minutes, telegram_webhook_secret, weekly_digest_hour_utc
   - 4 bond layer circuit breaker integration tests

2. **Task 2: Daily P&L bond separation, snapshots, top movers, dual currency** - `e7849e1` (test), `e4b53c6` (feat)
   - _daily_reset separates US/MOEX equity/MOEX bonds
   - Bond P&L from LayerLedger.current_equity aggregation
   - _persist_equity_snapshots and _load_baseline_from_db scaffolding
   - _compute_top_movers returns top 3 by absolute %
   - on_daily_summary extended with top_movers and total_equity_rub

## Files Created/Modified
- `src/finalayze/core/trading_loop.py` - Bond cycle gating, preflight, daily P&L fix, gRPC lock, snapshot persistence
- `src/finalayze/core/alerts.py` - on_daily_summary extended with top_movers and total_equity_rub
- `src/finalayze/core/models.py` - DailyEquitySnapshot model
- `config/settings.py` - bond_cycle_minutes, telegram_webhook_secret, telegram_allowed_chat_ids, weekly_digest_hour_utc
- `tests/unit/test_preflight.py` - 5 preflight check tests
- `tests/unit/test_daily_pnl.py` - 7 daily P&L tests
- `tests/unit/test_trading_loop_holidays.py` - 4 bond cycle holiday gating tests
- `tests/unit/test_trading_loop_bonds.py` - 5 bond cycle integration tests
- `tests/integration/test_circuit_breaker_integration.py` - 4 bond layer circuit breaker tests

## Decisions Made
- Bond cycle skip uses structlog only (no Telegram alert) per user decision from 05-CONTEXT
- Preflight independent degradation: bond disabled on failure, equity cycle unaffected
- Bond P&L computed from LayerLedger.current_equity (not broker portfolio) per research recommendation
- on_daily_summary params are backward-compatible (optional kwargs with defaults)
- _persist_equity_snapshots and _load_baseline_from_db are scaffolds (async DB session wiring deferred to sandbox testing)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TradingLoop fully wired with bond cycle gating, preflight, and daily P&L separation
- Ready for Plan 05-03 (sandbox end-to-end testing)
- DB snapshot persistence scaffold in place -- needs async session wiring during sandbox integration
- All 43 tests passing, lint clean

---
*Phase: 05-integration-and-telegram*
*Completed: 2026-03-14*

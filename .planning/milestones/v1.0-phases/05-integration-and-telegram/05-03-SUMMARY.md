---
phase: 05-integration-and-telegram
plan: 03
subsystem: telegram-monitoring
tags: [telegram, webhook, bot-commands, cbr-alerts, coupon-alerts, weekly-digest]

# Dependency graph
requires:
  - phase: 05-integration-and-telegram
    plan: 01
    provides: TelegramAlerter with priority queue, HTML formatting, persistent httpx client
  - phase: 05-integration-and-telegram
    plan: 02
    provides: TradingLoop with bond cycle, preflight, daily P&L, DailyEquitySnapshot
provides:
  - TelegramBotHandler with /status and /breakers read-only commands
  - FastAPI webhook endpoint with secret token validation and chat_id whitelist
  - CBR meeting alerts with rate decision in _cbr_day_refresh
  - Coupon reinvestment alerts in BondCycleProcessor._process_layer
  - Weekly digest scheduled on Sunday via CronTrigger
affects: [06-sandbox-testing]

# Tech tracking
tech-stack:
  added: []
  patterns: [webhook-secret-validation, chat-id-whitelist, factory-router-pattern, cron-scheduled-digest]

key-files:
  created:
    - src/finalayze/core/telegram_bot.py
    - src/finalayze/api/v1/telegram.py
    - tests/unit/test_telegram_webhook.py
  modified:
    - src/finalayze/api/v1/router.py
    - src/finalayze/core/trading_loop.py
    - src/finalayze/core/bond_cycle.py
    - tests/unit/test_trading_loop_bonds.py
    - tests/unit/test_bond_cycle.py

key-decisions:
  - "Factory router pattern for telegram webhook (create_telegram_router) -- requires runtime dependencies"
  - "Read-only commands only (/status, /breakers) -- no trading via Telegram per user decision"
  - "chat_id as string comparison against whitelist (Telegram sends int, we convert)"
  - "CBR alert fires on_cbr_meeting with rate from MacroSnapshot.last_cbr_decision"
  - "Coupon alert fires on reinvestment step in _process_layer (not discrete coupon events)"
  - "Weekly digest uses CronTrigger(day_of_week='sun') at configurable hour"

patterns-established:
  - "Webhook secret validation via X-Telegram-Bot-Api-Secret-Token header"
  - "Factory router for endpoints needing runtime dependencies"
  - "Source-level inspection tests for wiring verification"

requirements-completed: [MON-04, MON-05]

# Metrics
duration: 7min
completed: 2026-03-14
---

# Phase 5 Plan 3: Telegram Webhook & Monitoring Alerts Summary

**Telegram webhook with /status and /breakers commands, CBR meeting and coupon alerts wired into TradingLoop, and weekly digest scheduled on Sunday evening**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-14T20:15:45Z
- **Completed:** 2026-03-14T20:23:14Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- TelegramBotHandler with /status (portfolio equity, positions, bond layers) and /breakers (circuit breaker levels, bond layer breakers)
- FastAPI webhook endpoint with X-Telegram-Bot-Api-Secret-Token validation (403 on mismatch)
- Chat ID whitelist -- non-whitelisted chat_ids silently ignored
- CBR meeting alerts fire with rate decision after macro refresh in _cbr_day_refresh
- Error alert when macro data stale after CBR day refresh
- Coupon reinvestment alerts fire on_coupon_received in BondCycleProcessor._process_layer
- Weekly digest scheduled via CronTrigger on Sunday at weekly_digest_hour_utc
- Weekly digest includes week P&L, per-market breakdown, bond layer P&L, top movers
- 38 tests passing across 3 test files (9 webhook + 11 trading loop bonds + 18 bond cycle)

## Task Commits

Each task was committed atomically (TDD: test then feat):

1. **Task 1: Telegram webhook with /status and /breakers** - `7a596b6` (test), `614eab0` (feat)
   - TelegramBotHandler class with command dispatch
   - FastAPI webhook endpoint (create_telegram_router factory)
   - Secret validation, chat_id whitelist, HTML-formatted responses
   - 9 unit tests: secret validation, whitelist, command dispatch, malformed JSON

2. **Task 2: CBR/coupon alerts and weekly digest** - `b04a9f5` (test), `c428eb4` (feat)
   - _cbr_day_refresh fires on_cbr_meeting with rate decision
   - _cbr_day_refresh sends error alert on stale macro data
   - BondCycleProcessor fires on_coupon_received on coupon reinvestment
   - _weekly_digest method with CronTrigger scheduling
   - 6 new tests: CBR alerts, weekly digest existence/scheduling/alert

## Files Created/Modified
- `src/finalayze/core/telegram_bot.py` - TelegramBotHandler with /status, /breakers commands
- `src/finalayze/api/v1/telegram.py` - FastAPI webhook endpoint with secret validation
- `src/finalayze/api/v1/router.py` - Comment noting telegram router mounted at app startup
- `src/finalayze/core/trading_loop.py` - CBR alert wiring, weekly digest method + scheduling
- `src/finalayze/core/bond_cycle.py` - Coupon reinvestment alert in _process_layer
- `tests/unit/test_telegram_webhook.py` - 9 tests for webhook and bot handler
- `tests/unit/test_trading_loop_bonds.py` - 6 new tests for CBR alerts and weekly digest
- `tests/unit/test_bond_cycle.py` - 1 new test for coupon alert

## Decisions Made
- Factory router pattern for telegram endpoint (needs runtime bot_handler + secret)
- Read-only commands only -- no trading via Telegram per user decision from 05-CONTEXT
- CBR alert extracts decision from MacroSnapshot.last_cbr_decision after refresh
- Coupon alert fires on coupon cash reinvestment step (Layer:name as symbol identifier)
- Weekly digest uses configurable weekly_digest_hour_utc setting (default 16 = 19:00 MSK)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing test failure in test_pairs_strategy.py (unrelated, pairs strategy disabled on all MOEX segments per 02-02 decision)

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 5 fully complete: Telegram priority queue, trading loop bond integration, webhook commands
- System is fully observable via Telegram without opening dashboard
- Ready for Phase 6 (sandbox end-to-end testing)
- All 38 plan tests passing, lint clean

---
*Phase: 05-integration-and-telegram*
*Completed: 2026-03-14*

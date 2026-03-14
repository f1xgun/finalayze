# Phase 5: Integration and Telegram - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire equity and bond cycles into TradingLoop as concurrent APScheduler jobs with reliable Telegram alerting for all trade events. Fix daily P&L summary (currently shows zero), add priority message queue with rate limiting, implement interactive bot commands, and gate cycles on MOEX holidays and market hours. Prove that both cycles run concurrently without gRPC errors and that circuit breakers fire correctly for both equity and bond layers.

</domain>

<decisions>
## Implementation Decisions

### Telegram Rate Limiting & Queue
- 3-tier priority queue: CRITICAL (circuit breaker, errors) > IMPORTANT (fills, stop-loss) > INFO (daily summary, coupon, CBR)
- CRITICAL alerts sent immediately; lower tiers queued with Telegram rate limiting
- During bursts: batch 5+ pending fills into single digest message ("5 fills executed: SBER +10, GAZP +5, ...")
- All messages get one retry after 5s on failure (no complex backoff)
- HTML formatting for messages (Telegram parse_mode="HTML") — bold for symbols, monospace for prices, existing emoji prefixes kept

### Interactive Bot Commands
- /status command returns current portfolio state (positions + P&L)
- /breakers command returns circuit breaker states for all layers
- Read-only commands only — no trading commands via Telegram
- Webhook transport: add /api/telegram/webhook endpoint to existing FastAPI app
- Auth: chat ID whitelist — only respond to configured chat_id, reject all others silently

### Concurrent Cycle Safety
- Shared gRPC AsyncClient between equity and bond cycles, serialized with asyncio.Lock to prevent contention
- Full isolation between cycles: each runs in its own try/except. Bond crash logs error + sends Telegram alert, equity continues unaffected
- Bond cycle frequency: configurable via bond_cycle_minutes setting (default 1440 = daily at 10:30 MSK)
- Preflight checks on startup: verify gRPC connectivity, check macro data freshness, validate LayerLedger state, send startup Telegram alert. Fail fast if critical check fails
- Independent degradation: if bond preflight fails, disable bond cycle but keep equity running. Telegram alert about degraded state

### Startup Reconciliation
- LayerLedger reconciliation runs on every startup (not just after crashes)
- Unknown positions (in T-Invest but not in ledger): add to Core layer + send Telegram alert (consistent with Phase 4 decision)

### Daily P&L Fix
- Separate bond P&L line in daily summary: "US +$342 | MOEX Equity +1,200₽ | MOEX Bonds +850₽ | Total: 2.5M₽ ($28,400)"
- Both currencies shown: native currency for P&L lines, both RUB and USD equivalent for total equity (uses FXRateService)
- P&L computation: snapshot diff method — store portfolio equity at start of day, P&L = current - start_of_day
- Start-of-day equity snapshots persisted in TimescaleDB (reuse async SQLAlchemy pattern from MacroSnapshot)
- Include top 3 movers in daily summary: "Top: SBER +2.1%, GAZP -0.8%, SU26244 +0.3%"
- Weekly digest on Sunday evening: week P&L, equity curve direction, trade count, best/worst positions, circuit breaker event count

### MOEX Holiday Gating
- Gate inside cycle body (not APScheduler trigger): bond cycle fires on schedule, first line checks is_moex_holiday(). If holiday → log, skip, return
- Both equity and bond cycles check MOEX holidays: equity cycle skips MOEX instruments on MOEX holidays but still trades US instruments
- Gate on market hours too (10:00-18:45 MSK): protects against configurable schedules that fire outside trading hours
- Macro refresh runs 7 days/week regardless (no holiday gate)
- Holiday/hours skips: structlog only, no Telegram alert (expected behavior, not worth notifying)
- Unexpected skips (e.g., macro data missing): Telegram alert

### Claude's Discretion
- Exact priority queue implementation (asyncio.PriorityQueue vs custom)
- Telegram message templates and formatting details
- Preflight check timeout values
- Weekly digest scheduling (Sunday evening exact time)
- Async lock granularity (per-method vs per-request)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TelegramAlerter` (`core/alerts.py`): 12 alert methods already implemented (fills, stops, breakers, daily, coupon, CBR, startup, shutdown, error). Needs queue wrapper and HTML formatting
- `TradingLoop` (`core/trading_loop.py`): Bond cycle already scheduled via CronTrigger when bond_cycle_enabled=True. Has preflight-like structure (DI constructor)
- `BondCycleProcessor` (`core/bond_cycle.py`): Fully implemented run_cycle() with layer processing. Ready for TradingLoop integration
- `is_moex_holiday()` (`markets/moex_calendar.py`): Static per-year holiday frozensets. Ready to use for gating
- `FXRateService` (`markets/fx_service.py`): USD/RUB conversion available for P&L display
- `MacroCacheService` (`data/macro_cache.py`): Async SQLAlchemy persistence pattern — reuse for equity snapshots
- FastAPI app (`api/`): 20+ endpoints, authentication via X-API-Key. Add webhook endpoint here

### Established Patterns
- APScheduler BackgroundScheduler with named executors ("default" 4 threads, "retrain" 1 thread)
- Async operations via `_run_async()` bridge (asyncio.run_coroutine_threadsafe to dedicated event loop thread)
- Fire-and-forget Telegram via asyncio.create_task — needs replacement with queue
- Settings class (Pydantic) for configuration — add bond_cycle_minutes, telegram webhook settings
- `moex_calendar` imported lazily in trading_loop to maintain dependency layering

### Integration Points
- `TradingLoop.__init__()` already accepts `bond_cycle_processor: BondCycleProcessor | None`
- `TradingLoop.start()` schedules bond cycle when `bond_cycle_enabled=True`
- `Settings` needs: `bond_cycle_minutes`, `telegram_webhook_secret`, `telegram_allowed_chat_ids`
- `_daily_reset()` method is where start-of-day snapshot should be stored
- FastAPI app needs `/api/telegram/webhook` POST endpoint
- Circuit breaker alerting already calls `self._alerter.on_circuit_breaker_trip()` — needs to work with queue

</code_context>

<specifics>
## Specific Ideas

- Telegram Bot API rate limit is 30 messages per second per bot, 20 messages per minute to same group chat — the queue must respect the per-chat limit
- T-Invest gRPC channels are thread-safe for concurrent reads but not for concurrent write+read — the async lock is appropriate
- The existing `_strategy_cycle` already has MOEX market hours constants (`_MOEX_OPEN_UTC`, `_MOEX_CLOSE_UTC`) — reuse for gating
- Bond carry strategy on OFZ-PK is expected to be low-frequency (trades ~monthly) so burst scenarios are mainly from equity side
- Weekly digest should run even if system was restarted mid-week — use TimescaleDB snapshots to reconstruct week

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-integration-and-telegram*
*Context gathered: 2026-03-14*

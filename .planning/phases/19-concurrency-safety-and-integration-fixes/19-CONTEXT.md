# Phase 19: Concurrency Safety and Integration Fixes - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix critical concurrency bugs in the trading system: stop-loss TOCTOU double-sell race, TinkoffBroker threading.Lock in async code, event loop TOCTOU race, macro_cache session leak. Close v3.0 integration gaps (Telegram /gonogo import, HealthMonitor feed freshness wiring).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase.
Key constraints from audit findings:
- Stop-loss: read-check-sell-remove must be atomic under one lock per symbol
- TinkoffBroker: asyncio.Lock for async paths, threading.Lock only for sync _get_client()
- Event loop: thread-safe lazy init pattern (threading.Lock guard on _loop creation)
- macro_cache: async with session_factory() as session + rollback in except
- INT-01: Fix import in Telegram /gonogo handler (OPS-04)
- INT-02: Call update_feed_timestamp() in TradingLoop after data fetch

</decisions>

<code_context>
## Existing Code Insights

### Key Files to Modify
- `src/finalayze/core/trading_loop.py:1586-1612` — stop-loss TOCTOU (self._stop_loss_lock)
- `src/finalayze/execution/tinkoff_broker.py:100-112` — threading.Lock in async _get_services_async
- `src/finalayze/execution/tinkoff_broker.py:148-154` — TOCTOU on _loop creation
- `src/finalayze/data/macro_cache.py:98-100` — session leak in _persist_snapshot
- `src/finalayze/core/telegram_bot.py` — /gonogo import fix (OPS-04)
- `src/finalayze/core/trading_loop.py` — HealthMonitor.update_feed_timestamp() wiring (OPS-02)

### Established Patterns
- `_sentiment_lock` and `_stop_loss_lock` are threading.Lock instances in trading_loop.py
- TinkoffBroker uses `_client_lock = threading.Lock()` and a background asyncio event loop
- macro_cache uses `self._db_session_factory` callable returning async sessions
- HealthMonitor has `update_feed_timestamp(market_id)` method ready to be called

### Integration Points
- Stop-loss check called from `_check_stop_losses` in strategy cycle
- TinkoffBroker used by broker_router for all Tinkoff operations
- macro_cache persistence triggered from `_persist_snapshot` via fire-and-forget task

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

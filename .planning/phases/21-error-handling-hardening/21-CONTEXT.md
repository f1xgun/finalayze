# Phase 21: Error Handling Hardening - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix silent failure modes: GARCH NaN propagation into sizing, EventBus suppress(Exception), unlogged Tinkoff data failures, no alerting on consecutive trading loop errors, no alerting on consecutive bond cycle gRPC failures. Add authentication to POST /kill endpoint.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase.
Key constraints from audit findings:
- ERR-01: GARCH failure returns historical rolling volatility (e.g., 20-day std), logs warning, never returns NaN
- ERR-02: EventBus.create_group catches redis.ResponseError only, not bare Exception
- ERR-03: TinkoffFetcher logs with structlog bind(ticker=, timeframe=, error_type=)
- ERR-04: TradingLoop adds _consecutive_errors counter per cycle type, Telegram alert after 3 failures
- ERR-05: BondCycleProcessor adds per-cycle error counter, escalates to _log.error after threshold
- API-01: POST /kill checks X-API-Key header via existing api_key_auth dependency

</decisions>

<code_context>
## Existing Code Insights

### Key Files to Modify
- `src/finalayze/risk/garch.py:95` — returns NaN on failure
- `src/finalayze/core/events.py:112` — contextlib.suppress(Exception) in create_group
- `src/finalayze/data/fetchers/tinkoff_data.py:238,316,377` — silent failures
- `src/finalayze/core/trading_loop.py` — no consecutive error counter
- `src/finalayze/core/bond_cycle.py` — 16 except Exception, no aggregate counter
- `src/finalayze/api/v1/system.py:409` — POST /kill without auth

### Established Patterns
- structlog used throughout with bind() for structured fields
- TelegramAlerter has send_alert(message, priority) for alert dispatch
- API auth uses api_key_auth dependency from api/auth.py
- Circuit breaker already has escalation levels (NORMAL, HALTED, LIQUIDATE)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

# Phase 30: Broker Resilience - Context

**Gathered:** 2026-03-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Add resilience to T-Bank gRPC broker: auto-reconnect on error 70001, last-known portfolio fallback, and CBR XML FX rate fallback when gRPC fails.

</domain>

<decisions>
## Implementation Decisions

### gRPC Channel Reconnect (GRPC-02)

- On StatusCode.INTERNAL (error 70001), close and re-create gRPC channel before next retry
- Add reconnect logic inside TinkoffBroker._run_async error handler
- Channel recreation must use the injected grpc_loop (from Phase 29)
- Limit reconnection attempts (3 max per cycle) to prevent infinite retry
- Log structured event: "grpc_channel_reconnected" with attempt count

### Portfolio Cache Fallback (GRPC-03)

- Cache last successful PortfolioState in TinkoffBroker._last_portfolio
- On portfolio fetch failure, return cached state with warning log
- Add timestamp to cached portfolio to track staleness
- Strategy cycle continues with cached positions instead of skipping
- Log structured event: "portfolio_using_cached" with cache_age_seconds

### FX Rate Fallback (OBS-03)

- CBR XML API endpoint: https://www.cbr.ru/scripts/XML_daily.asp
- FXRateService or fx_service.py already has CBR XML parsing — wire as fallback
- In TradingLoop._fx_update_cycle: try gRPC first, fallback to CBR XML on failure
- Log structured event: "fx_rate_cbr_fallback" with rate value
- Ensure finalayze_usd_rub_rate Prometheus metric gets updated from either source

### Claude's Discretion

- Whether reconnect creates fresh AsyncClient or resets existing channel
- Exact cache eviction policy for _last_portfolio (time-based vs on-success)
- HTTP timeout for CBR XML API call

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/finalayze/execution/tinkoff_broker.py` — TinkoffBroker with _run_async, grpc_loop injection
- `src/finalayze/markets/fx_service.py` — FXRateService with CBR XML parsing
- `src/finalayze/execution/retry.py` — RetryPolicy with exponential backoff

### Established Patterns
- Fire-and-forget error handling — never crash trading loop
- Structured logging with structlog for all broker events
- RetryPolicy wraps transient errors, skips fatal ones

### Integration Points
- `src/finalayze/orchestration/trading_loop.py` — _strategy_cycle uses get_portfolio(), _fx_update_cycle
- `scripts/run_sandbox.py` — FXRateService creation and wiring

</code_context>

<specifics>
## Specific Ideas

From sandbox analysis: 62 portfolio_fetch_failed events, FX rate = 0.0 throughout run.
Error 70001 caused multi-hour blind windows during MOEX market hours.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

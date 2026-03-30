# Phase 29: Core Stability - Context

**Gathered:** 2026-03-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the root cause of strategy cycle drift (gRPC BlockingIOError flooding asyncio loop) by consolidating gRPC onto a dedicated event loop thread. Fix Loki log pipeline (Promtail not shipping logs).

</domain>

<decisions>
## Implementation Decisions

### gRPC Loop Isolation

- Consolidate TinkoffBroker._loop and TinkoffFetcher._loop into a single shared _grpc_loop on TradingLoop
- TradingLoop creates _grpc_loop (dedicated daemon thread), passes it to both TinkoffBroker and TinkoffFetcher
- TinkoffBroker and TinkoffFetcher stop creating their own loops — accept loop parameter in constructor
- _async_loop remains for non-gRPC async work (httpx, SQLAlchemy, Telegram)
- _run_async() gets optional loop parameter to route calls to _grpc_loop vs _async_loop
- Add asyncio exception handler on _grpc_loop to suppress benign BlockingIOError from PollerCompletionQueue
- Verify asyncio.Lock still works correctly (lock created on one loop, awaited on same loop)

### Loki Pipeline Fix

- Mount /var/lib/docker/containers as read-only volume in Promtail service in docker-compose.sandbox.yml
- Fix Promtail pipeline_stages to correctly parse JSON logs from structlog (not expecting INFO: prefix)
- Verify Loki retention is set to 30 days in loki-config.yml
- Verify Grafana datasource points to correct Loki URL

### Claude's Discretion

- Exact method of passing _grpc_loop to TinkoffBroker/TinkoffFetcher (constructor param vs setter)
- Whether to add a health check for the gRPC loop thread liveness
- Promtail relabel_configs specifics for label extraction from JSON

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TradingLoop._async_loop` pattern in `orchestration/trading_loop.py` — background thread + event loop
- `TinkoffBroker._loop` in `execution/tinkoff_broker.py` — same pattern, can be unified
- `TinkoffFetcher._loop` in `data/fetchers/tinkoff_data.py` — same pattern, can be unified
- `_run_async()` in TradingLoop — submits coroutines to background loop

### Established Patterns
- Background event loop: `threading.Thread(target=loop.run_forever, daemon=True)`
- gRPC channel created inside the loop thread to bind PollerCompletionQueue to that loop
- `concurrent.futures.Future` for cross-thread result passing

### Integration Points
- `scripts/run_sandbox.py` creates TinkoffBroker and TinkoffFetcher — must pass shared loop
- `docker/docker-compose.sandbox.yml` — Promtail volumes and config
- `monitoring/promtail/promtail-config.yml` — pipeline stages
- `monitoring/loki/loki-config.yml` — retention config

</code_context>

<specifics>
## Specific Ideas

Research found that GRPC_POLL_STRATEGY env var will NOT fix this — the issue is multiple loops, not the poll strategy.
The key insight: consolidate to 2 loops (general async + dedicated gRPC), not try to make 3 loops coexist.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

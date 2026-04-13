# Phase 20: Async Correctness and Resource Management - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix all blocking calls in async paths (time.sleep in APScheduler, asyncio.run in monitors), fix RetryPolicy.aexecute coroutine discard bug, add run_in_executor for sync broker calls in FastAPI, and add explicit lifecycle management for gRPC channels and httpx clients.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase.
Key constraints from audit findings:
- ASYNC-01: Replace time.sleep(300) in _attempt_grpc_reconnect with asyncio.sleep or background thread
- ASYNC-02: RetryPolicy.aexecute() must await fn() when it returns a coroutine
- ASYNC-03: Portfolio API endpoint wraps broker.get_portfolio() with run_in_executor
- ASYNC-04: SandboxMonitorService._run_async_safe() should use asyncio event loop, not asyncio.run()
- RES-01: TinkoffBroker.close() logs exceptions instead of suppress(Exception)
- RES-02: TinkoffFetcher gRPC calls wrapped with asyncio.wait_for(timeout=60)
- RES-03: httpx clients in alerts.py closed during app shutdown (add close() method or aexit)

</decisions>

<code_context>
## Existing Code Insights

### Key Files to Modify
- `src/finalayze/core/trading_loop.py:294` — time.sleep(300) in _attempt_grpc_reconnect
- `src/finalayze/execution/retry.py:100` — aexecute calls fn() without await
- `src/finalayze/api/v1/portfolio.py:106` — sync broker.get_portfolio() in async endpoint
- `src/finalayze/monitoring/sandbox_monitor.py:117-123` — asyncio.run() in APScheduler thread
- `src/finalayze/execution/tinkoff_broker.py:124-138` — close() with suppress(Exception)
- `src/finalayze/data/fetchers/tinkoff_data.py:111` — asyncio.run() without timeout
- `src/finalayze/core/alerts.py:199` — httpx.AsyncClient never closed

### Established Patterns
- RetryPolicy has both sync execute() and async aexecute() — dual interface
- TradingLoop uses APScheduler BackgroundScheduler with ThreadPoolExecutor
- FastAPI endpoints are async by default
- Markets/FX service has explicit close() called during shutdown

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

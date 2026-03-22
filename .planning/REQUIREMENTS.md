# Requirements: Finalayze

**Defined:** 2026-03-22
**Core Value:** Autonomous profitable MOEX trading with acceptable risk limits

## v4.0 Requirements

Requirements for Architecture Hardening milestone. Each maps to roadmap phases.

### Concurrency Safety

- [x] **CONC-01**: Stop-loss check-and-sell is atomic under a single lock — no double-sell possible for the same symbol
- [x] **CONC-02**: TinkoffBroker uses asyncio.Lock (not threading.Lock) for async code paths, eliminating latent deadlock
- [x] **CONC-03**: TinkoffBroker event loop creation is thread-safe — no TOCTOU race on _loop initialization
- [x] **CONC-04**: macro_cache session is properly scoped with async-with and rollback on error — no connection pool leak

### Async Correctness

- [x] **ASYNC-01**: gRPC reconnect uses non-blocking sleep (asyncio.sleep or dedicated thread) — APScheduler thread pool not starved
- [x] **ASYNC-02**: RetryPolicy.aexecute() properly awaits coroutine functions — no silent coroutine discard
- [x] **ASYNC-03**: Portfolio API endpoint runs broker calls via run_in_executor — FastAPI event loop not blocked
- [x] **ASYNC-04**: sandbox_monitor uses async-safe persistence — no asyncio.run() blocking APScheduler threads

### Error Handling

- [x] **ERR-01**: GARCH failure returns historical volatility fallback (not NaN) and logs warning — NaN never propagates to sizing pipeline
- [x] **ERR-02**: EventBus.create_group suppresses only redis.ResponseError — Redis connectivity failures are logged and raised
- [x] **ERR-03**: Tinkoff data fetcher failures are logged with structured context (ticker, timeframe, error type)
- [x] **ERR-04**: trading_loop consecutive error counter triggers Telegram alert after N failures — silent degradation detected
- [x] **ERR-05**: bond_cycle per-cycle error counter escalates to error log after threshold — systematic gRPC failures visible

### Dependency Layers

- [x] **LAYER-01**: trading_loop.py and bond_cycle.py moved from core/ to dedicated orchestration module — core/ contains only L0 types
- [x] **LAYER-02**: telegram_bot.py and alerts.py moved from core/ to appropriate layer (L6 API/Dashboard)
- [ ] **LAYER-03**: MetricsCollector injected into trading loop via constructor — no direct import from L6
- [ ] **LAYER-04**: backtest/ and monitoring/ have documented layer assignments

### API Security

- [x] **API-01**: POST /kill endpoint requires X-API-Key authentication — no unauthenticated emergency shutdown

### Dead Code Cleanup

- [x] **DEAD-01**: Event bus streams (STREAM_MARKET_DATA, STREAM_SIGNALS, STREAM_EXECUTION) removed or wired to actual consumers
- [x] **DEAD-02**: Stub API endpoints (/signals, /trades, /news, /ml/status etc.) either implemented or removed with clear 501 Not Implemented response

### Resource Management

- [x] **RES-01**: TinkoffBroker.close() logs cleanup failures instead of suppressing all exceptions
- [x] **RES-02**: TinkoffFetcher gRPC calls have configurable timeout (default 60s) — no indefinite hang
- [x] **RES-03**: httpx clients in alerts.py and fetchers are explicitly closed on shutdown

### Integration Bug Fixes

- [x] **INT-01**: Telegram /gonogo import fixed (OPS-04 gap from v3.0)
- [x] **INT-02**: HealthMonitor.update_feed_timestamp() wired into TradingLoop (OPS-02 gap from v3.0)

## Future Requirements

### Performance Optimization

- **PERF-01**: APScheduler thread pool sized dynamically based on active cycles
- **PERF-02**: Connection pool metrics exposed via Prometheus

### Code Quality

- **QUAL-01**: Reduce except-Exception count from 140+ to <50 with targeted exception types
- **QUAL-02**: Add missing type annotations to core/trading_loop.py

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full trading_loop.py rewrite | Too risky mid-production; incremental extraction only |
| Event-driven architecture migration | Would require rewriting all data flow; current direct-call pattern works |
| Multi-process architecture | Single-process with APScheduler sufficient for current scale |
| New features or strategies | This milestone is purely hardening, no functional changes |
| US market fixes | MOEX-only focus continues |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CONC-01 | Phase 19 | Complete |
| CONC-02 | Phase 19 | Complete |
| CONC-03 | Phase 19 | Complete |
| CONC-04 | Phase 19 | Complete |
| ASYNC-01 | Phase 20 | Complete |
| ASYNC-02 | Phase 20 | Complete |
| ASYNC-03 | Phase 20 | Complete |
| ASYNC-04 | Phase 20 | Complete |
| ERR-01 | Phase 21 | Complete |
| ERR-02 | Phase 21 | Complete |
| ERR-03 | Phase 21 | Complete |
| ERR-04 | Phase 21 | Complete |
| ERR-05 | Phase 21 | Complete |
| LAYER-01 | Phase 22 | Complete |
| LAYER-02 | Phase 22 | Complete |
| LAYER-03 | Phase 22 | Pending |
| LAYER-04 | Phase 22 | Pending |
| API-01 | Phase 21 | Complete |
| DEAD-01 | Phase 22 | Complete |
| DEAD-02 | Phase 22 | Complete |
| RES-01 | Phase 20 | Complete |
| RES-02 | Phase 20 | Complete |
| RES-03 | Phase 20 | Complete |
| INT-01 | Phase 19 | Complete |
| INT-02 | Phase 19 | Complete |

**Coverage:**
- v4.0 requirements: 25 total
- Mapped to phases: 25
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-22*
*Last updated: 2026-03-22 after initial definition*

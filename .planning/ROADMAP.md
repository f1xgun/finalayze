# Roadmap: Finalayze

## Milestones

- ✅ **v1.0 MOEX MVP** -- Phases 1-7 (shipped 2026-03-19)
- ✅ **v2.0 MOEX Profitability** -- Phases 8-14 (shipped 2026-03-21)
- ✅ **v3.0 Production Readiness** -- Phases 15-18 (shipped 2026-03-22)
- 🚧 **v4.0 Architecture Hardening** -- Phases 19-22 (in progress)

## Phases

<details>
<summary>✅ v1.0 MOEX MVP (Phases 1-7) -- SHIPPED 2026-03-19</summary>

- [x] Phase 1: MOEX Equity Foundation (2/2 plans) -- completed 2026-03-14
- [x] Phase 2: MOEX Equity Validation (3/3 plans) -- completed 2026-03-14
- [x] Phase 3: Bond Data Pipeline (3/3 plans) -- completed 2026-03-14
- [x] Phase 4: Bond Execution (3/3 plans) -- completed 2026-03-14
- [x] Phase 5: Integration and Telegram (4/4 plans) -- completed 2026-03-14
- [x] Phase 6: Sandbox Validation (4/4 plans) -- completed 2026-03-15
- [x] Phase 7: News Pipeline and Go-Live (3/3 plans) -- completed 2026-03-15

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>✅ v2.0 MOEX Profitability (Phases 8-14) -- SHIPPED 2026-03-21</summary>

- [x] Phase 8: Data Foundation (3/3 plans) -- completed 2026-03-20
- [x] Phase 9: Strategy Wiring (2/2 plans) -- completed 2026-03-20
- [x] Phase 10: Macro Regime (2/2 plans) -- completed 2026-03-20
- [x] Phase 11: Advanced Strategies and ML (4/4 plans) -- completed 2026-03-21
- [x] Phase 12: Portfolio Assembly (2/2 plans) -- completed 2026-03-21
- [x] Phase 13: Script Wiring Fixes (1/1 plan) -- completed 2026-03-21 (gap closure)
- [x] Phase 14: Bond Backtest and Portfolio CLI (2/2 plans) -- completed 2026-03-21 (gap closure)

Full details: `.planning/milestones/v2.0-ROADMAP.md`

</details>

<details>
<summary>✅ v3.0 Production Readiness (Phases 15-18) -- SHIPPED 2026-03-22</summary>

- [x] Phase 15: Schemas, Config, and Rollout Foundation (2/2 plans) -- completed 2026-03-21
- [x] Phase 16: Sandbox Monitoring and Go/No-Go Gate (3/3 plans) -- completed 2026-03-21
- [x] Phase 17: Production Operations (3/3 plans) -- completed 2026-03-21
- [x] Phase 18: Dashboard and API Integration (2/2 plans) -- completed 2026-03-21

Full details: `.planning/milestones/v3.0-ROADMAP.md`

</details>

### v4.0 Architecture Hardening (In Progress)

**Milestone Goal:** Fix critical architectural defects discovered in comprehensive audit -- concurrency bugs that risk money loss, async correctness issues that cause silent degradation, error handling gaps that mask failures, and dependency layer violations that hinder maintainability.

- [ ] **Phase 19: Concurrency Safety and Integration Fixes** - Fix money-losing race conditions, lock misuse, session leaks, and v3.0 integration gaps
- [ ] **Phase 20: Async Correctness and Resource Management** - Fix blocking calls in async paths, coroutine discard bugs, and resource lifecycle gaps
- [ ] **Phase 21: Error Handling Hardening** - Fix NaN propagation, exception suppression, silent degradation, and API security gap
- [ ] **Phase 22: Dependency Layer Cleanup** - Extract orchestrators from core/, assign module layers, remove dead infrastructure

## Phase Details

### Phase 19: Concurrency Safety and Integration Fixes
**Goal**: Trading system has no race conditions that can cause double-sells, deadlocks, or connection pool exhaustion -- and v3.0 integration gaps are closed
**Depends on**: Phase 18 (v3.0 complete)
**Requirements**: CONC-01, CONC-02, CONC-03, CONC-04, INT-01, INT-02
**Success Criteria** (what must be TRUE):
  1. Stop-loss sell for a symbol acquires an async lock before checking position and releasing after order submission -- concurrent signals for the same symbol are serialized, eliminating double-sell
  2. TinkoffBroker uses asyncio.Lock for all async code paths -- threading.Lock is not used anywhere in async broker methods
  3. TinkoffBroker event loop initialization uses a thread-safe pattern (e.g., asyncio.get_running_loop or lazy init with threading.Lock guard) -- no TOCTOU race on _loop attribute
  4. macro_cache database session uses async-with context manager and issues rollback on exception -- connection pool leak under error conditions is eliminated
  5. Telegram /gonogo command imports and runs successfully (OPS-04 integration fix verified by test)
  6. HealthMonitor.update_feed_timestamp() is called by TradingLoop after each data fetch cycle -- feed freshness monitoring is operational
**Plans**: 2 plans
Plans:
- [x] 19-01-PLAN.md -- Fix TinkoffBroker lock types, event loop TOCTOU, and macro_cache session leak
- [ ] 19-02-PLAN.md -- Atomic stop-loss, /gonogo import verification, feed timestamp wiring

### Phase 20: Async Correctness and Resource Management
**Goal**: All async code paths are non-blocking and all external resources (gRPC channels, HTTP clients) have explicit lifecycle management
**Depends on**: Phase 19
**Requirements**: ASYNC-01, ASYNC-02, ASYNC-03, ASYNC-04, RES-01, RES-02, RES-03
**Success Criteria** (what must be TRUE):
  1. gRPC reconnect uses asyncio.sleep or a background task instead of time.sleep(300) -- APScheduler thread pool is never starved by a 5-minute blocking sleep
  2. RetryPolicy.aexecute() checks if fn() returns a coroutine and awaits it -- coroutine objects are never silently discarded
  3. Portfolio API endpoint wraps synchronous broker calls with run_in_executor -- FastAPI event loop latency is not blocked by broker I/O
  4. SandboxMonitorService persistence does not call asyncio.run() from within APScheduler threads -- no nested event loop errors
  5. TinkoffBroker.close() logs cleanup exceptions with structured context (resource name, error type) instead of bare except: pass
  6. TinkoffFetcher gRPC calls have a configurable timeout parameter (default 60s) -- no indefinite hang on unresponsive gRPC server
  7. httpx clients in alerts.py and fetcher modules are explicitly closed during application shutdown
**Plans**: TBD

### Phase 21: Error Handling Hardening
**Goal**: Failures in GARCH, EventBus, data fetchers, and trading loops are visible through logs and alerts -- no silent degradation or NaN propagation
**Depends on**: Phase 19
**Requirements**: ERR-01, ERR-02, ERR-03, ERR-04, ERR-05, API-01
**Success Criteria** (what must be TRUE):
  1. When GARCH model fitting fails or produces NaN, the volatility module returns historical rolling volatility as fallback and logs a warning -- NaN never reaches the position sizing pipeline
  2. EventBus.create_group catches only redis.ResponseError (not bare Exception) -- unexpected Redis errors are logged and re-raised
  3. TinkoffFetcher logs failures with structured fields: ticker, timeframe, error_type, and request_id -- operators can filter and diagnose data issues
  4. TradingLoop increments a consecutive error counter per cycle type; after N consecutive failures (configurable, default 3), a Telegram warning alert is sent
  5. BondCycleProcessor logs escalated error after threshold consecutive gRPC failures per cycle -- systematic broker issues are visible in logs
  6. POST /kill endpoint requires X-API-Key header matching a configured secret -- unauthenticated requests receive 401
**Plans**: TBD

### Phase 22: Dependency Layer Cleanup
**Goal**: core/ contains only Layer 0 types and schemas; orchestration logic lives in a dedicated module; dead infrastructure is removed
**Depends on**: Phase 20, Phase 21
**Requirements**: LAYER-01, LAYER-02, LAYER-03, LAYER-04, DEAD-01, DEAD-02
**Success Criteria** (what must be TRUE):
  1. trading_loop.py and bond_cycle.py are importable from a new orchestration/ module (not core/) -- core/ has no files that import from Layer 3+ modules
  2. telegram_bot.py and alerts.py reside under an appropriate Layer 6 module -- core/ does not contain API/dashboard layer code
  3. MetricsCollector is injected into TradingLoop via constructor parameter -- TradingLoop has no direct import of monitoring/ or api/ modules
  4. backtest/ and monitoring/ modules have layer assignments documented in their respective CLAUDE.md files and confirmed by import analysis
  5. Event bus stream constants (STREAM_MARKET_DATA, STREAM_SIGNALS, STREAM_EXECUTION) are either wired to actual consumers or removed -- no dead pub/sub infrastructure
  6. Stub API endpoints (/signals, /trades, /news, /ml/status) either serve real data or return 501 Not Implemented with a clear message
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 19 -> 20 -> 21 -> 22
Note: Phase 21 depends on Phase 19 (not 20), so 20 and 21 could run in parallel if needed.

| Phase | Milestone | Plans | Status | Completed |
|-------|-----------|-------|--------|-----------|
| 1-7 | v1.0 | 22/22 | Complete | 2026-03-19 |
| 8-14 | v2.0 | 16/16 | Complete | 2026-03-21 |
| 15-18 | v3.0 | 10/10 | Complete | 2026-03-22 |
| 19. Concurrency Safety | v4.0 | 1/2 | In Progress|  |
| 20. Async and Resources | v4.0 | 0/TBD | Not started | - |
| 21. Error Handling | v4.0 | 0/TBD | Not started | - |
| 22. Layer Cleanup | v4.0 | 0/TBD | Not started | - |

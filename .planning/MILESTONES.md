# Milestones

## v5.0 Data Flow Correctness (Shipped: 2026-03-24)

**Phases completed:** 4 phases, 7 plans, 10 tasks

**Key accomplishments:**

- Fixed SELL qty (held position), sector exposure (per-position prices), and CAUTION threshold (segment preset) in TradingLoop
- Trailing stop state machine with 5-step ratcheting logic and per-cycle re-entry guard wired into TradingLoop, matching SimulatedBroker behavior
- PositionSizingPipeline wired in live _build_order with all 14 pre-trade check parameters passed
- DataNormalizer candle validation, 48h staleness detection, and IMOEX volume column fix wired into live trading loop
- Persistent gRPC channel for all TinkoffFetcher bond methods and Brent crude caching via GenericFileCache
- News cycle skip guard when event_driven is disabled plus 4-hour half-life exponential sentiment decay
- Fixed T-Bank ticker mismatch (T -> TCSG) in entity extractor and added URL-based message deduplication to Telegram reader with 5000-entry LRU eviction

---

## v4.0 Architecture Hardening (Shipped: 2026-03-22)

**Phases completed:** 4 phases, 10 plans, 21 tasks

**Key accomplishments:**

- asyncio.Lock for async broker paths, threading.Lock double-check for loop init, async-with session scoping in macro_cache
- Atomic stop-loss under single lock hold preventing double-sell TOCTOU race, plus direct feed timestamp wiring and verified /gonogo import
- Fixed three async bugs: non-blocking gRPC reconnect via _stop_event.wait(), coroutine-aware aexecute() with iscoroutine guard, and thread-safe persistence via background event loop replacing asyncio.run()
- Non-blocking portfolio endpoint via run_in_executor, structured close() failure logging, and configurable 60s gRPC timeout for TinkoffFetcher
- Idempotent TelegramAlerter.close() wired into FastAPI lifespan shutdown for both alerter instances (trading loop + bot handler), preventing httpx.AsyncClient resource leaks
- GARCH NaN fallback with rolling vol + structlog warnings, EventBus exception narrowing to redis.ResponseError, and POST /kill authenticated via X-API-Key
- Structured error_type logging in TinkoffFetcher, consecutive failure counters with Telegram alerting in TradingLoop, and per-layer error escalation in BondCycleProcessor
- Moved 4 misplaced modules from core/ to their correct dependency layers: orchestration/ (L5) and api/ (L6), with sys.modules shims for zero-breakage backward compatibility
- MetricsCollector injected into TradingLoop via constructor, eliminating 6 deferred L6 imports; backtest/ and monitoring/ assigned definitive layers
- Removed 3 dead EventBus stream constants and 2 unused event models; converted 7 stub API endpoints from empty-200 to explicit 501 Not Implemented

---

## v3.0 Production Readiness (Shipped: 2026-03-22)

**Phases completed:** 4 phases, 10 plans, 0 tasks

**Key accomplishments:**

- (none recorded)

---

## v2.0 MOEX Profitability (Shipped: 2026-03-21)

**Phases completed:** 7 phases, 16 plans, 0 tasks

**Key accomplishments:**

- (none recorded)

---

## v1.0 MOEX MVP (Shipped: 2026-03-19)

**Phases completed:** 7 phases, 22 plans
**Timeline:** 22 days (2026-02-22 → 2026-03-15)
**Codebase:** 35,199 LOC Python, 3,651 tests

**Key accomplishments:**

- MOEX equity trading with RUB-native Half-Kelly sizing, MOEX holiday calendar, 5 tuned strategies across 4 segments
- Full bond trading pipeline with QuantLib (YTM, duration, convexity), BondCycleProcessor, OFZ carry strategy (Sharpe +1.14)
- Autonomous TradingLoop with APScheduler equity + bond + news cycles, crash recovery, graceful error handling
- Telegram monitoring: priority message queue, trade/coupon/CBR alerts, /status + /stop commands, daily P&L
- Sandbox validation infrastructure: Docker stack, validation report generator, autonomous operation criteria
- Russian news pipeline: RSS fetcher (RBC, Interfax, TASS), Telegram reader, LLM entity extraction, event_driven at 15% weight

---

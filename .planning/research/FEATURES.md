# Feature Landscape

**Domain:** Production trading system stability and observability fixes
**Researched:** 2026-03-30
**Confidence:** HIGH -- based on codebase audit + sandbox logs + industry patterns

---

## Context: What This Milestone Fixes

10 issues discovered during week-long sandbox run (March 20-30, 2026). This is NOT greenfield
feature work -- it is hardening an existing system that already trades. Every fix has a specific
failure observed in production-like conditions.

Key sandbox metrics:
- 127 missed scheduler jobs from gRPC BlockingIOError flooding asyncio
- 62 portfolio_fetch_failed from T-Bank API error 70001
- 0 rows in orders/signals/news_articles/sentiment_scores tables
- 0 log entries in Loki (Promtail not shipping)
- FX rate = 0.0 (gRPC failure, no CBR XML fallback wired in trading loop)
- Strategy cycles fire 24/7 but MOEX is open 07:00-15:45 UTC only
- 35 LLM fallback activations/day from duplicate articles
- Only 1 trade in 62 hours (system mostly functional but blind)

---

## Table Stakes

Features that MUST work for the system to be production-viable. Missing = the system is
operationally blind, wastes resources, or loses data.

| # | Feature | Why Expected | Complexity | Existing Code |
|---|---------|--------------|------------|---------------|
| TS-1 | gRPC event loop isolation | BlockingIOError starves asyncio, causing 60-min cycle drift. Without this, scheduler misses 127 jobs/week. No trading system can tolerate 60-min signal delays. | High | `TinkoffBroker._run_async()` uses `run_coroutine_threadsafe` on dedicated thread, but gRPC C-core still emits blocking I/O that leaks into the APScheduler thread. |
| TS-2 | T-Bank API error 70001 resilience | Portfolio fetch fails for hours during market hours. Without portfolio state, no position sizing, no risk checks, no trades. Multi-hour blind windows are unacceptable. | Medium | `TinkoffBroker._get_services_async()` creates client but has no reconnect on channel failure. RetryPolicy exists but doesn't handle channel-level gRPC errors. |
| TS-3 | DB persistence for orders/signals/news/sentiment | 0 rows after 5 days means complete data loss. Post-mortem analysis impossible, regulatory audit trail missing, no way to debug strategy behavior. | Medium | ORM models exist (`OrderModel`, `SignalModel`, `NewsArticleModel`, `SentimentScoreModel` in `core/models.py`). `DailyEquitySnapshot` persistence works. The persist path is broken (never called or fire-and-forget fails silently). |
| TS-4 | Loki log pipeline operational | 0 log entries stored means no observability. Can't search logs, can't correlate events, can't debug failures. Grafana dashboards for logs are useless. | Low | Docker Compose has Loki + Promtail services. Promtail config targets `finalayze-sandbox-app` container. Issue is likely Docker log driver or volume mount permissions. |
| TS-5 | FX rate fallback | FX rate = 0.0 breaks all RUB/USD conversion in position sizing. Every MOEX position size calculation is wrong. CBR XML API (`fx_service.py`) exists but isn't wired as fallback when gRPC FX fetch fails. | Low | `FXRateService` with CBR XML parsing fully implemented. `CurrencyConverter.set_rate()` works. Gap: trading loop's FX path uses gRPC only, doesn't call `FXRateService` on failure. |
| TS-6 | Market-hours gate at cycle level | Strategy cycles fire 24/7 but MOEX open 07:00-15:45 UTC. Off-hours cycles waste compute, generate misleading "0 instruments" logs, and pollute metrics. | Low | `_is_market_open()` exists and is called inside `_process_market_cycle()`. But the cycle still runs (portfolio fetch, circuit breaker check) before reaching the gate. Need to gate earlier -- at `_strategy_cycle` entry or scheduler level. |

## Differentiators

Features that improve operational quality. Not strictly required for the system to trade,
but significantly reduce operational burden and improve reliability.

| # | Feature | Value Proposition | Complexity | Existing Code |
|---|---------|-------------------|------------|---------------|
| D-1 | LLM article deduplication | 35 fallback activations/day from duplicate articles wastes LLM quota and causes Groq fallback (slower, less accurate). Dedup reduces LLM calls ~50%. | Low | RSS fetcher has `_seen_urls` set. Telegram reader has `_seen_messages`. But dedup doesn't cross sources (same article from RSS + Telegram). Also, RSS `_seen_urls` resets on restart. |
| D-2 | Stale ticker cleanup | 5 stale tickers (FIVE->X5, FIXP/POLY removed, YNDX->YDEX, HHRU->HH) cause failed lookups, wasted API calls, and confusing logs. | Low | `config/segments.py` has `HHRU` in `ru_tech`. `config/universes/moex_blue_chips.json` has `YNDX`. Need to update both config files and entity extractor valid tickers list. |
| D-3 | Telegram alerter startup resilience | Alerter failure at startup crashes the entire trading loop. Alerter should be best-effort, not critical path. | Low | `_alerter_ref` set via attribute mutation. Likely throws on `__init__` if Telegram token invalid or network unreachable. |

## Anti-Features

Features to explicitly NOT build in this milestone. Tempting but would over-engineer the fixes.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Full gRPC async migration | Rewriting TinkoffBroker to use `grpc.aio` natively would be ideal but is a multi-week effort touching every broker call site. Risk of regression too high for a stability milestone. | Isolate blocking gRPC calls in a dedicated `ThreadPoolExecutor` with bounded queue. Keep the `run_coroutine_threadsafe` pattern but ensure gRPC C-core I/O never touches the asyncio event loop thread. |
| Real-time WebSocket data feeds | Would eliminate polling entirely but requires fundamental architecture change. | Keep interval-based polling with market-hours gate to reduce wasted cycles. |
| Comprehensive trade reconciliation system | Full reconciliation with broker statements is important but out of scope for "wire the persist path." | Persist orders/signals as they happen. Reconciliation is a future milestone. |
| Distributed tracing (Jaeger/Tempo) | Production observability gold standard but overkill for single-process system. | Fix Promtail/Loki first. Structured logging with correlation IDs is sufficient. |
| Automatic ticker migration system | Building a system that detects MOEX ticker changes and auto-updates config is over-engineering. | Manual update now. These changes happen 1-2 times/year on MOEX. |
| Multi-source FX rate aggregation | Averaging rates from CBR + MOEX ISS + Bloomberg is unnecessary precision. | CBR XML is the official rate. Use it as primary fallback. gRPC rate as optimization only. |
| APScheduler job persistence (SQLAlchemy jobstore) | Would survive restarts but adds DB dependency to scheduler. Import already exists but is guarded. | Keep in-memory jobstore. System restarts are infrequent and jobs re-register on startup. |
| Custom log shipping (direct to Loki API) | Bypasses Promtail entirely but couples application to observability stack. | Fix Promtail config. Standard Docker log shipping is battle-tested. |

## Feature Dependencies

```
TS-1 (gRPC isolation) ── no dependencies, standalone fix
TS-2 (70001 resilience) ── benefits from TS-1 (fewer BlockingIOErrors) but independent
TS-3 (DB persistence) ── no dependencies, models exist, need to wire persist calls
TS-4 (Loki pipeline) ── no dependencies, Docker/config fix only
TS-5 (FX fallback) ── no dependencies, FXRateService exists, wire into trading loop
TS-6 (market-hours gate) ── no dependencies, _is_market_open() exists, move gate earlier
D-1 (article dedup) ── no dependencies, extend existing _seen_urls pattern
D-2 (stale tickers) ── no dependencies, config-only change
D-3 (alerter resilience) ── no dependencies, wrap startup in try/except
```

All features are independent. No blocking dependency chains. Can be parallelized freely.

## Complexity Assessment

### High Complexity: TS-1 (gRPC Event Loop Isolation)

The root cause is architectural: `TinkoffBroker._run_async()` dispatches gRPC coroutines
to a background thread's event loop via `run_coroutine_threadsafe()`, but the gRPC C-core
library performs blocking I/O operations that can leak back to whichever thread triggers
the call. When APScheduler's thread pool executor calls `_strategy_cycle` -> `_run_async()`,
the `future.result()` call blocks the APScheduler thread, and if gRPC C-core emits
`BlockingIOError`, it floods the thread's stderr/logging, causing 127 missed jobs.

**Minimum viable fix:** Wrap all `_run_async()` calls in a dedicated
`concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="grpc")` that is
separate from APScheduler's executor. This confines gRPC blocking I/O to grpc-worker threads.
Add a timeout to `future.result(timeout=30)` so APScheduler threads never block indefinitely.

**Over-engineering boundary:** Do NOT rewrite to `grpc.aio` natively. Do NOT add circuit
breaker around individual gRPC calls (RetryPolicy already handles transient failures).

### Medium Complexity: TS-2 (T-Bank 70001 Resilience)

Error 70001 is a T-Bank sandbox-specific error indicating the portfolio service is temporarily
unavailable. The current code creates the gRPC channel once and never reconnects.

**Minimum viable fix:** On 70001 (or any `StatusCode.UNAVAILABLE`/`StatusCode.INTERNAL`),
close the current channel, set `_services = None`, and let the next call re-create it
via `_get_services_async()`. Add a last-known-portfolio cache so position sizing can
continue with stale (but recent) portfolio data during outage windows.

**Over-engineering boundary:** Do NOT implement health-check-based channel management.
Do NOT add a gRPC interceptor. Simple reconnect + cache is sufficient.

### Medium Complexity: TS-3 (DB Persistence)

The ORM models exist. The persist path for `DailyEquitySnapshot` works. The issue is that
order/signal/news/sentiment persist calls are either never invoked or fail silently.

**Minimum viable fix:** Add fire-and-forget `_persist_order()`, `_persist_signal()`,
`_persist_article()`, `_persist_sentiment()` calls at the appropriate points in
`_strategy_cycle_impl()` and `_news_cycle()`. Follow the existing `DailyEquitySnapshot`
pattern: get async session factory, create model, commit. Wrap in try/except to never
crash the trading loop.

**Over-engineering boundary:** Do NOT build a write-ahead queue or batch persistence.
Individual fire-and-forget inserts are fine for the ~10-20 events/day this system generates.

### Low Complexity: All Others

- **TS-4 (Loki):** Likely a Promtail config issue (Docker log path, container name filter,
  or missing `/var/lib/docker/containers` volume mount). Debug with `docker logs promtail`
  and `curl http://localhost:9080/ready`. Fix is config-only.
- **TS-5 (FX fallback):** Wire `FXRateService.update_usdrub()` as fallback in the FX
  update cycle. If gRPC FX fetch returns 0.0 or fails, call CBR XML. ~10 lines of code.
- **TS-6 (market-hours gate):** Move `_is_market_open()` check to the top of
  `_strategy_cycle()` or use APScheduler's cron trigger instead of interval trigger
  to only fire during MOEX hours (07:00-15:45 UTC on weekdays).
- **D-1 (article dedup):** Add content hash (title + first 100 chars) to a shared
  `_seen_articles: set[str]` across RSS and Telegram sources. Persist to Redis for
  restart survival. ~20 lines.
- **D-2 (stale tickers):** Update 2 config files. 5-minute task.
- **D-3 (alerter resilience):** Wrap alerter init in try/except, use a no-op alerter
  on failure. ~10 lines.

## MVP Recommendation

**Priority 1 (critical -- system is broken without these):**
1. TS-1: gRPC event loop isolation -- without this, scheduler drifts 60 min
2. TS-2: T-Bank 70001 resilience -- without this, multi-hour blind windows
3. TS-3: DB persistence -- without this, no audit trail, no post-mortem capability

**Priority 2 (important -- significant operational improvement):**
4. TS-4: Loki log pipeline -- enables log-based debugging
5. TS-5: FX rate fallback -- prevents 0.0 position sizes
6. TS-6: Market-hours gate -- eliminates wasted off-hours cycles

**Priority 3 (nice to have -- reduces noise):**
7. D-1: Article deduplication -- reduces LLM cost
8. D-2: Stale tickers -- eliminates failed lookups
9. D-3: Alerter resilience -- prevents startup crash from Telegram issues

**Defer:** Nothing. All 9 items are small enough to ship in one milestone.
The total estimated effort is 3-5 phases (2-4 days).

## Production Trading System Patterns (Reference)

### gRPC Resilience in Trading Systems

Production trading systems using gRPC follow these patterns:
- **Dedicated gRPC thread pool:** Never share executor threads between gRPC and application logic.
  gRPC C-core performs its own I/O multiplexing that can block threads unexpectedly.
- **Channel lifecycle management:** Recreate channels on `UNAVAILABLE`, `INTERNAL`, or
  `DEADLINE_EXCEEDED` status codes. Don't try to keep channels alive indefinitely.
- **Bounded timeouts:** Every gRPC call gets a deadline. No unbounded waits.
  30s for data fetches, 10s for order submissions, 5s for portfolio queries.
- **Last-known-good cache:** Cache portfolio state, FX rates, and instrument data.
  Use cached values when live fetch fails, with staleness warnings.

### Trade Data Persistence Requirements

For a personal/proprietary trading system (not broker-dealer), regulatory audit trail
(SEC Rule 613 CAT) does not apply. But operational requirements still demand:
- **Every order must be persisted:** order_id, symbol, side, quantity, price, timestamp, status.
  This is the minimum for post-mortem analysis and tax reporting.
- **Every signal must be persisted:** strategy_name, symbol, direction, confidence, timestamp.
  Required for strategy performance attribution.
- **Fire-and-forget is acceptable:** Trading loop must never block on DB write.
  If persist fails, log the error and continue trading. Data loss is preferable to missed trades.
- **Idempotent writes:** Use order_id as natural key. Duplicate persist calls should upsert,
  not create duplicate rows.

### Log Aggregation Best Practices

For Docker-based trading systems:
- **Structured JSON logging is mandatory.** Already done (structlog).
- **Promtail + Loki is the standard lightweight stack.** Correct choice for single-node.
- **Common Promtail failure modes:** (1) Docker log path not mounted, (2) container name
  filter mismatch, (3) json-file logging driver not set, (4) Loki URL incorrect.
- **Always mount `/var/lib/docker/containers`** in addition to Docker socket when using
  `docker_sd_configs`. The socket discovers containers; the volume reads their logs.

### Market-Hours Scheduling

Production systems use two approaches:
- **Cron-based scheduling:** APScheduler `CronTrigger` with `day_of_week='mon-fri'` and
  `hour`/`minute` constraints. Already used for bond_cycle. Preferred approach.
- **Guard at cycle entry:** Check `_is_market_open()` at the very top of `_strategy_cycle()`
  before any work (portfolio fetch, circuit breaker check). Simpler but still runs the
  scheduler tick.

Best practice: Use cron trigger for the strategy cycle (fires only during market hours)
AND keep the guard as defense-in-depth.

### FX Rate Redundancy

- **Primary:** Real-time rate from broker API (gRPC). Updates every cycle.
- **Fallback 1:** Central bank daily rate (CBR XML). Updates once/day but always available.
- **Fallback 2:** Static configured rate. Last resort, logged as warning.
- **Cache with TTL:** Cache the last good rate with a 4-hour TTL. If all sources fail
  within TTL, use cached rate. If TTL expired, halt trading (stale FX is dangerous
  for position sizing).

Already implemented: `CurrencyConverter` holds rate in memory, `FXRateService` fetches from
CBR XML. Gap is only the wiring in the trading loop's FX update path.

## Sources

- [gRPC AsyncIO API documentation](https://grpc.github.io/grpc/python/grpc_asyncio.html) -- official gRPC Python async patterns
- [gRPC multi-thread support issue #25364](https://github.com/grpc/grpc/issues/25364) -- thread safety discussion
- [gRPC performance best practices](https://grpc.io/docs/guides/performance/) -- channel management guidance
- [Python asyncio developing docs](https://docs.python.org/3/library/asyncio-dev.html) -- run_in_executor patterns
- [Promtail troubleshooting guide](https://oneuptime.com/blog/post/2026-01-21-promtail-troubleshooting/view) -- Docker log scraping fixes
- [Promtail Docker container log collection](https://community.grafana.com/t/promtail-does-not-collect-logs-from-other-containers/87000) -- volume mount requirements
- [Grafana Loki issue #5955](https://github.com/grafana/loki/issues/5955) -- "Unable to find any logs to tail" resolution

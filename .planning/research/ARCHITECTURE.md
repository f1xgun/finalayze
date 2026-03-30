# Architecture Patterns

**Domain:** Sandbox stability and observability fixes for MOEX autonomous trading system
**Researched:** 2026-03-30
**Confidence:** HIGH -- based on direct codebase inspection of all modified components

## Current Architecture Overview

```
                     APScheduler (BackgroundScheduler)
                     ├── news_cycle     (executor: "news", 1 thread)
                     ├── strategy_cycle (executor: "default", 4 threads)
                     ├── bond_cycle     (cron: 10:30 MSK)
                     ├── daily_reset    (cron)
                     ├── fx_update      (interval: 60 min)
                     └── macro_refresh  (cron: 10:00 MSK)
                              │
                     ┌────────┴─────────┐
                     │  TradingLoop     │  (Layer 5 -- orchestration/)
                     │  _async_loop ────│──── background asyncio thread (SHARED)
                     │  _run_async()    │     ├── gRPC calls (TinkoffBroker)
                     │                  │     ├── gRPC calls (TinkoffFetcher)
                     │                  │     ├── FX updates (httpx async)
                     │                  │     ├── Telegram send (httpx async)
                     │                  │     └── DB persist (SQLAlchemy async)
                     └──────────────────┘
                              │
              ┌───────────────┼──────────────────┐
              │               │                  │
     TinkoffBroker    TinkoffFetcher      FXRateService
     (own _loop)      (own _loop)         (uses TL._async_loop)
     execution/       data/fetchers/      markets/
```

**Critical problem:** Three separate `asyncio.new_event_loop()` instances exist:
1. `TradingLoop._async_loop` -- background thread for general async work
2. `TinkoffBroker._loop` -- own background thread for gRPC broker calls
3. `TinkoffFetcher._loop` -- own background thread for gRPC data calls

The gRPC C-core library registers its `PollerCompletionQueue` callbacks on whichever asyncio event loop creates the gRPC channel. When multiple gRPC channels coexist on a loop with non-gRPC work (FX updates, DB writes, Telegram), the poller saturates the loop's self-pipe buffer, producing `BlockingIOError` and starving APScheduler cycles (drift up to 60 min).

### Component Boundaries

| Component | Responsibility | Communicates With | Layer |
|-----------|---------------|-------------------|-------|
| `TradingLoop` | Orchestrates all scheduled cycles | All below | L5 (orchestration/) |
| `TinkoffBroker` | Order submission, portfolio queries via gRPC | T-Bank gRPC API | L5 (execution/) |
| `TinkoffFetcher` | Candle/instrument data via gRPC | T-Bank gRPC API | L2 (data/fetchers/) |
| `FXRateService` | USD/RUB rate from CBR XML | CBR HTTP API | L2 (markets/) |
| `TelegramAlerter` | Alert dispatch via Telegram Bot API | Telegram HTTP API | L6 (api/) |
| `SandboxMonitorService` | Cycle metrics persistence | PostgreSQL (async) | L6 (monitoring/) |
| `ValidationLogger` | Structured cycle log entries | stdout (structlog) | L0 (core/) |
| `OrderModel` / `SignalModel` | DB persistence for orders/signals | PostgreSQL | L0 (core/models.py) |
| `NewsArticleModel` | DB persistence for news | PostgreSQL | L0 (core/models.py) |
| Promtail | Log shipper (Docker container) | Loki via HTTP push | Infrastructure |
| Loki | Log aggregation | Grafana queries | Infrastructure |

---

## Recommended Architecture (Post-Fix)

```
                     APScheduler (BackgroundScheduler)
                     ├── news_cycle
                     ├── strategy_cycle
                     ├── bond_cycle
                     ├── daily_reset
                     └── fx_update
                              │
                     ┌────────┴─────────┐
                     │  TradingLoop     │
                     │  _async_loop ────│──── background thread (general async)
                     │                  │     ├── FX updates (httpx)
                     │                  │     ├── Telegram (httpx)
                     │                  │     └── DB persist (SQLAlchemy async)
                     │                  │
                     │  _grpc_loop ─────│──── DEDICATED gRPC thread (NEW)
                     │                  │     ├── TinkoffBroker calls
                     │                  │     └── TinkoffFetcher calls
                     └──────────────────┘
                              │
              ┌───────────────┼──────────────────┐
              │               │                  │
     TinkoffBroker    TinkoffFetcher      FXRateService
     (uses TL._grpc_loop)  (uses TL._grpc_loop) (uses TL._async_loop)
```

**Key change:** All gRPC work is isolated to a single dedicated event loop thread. TinkoffBroker and TinkoffFetcher no longer manage their own loops -- they accept the shared gRPC loop from TradingLoop. Non-gRPC async work (HTTP, DB) stays on `_async_loop`, completely free from gRPC poller contention.

---

## Integration Analysis: 10 Fixes

### Fix 1: gRPC Event Loop Isolation

**Problem:** gRPC C-core's `PollerCompletionQueue` registers on the asyncio loop that creates the channel. When TradingLoop's `_async_loop` hosts both gRPC and non-gRPC coroutines, the poller saturates the pipe buffer, causing `BlockingIOError` and starving APScheduler cycles (drift up to 60 min).

**Files to modify:**
| File | Change |
|------|--------|
| `src/finalayze/orchestration/trading_loop.py` | Add `_grpc_loop` + `_grpc_thread`. Add `_run_grpc(coro)` method. Route all broker/fetcher calls through it. |
| `src/finalayze/execution/tinkoff_broker.py` | Remove self-managed `_loop` / `_loop_thread`. Accept external `grpc_loop: asyncio.AbstractEventLoop` via constructor or setter. |
| `src/finalayze/data/fetchers/tinkoff_data.py` | Same: accept external `grpc_loop`, remove self-managed loop. |

**Data flow change:**
```
Before: APScheduler thread -> TradingLoop._run_async() -> TL._async_loop (gRPC + everything)
After:  APScheduler thread -> TradingLoop._run_grpc()  -> TL._grpc_loop  (gRPC only)
        APScheduler thread -> TradingLoop._run_async() -> TL._async_loop (FX, DB, Telegram)
```

**Call sites that must switch from `_run_async()` to `_run_grpc()`:**
- `_get_cached_portfolio()` -- calls `broker.get_portfolio()` which uses gRPC
- `_process_instrument()` -- calls fetcher for candles (gRPC)
- `_submit_order()` -- calls `broker_router.submit()` which routes to TinkoffBroker (gRPC)
- `_bond_cycle()` -- bond processor calls TinkoffBroker (gRPC)
- `_reconcile_inflight_orders()` -- calls `broker.get_open_orders()` (gRPC)
- `_attempt_grpc_reconnect()` -- calls `broker.reconnect_client()` (gRPC)

**Call sites that stay on `_run_async()`:**
- `_fx_update_cycle()` -- CBR HTTP via httpx
- `_persist_snapshots_async()` -- SQLAlchemy async DB write
- `_news_cycle()` / `_analyze_impact_batch()` -- LLM HTTP calls
- Telegram alert sends

**Lifecycle changes:**
- `start()` must initialize both loops before starting scheduler
- `stop()` / `close()` must tear down both loops
- TinkoffBroker.`close()` no longer stops its own loop (it does not own one)

**Risk to live trading:** MEDIUM. Most architectural change. But the current state is already broken (60-min drift), so the risk of NOT fixing is higher. The `_run_grpc()` method is structurally identical to `_run_async()` -- just targets a different loop.

---

### Fix 2: T-Bank API Error 70001 Resilience

**Problem:** T-Bank Sandbox API returns error code 70001 intermittently. `get_portfolio()` fails, causing strategy cycle to skip the market entirely for hours.

**Files to modify:**
| File | Change |
|------|--------|
| `src/finalayze/execution/tinkoff_broker.py` | Add `_last_known_portfolio: PortfolioState` cache. On successful `get_portfolio()`, save result. On 70001 failure, return cached copy. Track staleness age. |

**Data flow change:**
```
Before: get_portfolio() -> gRPC -> 70001 -> raise BrokerError -> market skipped
After:  get_portfolio() -> gRPC -> 70001 -> return _last_known_portfolio (+ warning log)
```

**Integration points:**
- `TinkoffBroker.get_portfolio()` is the sync method called from `_get_cached_portfolio()` in TradingLoop
- Must detect "70001" in exception message or gRPC status detail
- Log the staleness age so operators know how old the fallback is
- After N consecutive 70001 errors (e.g., 5), trigger `reconnect_client()` automatically

**Dependency on Fix 1:** The broker refactoring (removing self-managed loop) should be done first. Fix 2 then adds fallback logic to the already-refactored broker.

**Risk to live trading:** LOW. Stale portfolio data is acceptable for position sizing (positions change slowly). The alternative (market skip for hours) is worse.

---

### Fix 3: DB Persistence for Orders, Signals, News, Sentiment

**Problem:** `OrderModel`, `SignalModel`, `NewsArticleModel`, `SentimentScoreModel` tables exist in `core/models.py` with Alembic migrations, but TradingLoop never writes to them. All trade data is ephemeral (log-only).

**Files to modify:**
| File | Change |
|------|--------|
| `src/finalayze/orchestration/persistence.py` | **NEW FILE.** Extract persistence helpers. `persist_signal()`, `persist_order()`, `persist_news_article()`, `persist_sentiment()`. |
| `src/finalayze/orchestration/trading_loop.py` | Call persistence helpers from `_process_instrument()` (signals), `_submit_order()` (orders), `_news_cycle()` (articles/sentiment). |

**Data flow -- signal persistence:**
```
_process_instrument():
  combiner.generate_signal() -> Signal schema
  -> NEW: persist_signal(signal, segment_id, market_id) -> SignalModel -> DB (via _run_async)
  -> returns signal_id: UUID
  -> _submit_order(order, market_id, signal_id=signal_id)
```

**Data flow -- order persistence:**
```
_submit_order():
  broker_router.submit(order) -> OrderResult
  -> NEW: persist_order(order, result, signal_id, market_id) -> OrderModel -> DB (via _run_async)
```

**Data flow -- news persistence:**
```
_news_cycle():
  _analyze_impact_batch(articles)
    -> for each article: analyzer.analyze(article) -> NewsImpactResult
    -> NEW: persist_news_article(article, result) -> NewsArticleModel -> DB
    -> NEW: persist_sentiment(symbol, score, market_id) -> SentimentScoreModel -> DB
```

**Integration points:**
- DB writes go through `_run_async()` (not `_run_grpc()`): SQLAlchemy async, not gRPC
- `get_async_session_factory()` from `core/db.py` provides the session factory
- `_process_instrument()` signature must pass `signal_id` to `_submit_order()` for FK linking
- All persistence is fire-and-forget: wrap in try/except, log warning on failure, never block trading
- Separate file (`persistence.py`) because `trading_loop.py` is already 2400+ lines

**Risk to live trading:** LOW. All persistence is additive. Failure to persist logs a warning but does not interrupt trading.

---

### Fix 4: Loki Log Pipeline (Promtail Configuration)

**Problem:** Promtail ships 0 log entries to Loki despite correct config structure. Root cause: Promtail uses Docker service discovery (`docker_sd_configs`) to find containers, but it only mounts `/var/run/docker.sock` -- it can discover the container but cannot read its log files because the Docker log directory (`/var/lib/docker/containers/`) is not volume-mounted.

**Files to modify:**
| File | Change |
|------|--------|
| `docker/docker-compose.sandbox.yml` | Add volume mount for Docker container logs into Promtail service. |
| `monitoring/promtail/promtail-config.yml` | Possibly no change needed if Docker SD + log file mount works. May need to verify `__path__` label resolution. |

**Fix:**
```yaml
# docker-compose.sandbox.yml - promtail service volumes
volumes:
  - ../monitoring/promtail/promtail-config.yml:/etc/promtail/config.yml:ro
  - /var/run/docker.sock:/var/run/docker.sock:ro
  - /var/lib/docker/containers:/var/lib/docker/containers:ro  # ADD THIS
```

**Data flow (corrected):**
```
app container -> stdout -> Docker json-file driver -> /var/lib/docker/containers/<id>/*-json.log
  -> Promtail (now has read access via volume mount)
  -> Promtail parses JSON (event, level, timestamp per pipeline_stages)
  -> HTTP push to http://loki:3100/loki/api/v1/push
  -> Loki stores in /loki/chunks (tsdb store, filesystem backend)
  -> Grafana queries Loki via provisioned datasource
```

**Integration points:**
- The app's structlog JSON output must match Promtail's `pipeline_stages.json.expressions` (keys: `event`, `level`, `timestamp`)
- Loki config (`monitoring/loki/loki-config.yml`) is already correct: tsdb store, filesystem backend, schema v13
- Grafana Loki datasource provisioning exists at `monitoring/grafana/provisioning/datasources/loki.yml`
- Drop rules for noisy health/metrics access logs already configured

**Risk to live trading:** NONE. Infrastructure-only change. No application code modified.

---

### Fix 5: FX Rate Fallback via CBR XML API

**Problem:** `FXRateService.update_usdrub()` uses CBR daily XML feed (HTTP). On failure, returns None and the `CurrencyConverter` retains its last-set rate. If the service never succeeds on startup, the rate stays at whatever initial default was set (potentially stale).

**Files to modify:**
| File | Change |
|------|--------|
| `src/finalayze/markets/fx_service.py` | Add fallback chain: (1) CBR XML_daily.asp (existing), (2) CBR XML_dynamic.asp (last 3 days, via `CBRClient`), (3) keep previous rate + log warning. |
| `src/finalayze/markets/currency.py` | Add `_rate_updated_at: dict[str, datetime]` to track freshness. Add `rate_age()` method. |

**Data flow:**
```
Before: FXRateService -> CBR XML_daily -> fail -> return None (no update)
After:  FXRateService -> CBR XML_daily -> fail
                       -> CBR XML_dynamic (last 3 days, from data/fetchers/cbr.py) -> fail
                       -> log warning, keep previous rate
```

**Integration points:**
- `CBRClient` from `data/fetchers/cbr.py` already has `fetch_fx_rates()` that queries `XML_dynamic.asp` -- same layer (L2), no import violation
- TradingLoop's `_fx_update_cycle()` calls `self._run_async(self._fx_service.update_usdrub())` -- no change needed
- Staleness tracking in `CurrencyConverter` is informational (log + metrics)

**Risk to live trading:** LOW. Additive fallback logic. If all HTTP sources fail, existing rate is preserved.

---

### Fix 6: Market-Hours Gate in Strategy Cycle

**Problem:** Strategy cycle runs every N minutes regardless of market hours, wasting gRPC calls and producing stale-candle warnings off-hours.

**Current state:** `_is_market_open()` exists (line 1312) and is already called per-market at line 1260 (`_strategy_cycle_impl()`), and for bonds at line 654. Individual markets are skipped if closed.

**What is actually missing:** The cycle still runs the full startup sequence (increment counter, clear caches, create CycleLogEntry, compute drawdown) even when ALL markets are closed. The fix is a top-level early exit.

**Files to modify:**
| File | Change |
|------|--------|
| `src/finalayze/orchestration/trading_loop.py` | Add early return at top of `_strategy_cycle()` if no market in `_circuit_breakers.keys()` is currently open. Log at DEBUG level. |

**Code sketch:**
```python
def _strategy_cycle(self) -> None:
    now = self._now()
    if not any(self._is_market_open(m, now) for m in self._circuit_breakers):
        _log.debug("strategy_cycle_all_markets_closed")
        return
    # ... existing code
```

**Risk to live trading:** LOW. Conservative check -- if ANY market is open, full cycle runs.

---

### Fix 7: Stale Ticker Cleanup

**Problem:** Segment configs reference tickers renamed on MOEX.

**Current state in `config/segments.py`:**
- `FIVE` already updated to `X5` (line 169)
- `YNDX` already updated to `YDEX` (line 103)
- `HHRU` present in `ru_tech` (line 103) -- needs updating to `HH`
- `FIXP` and `POLY` not present in any segment (already removed)

**Files to modify:**
| File | Change |
|------|--------|
| `config/segments.py` | Change `HHRU` to `HH` in `ru_tech` segment (line 103). |
| Instrument registry / FIGI mapping | Verify FIGI for `HH` ticker. |

**Risk to live trading:** LOW. Config-only change. Must verify FIGI mapping is correct before deploying.

---

### Fix 8: LLM Article Deduplication

**Problem:** RSS and Telegram fetchers have URL-based dedup (`_seen_urls` OrderedDict), but the same story from different sources (RBC vs Interfax vs Telegram) with different URLs gets analyzed twice, wasting LLM quota.

**Files to modify:**
| File | Change |
|------|--------|
| `src/finalayze/analysis/dedup.py` | **NEW FILE.** `deduplicate_articles(articles: list[NewsArticle]) -> list[NewsArticle]` using normalized title hash. |
| `src/finalayze/orchestration/trading_loop.py` | Call `deduplicate_articles()` after fetching but before `_analyze_impact_batch()` in `_news_cycle()`. |

**Data flow:**
```
Before: articles = rss_articles + tg_articles -> _analyze_impact_batch(articles)
After:  articles = rss_articles + tg_articles
        -> deduplicate_articles(articles)  # title-hash dedup
        -> _analyze_impact_batch(unique_articles)
```

**Dedup approach:** Normalize title (lowercase, strip punctuation, collapse whitespace), take first 100 chars, hash. Keep first occurrence per hash. Simple, deterministic, no ML needed.

**Risk to live trading:** LOW. Reduces LLM calls, does not affect trading logic.

---

### Fix 9: Telegram Alerter Startup Resilience

**Problem:** If Telegram bot token is invalid or Telegram API is unreachable at startup, the TradingLoop may fail to start because `on_startup()` sends a Telegram alert synchronously.

**Files to modify:**
| File | Change |
|------|--------|
| `src/finalayze/api/alerts.py` | Ensure `send_alert()` catches all exceptions internally when used in fire-and-forget mode. |
| `src/finalayze/orchestration/trading_loop.py` | Wrap `_alerter.on_startup()` call in try/except. Trading must not fail because Telegram is unreachable. |

**Risk to live trading:** LOW. Telegram is monitoring-only, not trading-critical.

---

### Fix 10: Combined FX Staleness Tracking (extension of Fix 5)

**Files to modify:** Same as Fix 5 (`currency.py`, `fx_service.py`). Adds `_rate_updated_at` tracking and a `rate_age()` method. TradingLoop can optionally log/alert when FX rate is older than 24 hours.

---

## Build Order (Dependency-Aware)

### Dependency Graph

```
Fix 7 (stale tickers) ---- independent, trivial
Fix 6 (market-hours)  ---- independent, trivial
Fix 9 (telegram)      ---- independent, trivial

Fix 1 (gRPC isolation)
  |
  +---> Fix 2 (70001 resilience) -- depends on broker refactoring from Fix 1

Fix 4 (Loki pipeline) ---- independent, infrastructure-only

Fix 3 (DB persistence) ---- independent, but more valuable after Fix 1 (stable cycles)
Fix 5 (FX fallback)    ---- independent
Fix 8 (LLM dedup)      ---- independent
```

### Recommended Build Order

| Phase | Fix | Rationale | Risk | Effort |
|-------|-----|-----------|------|--------|
| 1 | **Fix 7: Stale tickers** | Config-only, zero risk, unblocks correct data flow | NONE | XS |
| 1 | **Fix 6: Market-hours gate** | Simple early-exit, reduces off-hours noise immediately | LOW | S |
| 1 | **Fix 9: Telegram resilience** | try/except wrapper, prevents startup crash | LOW | S |
| 2 | **Fix 1: gRPC isolation** | Most critical fix. Strategy cycle 60-min drift is the top blocker. | MEDIUM | L |
| 2 | **Fix 4: Loki pipeline** | Infrastructure-only, no code risk. Enables log visibility for remaining fixes. | NONE | S |
| 3 | **Fix 2: 70001 resilience** | Builds on Fix 1 broker refactoring. Adds fallback portfolio cache. | LOW | M |
| 3 | **Fix 5+10: FX fallback + staleness** | Independent but lower priority. Rates rarely fail for extended periods. | LOW | S |
| 4 | **Fix 3: DB persistence** | Largest scope. Needs stable gRPC (Fix 1) to produce useful data. | LOW | L |
| 4 | **Fix 8: LLM dedup** | Reduces wasted LLM calls. Lower urgency than core stability. | LOW | S |

### Phase Ordering Rationale

1. **Phase 1 (Quick wins):** Zero-risk config fixes and simple guards. Deploy immediately to reduce noise and fix trivial bugs. Can ship same day.
2. **Phase 2 (Core stability):** gRPC isolation is THE blocking issue -- strategy cycles drifting 60 min makes the system non-functional. Loki gives visibility into remaining problems once cycles are stable.
3. **Phase 3 (Resilience):** Error 70001 and FX fallback are defense-in-depth after core stability is fixed. Fix 2 depends on Fix 1's broker refactoring.
4. **Phase 4 (Data capture):** DB persistence and dedup are valuable but not urgent. They require stable infrastructure to produce meaningful data.

---

## Patterns to Follow

### Pattern 1: Dedicated Event Loop per IO Domain
**What:** Separate asyncio event loops for gRPC vs HTTP/DB work.
**When:** gRPC C-core's PollerCompletionQueue monopolizes the event loop's self-pipe.
**Example:**
```python
class TradingLoop:
    def _init_loops(self) -> None:
        # General async (HTTP, DB, Telegram)
        self._async_loop = asyncio.new_event_loop()
        threading.Thread(target=self._async_loop.run_forever, daemon=True).start()

        # gRPC-only (TinkoffBroker, TinkoffFetcher)
        self._grpc_loop = asyncio.new_event_loop()
        threading.Thread(target=self._grpc_loop.run_forever, daemon=True).start()

    def _run_grpc(self, coro: Any, *, timeout: int = 30) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._grpc_loop)
        return future.result(timeout=timeout)
```

### Pattern 2: Fire-and-Forget Persistence
**What:** DB writes are non-blocking, non-fatal. Trading continues even if persistence fails.
**When:** Any DB write from the trading loop hot path.
**Example:**
```python
def _persist_signal(self, signal: Signal, seg_id: str, market_id: str) -> uuid.UUID | None:
    try:
        signal_id = uuid.uuid4()
        self._run_async(self._persist_signal_async(signal_id, signal, seg_id, market_id))
        return signal_id
    except Exception:
        _log.warning("signal_persist_failed", symbol=signal.symbol, exc_info=True)
        return None
```

### Pattern 3: Last-Known Fallback Cache
**What:** Cache last successful API response; return stale data on transient failure.
**When:** Portfolio queries, FX rates -- any data that changes slowly.
**Example:**
```python
class TinkoffBroker:
    _last_known_portfolio: PortfolioState | None = None
    _last_portfolio_at: datetime | None = None

    def get_portfolio(self) -> PortfolioState:
        try:
            portfolio = self._call(lambda: self._run_async(...))
            self._last_known_portfolio = portfolio
            self._last_portfolio_at = datetime.now(UTC)
            return portfolio
        except BrokerError as exc:
            if "70001" in str(exc) and self._last_known_portfolio is not None:
                age = datetime.now(UTC) - self._last_portfolio_at
                _log.warning("portfolio_fallback_stale", age_s=age.total_seconds())
                return self._last_known_portfolio
            raise
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Shared Event Loop for Mixed IO
**What:** Running gRPC and non-gRPC coroutines on the same asyncio event loop.
**Why bad:** gRPC C-core registers `PollerCompletionQueue` callbacks that saturate the loop's self-pipe, causing `BlockingIOError` and starving other coroutines.
**Instead:** Dedicate a separate event loop (and thread) for all gRPC work.

### Anti-Pattern 2: Blocking DB Writes in Trading Hot Path
**What:** Awaiting DB persistence before processing the next instrument.
**Why bad:** DB latency delays signal generation for subsequent instruments, compounding cycle drift.
**Instead:** Use fire-and-forget async persistence. Log failures but do not block trading.

### Anti-Pattern 3: Hard Failure on Monitoring Subsystem
**What:** Letting Telegram/alerter/Loki failures crash the trading loop.
**Why bad:** Monitoring is secondary to trading. A Telegram API timeout should not halt live trading.
**Instead:** Wrap all monitoring/alerting calls in try/except at the integration boundary.

### Anti-Pattern 4: Each gRPC Client Managing Its Own Event Loop
**What:** TinkoffBroker, TinkoffFetcher, and TradingLoop each create `asyncio.new_event_loop()`.
**Why bad:** Three daemon threads running three event loops. gRPC channels created on different loops cannot share connection state. Resource waste and complexity.
**Instead:** Single gRPC loop owned by TradingLoop, shared via constructor injection into broker and fetcher.

---

## File Modification Summary

| File | Fixes | Type | Est. Lines |
|------|-------|------|-----------|
| `src/finalayze/orchestration/trading_loop.py` | 1,3,6,8,9 | Modify | ~200 new |
| `src/finalayze/execution/tinkoff_broker.py` | 1,2 | Modify | ~80 new |
| `src/finalayze/data/fetchers/tinkoff_data.py` | 1 | Modify | ~30 changed |
| `src/finalayze/markets/fx_service.py` | 5,10 | Modify | ~40 new |
| `src/finalayze/markets/currency.py` | 10 | Modify | ~15 new |
| `src/finalayze/api/alerts.py` | 9 | Modify | ~10 new |
| `config/segments.py` | 7 | Modify | ~2 changed |
| `monitoring/promtail/promtail-config.yml` | 4 | Modify | ~1 added |
| `docker/docker-compose.sandbox.yml` | 4 | Modify | ~1 added |
| `src/finalayze/analysis/dedup.py` | 8 | **New** | ~40 |
| `src/finalayze/orchestration/persistence.py` | 3 | **New** | ~120 |

**Total:** 9 modified files, 2 new files, ~540 new/changed lines.

## Sources

- Codebase: `src/finalayze/orchestration/trading_loop.py` (2400+ lines, all cycle methods inspected)
- Codebase: `src/finalayze/execution/tinkoff_broker.py` (519 lines, event loop and reconnect logic)
- Codebase: `src/finalayze/data/fetchers/tinkoff_data.py` (persistent gRPC client pattern)
- Codebase: `src/finalayze/core/models.py` (ORM models for SignalModel, OrderModel, NewsArticleModel, SentimentScoreModel)
- Codebase: `docker/docker-compose.sandbox.yml` + `monitoring/promtail/promtail-config.yml` + `monitoring/loki/loki-config.yml`
- Codebase: `src/finalayze/markets/fx_service.py` (CBR XML integration)
- Codebase: `src/finalayze/data/fetchers/cbr.py` (CBR XML_dynamic.asp fallback endpoint)
- Codebase: `config/segments.py` (ticker universe -- HHRU needs HH rename)
- gRPC Python docs: C-core PollerCompletionQueue + asyncio interaction (HIGH confidence)
- Docker/Promtail docs: Docker SD requires container log directory mount (HIGH confidence)

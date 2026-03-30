# Technology Stack: v6.0 Sandbox Stability & Observability

**Project:** Finalayze MOEX MVP
**Researched:** 2026-03-30
**Confidence:** MEDIUM-HIGH (verified with official docs + grpc GitHub issues)

## Existing Stack (DO NOT CHANGE)

Already validated and operational. Listed for reference only.

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.12 | Runtime |
| grpcio | 1.78.1 | gRPC transport (via t-tech-investments) |
| t-tech-investments | 0.3.3 | T-Bank (Tinkoff Invest) SDK |
| SQLAlchemy | 2.0.46 | Async ORM + session management |
| asyncpg | (installed) | PostgreSQL async driver |
| httpx | (installed) | HTTP client (sync + async) |
| lxml | (installed) | XML parsing (CBR) |
| structlog | (installed) | Structured logging |
| APScheduler | (installed) | Background job scheduling |
| Grafana | 11.4.0 | Dashboards |
| Loki | 3.4.2 | Log aggregation |
| Promtail | 3.4.2 | Log shipping |
| Prometheus | 2.51.0 | Metrics |
| TimescaleDB | 2.17.2-pg16 | Time-series PostgreSQL |
| Redis | 7.4 | Cache + event bus |

## Changes Needed for v6.0 Fixes

### No new libraries required.

All six fix areas can be addressed with existing dependencies. This is intentional -- the system is mature enough that stability fixes should not introduce new dependencies.

---

## Fix 1: gRPC AsyncIO Isolation (BlockingIOError from PollerCompletionQueue)

### Problem

grpcio >= 1.75 introduced a `PollerCompletionQueue` that registers a pipe fd reader on the asyncio event loop. When gRPC channels are used from APScheduler threads (which call `run_coroutine_threadsafe` into a background event loop), the poller's `_handle_events` callback fires EAGAIN (`BlockingIOError: [Errno 11] Resource temporarily unavailable`). This floods the event loop's exception handler and can starve legitimate coroutines, causing strategy cycles to drift from 5 min to 60+ min.

### Root Cause Analysis

The current architecture creates gRPC channels (`AsyncClient`) on a dedicated background event loop via `asyncio.new_event_loop()` + `threading.Thread(target=loop.run_forever)`. This is correct for channel persistence (avoiding `asyncio.run()` which closes the loop), but problematic because:

1. **Both TinkoffFetcher and TinkoffBroker create independent background loops** -- two gRPC channels on two separate event loops, doubling the poller fd contention.
2. **APScheduler's BackgroundScheduler runs jobs in a thread pool**, and each job calls `run_coroutine_threadsafe()` which cross-thread-wakes the background loop. The PollerCompletionQueue fd reader can race with these wakeups.
3. **On macOS (dev)**: only `poll` strategy available (no `epoll1`). On Linux (Docker prod): `epoll1` is default and slightly better, but the fundamental issue persists.

### Solution: Single Shared gRPC Event Loop

**Pattern:** Extract the background event loop into a shared singleton. Both TinkoffFetcher and TinkoffBroker use the same loop and the same `AsyncClient`. This eliminates duplicate pollers.

```python
# Shared gRPC loop singleton (new module: src/finalayze/execution/grpc_loop.py)
import asyncio
import threading

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None

def get_grpc_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent background event loop for all gRPC operations."""
    global _loop, _thread
    if _loop is None or _loop.is_closed():
        with _lock:
            if _loop is None or _loop.is_closed():
                _loop = asyncio.new_event_loop()
                _thread = threading.Thread(target=_loop.run_forever, daemon=True)
                _thread.start()
    return _loop

def run_grpc_coro(coro, timeout=30):
    """Submit a coroutine to the gRPC loop and block for result."""
    loop = get_grpc_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)
```

**Why not `grpc.aio` native channels?** The t-tech-investments SDK wraps `grpc.aio` internally via `AsyncClient`. We cannot control the channel creation. The SDK's `AsyncClient` must be entered as an async context manager (`async with client as services:`), so we must provide an event loop for it. The shared singleton loop is the cleanest pattern given this constraint.

### GRPC_POLL_STRATEGY Environment Variable

| Value | Platform | Description |
|-------|----------|-------------|
| `epoll1` | Linux only | Sharded pollset neighborhoods. Best for Linux containers. |
| `poll` | All POSIX | Portable fallback using `poll()`. Only option on macOS. |
| `legacy` | All POSIX | Deprecated original engine. Do NOT use. |

**Recommendation:** Do NOT set `GRPC_POLL_STRATEGY`. Let grpcio auto-select (`epoll1` on Linux, `poll` on macOS). The BlockingIOError is not caused by the poll strategy choice -- it is caused by cross-thread event loop fd contention. Setting this env var will not fix the issue.

**What WILL help:**
- Set a custom `asyncio` exception handler on the gRPC loop to suppress EAGAIN from PollerCompletionQueue (these are benign):

```python
def _grpc_exception_handler(loop, context):
    exc = context.get("exception")
    if isinstance(exc, BlockingIOError):
        return  # Suppress benign EAGAIN from PollerCompletionQueue
    loop.default_exception_handler(context)

loop.set_exception_handler(_grpc_exception_handler)
```

### Confidence: MEDIUM-HIGH

The shared-loop singleton pattern is well-established in the codebase (both TinkoffFetcher and TinkoffBroker already use it independently). Consolidating into one loop is a straightforward refactor. The EAGAIN suppression is validated by the LangChain/LangGraph community experiencing the same grpcio 1.75+ issue.

**Sources:**
- [gRPC Python AsyncIO API docs](https://grpc.github.io/grpc/python/grpc_asyncio.html)
- [grpc/grpc#25364 - Multi-thread async clients](https://github.com/grpc/grpc/issues/25364)
- [gRPC environment variables](https://grpc.github.io/grpc/core/md_doc_environment_variables.html)
- [LangGraph PollerCompletionQueue issue](https://forum.langchain.com/t/pollercompletionqueue-handle-events-blockingioerror-spam-in-langgraph-cloud-logs/3232)

---

## Fix 2: gRPC Channel Reconnection on INTERNAL Errors (T-Bank 70001)

### Problem

T-Bank Sandbox API returns error code 70001 (INTERNAL) for portfolio fetches. Once this happens, the channel enters a bad state and subsequent calls fail for hours until the pod restarts.

### Solution: Channel Reset on INTERNAL Errors

No new libraries needed. The fix is retry logic in the existing `TinkoffBroker`:

1. **Catch `grpc.RpcError` with `code() == grpc.StatusCode.INTERNAL`** in `get_portfolio()` and order methods.
2. **Reset the channel**: set `self._client = None`, `self._services = None` to force lazy re-creation on next call.
3. **Cache last-known portfolio**: store the last successful `PortfolioState` and return it as fallback when INTERNAL errors persist (with a staleness warning).
4. **Exponential backoff** on reconnection attempts (already have `RetryPolicy` for this).

```python
# In TinkoffBroker methods:
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.INTERNAL:
        _log.warning("grpc_internal_error", detail=str(e.details()))
        await self._reset_channel()  # force reconnect on next call
        if self._last_portfolio and self._portfolio_age < MAX_STALENESS:
            return self._last_portfolio  # fallback
    raise BrokerError(str(e)) from e
```

### Confidence: HIGH

Standard gRPC resilience pattern. The `RetryPolicy` already exists. Channel reset is a simple state clear.

---

## Fix 3: DB Persistence Wiring (SQLAlchemy Async Sessions Across Threads)

### Problem

APScheduler's `BackgroundScheduler` runs jobs in a thread pool. The `TradingLoop` needs to persist orders, signals, news articles, and sentiment scores to PostgreSQL. But `AsyncSession` is bound to an event loop and cannot be shared across threads.

### SQLAlchemy Async Session Rules

From [SQLAlchemy 2.0 asyncio docs](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html):

1. **One AsyncSession per task** -- never share across tasks or threads.
2. **`async_sessionmaker` is thread-safe** -- the factory itself can be called from any thread to produce new sessions.
3. **`expire_on_commit=False`** is required for async sessions (already configured in `core/db.py`).
4. **AsyncSession must be used on the event loop it was created on**.

### Solution: Pass the Session Factory, Not Sessions

The existing `get_async_session_factory()` in `core/db.py` is already correct. It returns an `async_sessionmaker` which is thread-safe to call. Each APScheduler job should:

1. Get the factory (cached, thread-safe).
2. Create a new `AsyncSession` on the gRPC background loop (or a dedicated DB event loop).
3. Use it within `async with factory() as session:` scope.
4. Session auto-closes when the context manager exits.

```python
# In TradingLoop cycle methods:
async def _persist_signal(self, signal: Signal) -> None:
    factory = get_async_session_factory()
    async with factory() as session:
        session.add(SignalRecord.from_signal(signal))
        await session.commit()
```

**Key insight:** Since APScheduler jobs already call `run_coroutine_threadsafe()` to the gRPC background loop, DB operations can run on the same loop. The `async_sessionmaker` factory creates sessions bound to whatever event loop they are `await`-ed on. This works because `asyncpg` connects per-session, not per-loop.

**Alternative: Dedicated DB event loop.** If gRPC and DB operations on the same loop cause contention, spin up a second background loop for DB-only work. But start with the shared loop -- it is simpler and likely sufficient given the low frequency of DB writes (once per cycle, ~5 min intervals).

### Confidence: HIGH

The `async_sessionmaker` factory pattern is well-documented in SQLAlchemy 2.0. The existing `core/db.py` already implements it correctly with `expire_on_commit=False` and connection pooling.

**Sources:**
- [SQLAlchemy 2.0 AsyncIO docs](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [SQLAlchemy async_sessionmaker discussion](https://github.com/sqlalchemy/sqlalchemy/discussions/11539)

---

## Fix 4: Promtail -> Loki Pipeline Debugging

### Problem

Promtail is configured with `docker_sd_configs` to scrape the `finalayze-sandbox-app` container, but 0 log entries have ever appeared in Loki.

### Root Cause (Likely)

The current Promtail config mounts only `/var/run/docker.sock` but **does NOT mount the container log directory**. Docker's `docker_sd_configs` uses the socket for service discovery (finding which containers exist), but reads actual logs from the filesystem at `/var/lib/docker/containers/`.

### Solution: Add Container Log Volume Mount

In `docker-compose.sandbox.yml`, add the missing volume mount to the `promtail` service:

```yaml
promtail:
  volumes:
    - ../monitoring/promtail/promtail-config.yml:/etc/promtail/config.yml:ro
    - /var/run/docker.sock:/var/run/docker.sock:ro
    - /var/lib/docker/containers:/var/lib/docker/containers:ro  # ADD THIS
```

Additionally, verify the Promtail config has correct `__path__` relabeling. The `docker_sd_configs` provides a `__meta_docker_container_log_path` label that must be mapped to `__path__`:

```yaml
relabel_configs:
  - source_labels: ['__meta_docker_container_name']
    regex: '/(.*)'
    target_label: 'container'
  - source_labels: ['__meta_docker_container_log_path']  # ADD THIS
    target_label: '__path__'
```

Without the `__path__` label, Promtail discovers the container but has no file path to tail.

### No New Dependencies

All existing: Promtail 3.4.2, Loki 3.4.2, Docker socket.

### Confidence: HIGH

This is a well-documented Promtail misconfiguration. The missing volume mount + `__path__` relabel is the most common cause of "0 logs" with `docker_sd_configs`.

**Sources:**
- [Promtail Docker SD troubleshooting](https://community.grafana.com/t/promtail-does-not-collect-logs-from-other-containers/87000)
- [Docker + Promtail setup guide](https://ruanbekker.medium.com/logging-with-docker-promtail-and-grafana-loki-d920fd790ca8)
- [Promtail docker_sd_configs reference](https://gist.github.com/ruanbekker/c6fa9bc6882e6f324b4319c5e3622460)

---

## Fix 5: CBR XML API for FX Rate Fallback

### Problem

When gRPC T-Bank API fails for FX rates, there is no fallback. The system already has `FXRateService` in `markets/fx_service.py` that fetches from CBR's `XML_daily.asp`, but it is only used for periodic updates, not as a fallback when gRPC fails.

### CBR XML_daily.asp API Details

| Property | Value |
|----------|-------|
| **Endpoint** | `https://www.cbr.ru/scripts/XML_daily.asp` |
| **Method** | GET |
| **Date param** | `?date_req=dd/mm/yyyy` (optional, defaults to latest) |
| **Encoding** | windows-1251 |
| **Update time** | ~11:30 MSK (UTC+3) daily |
| **Rate limit** | None documented, but be polite (1 req/min max) |

**Response format:**
```xml
<?xml version="1.0" encoding="windows-1251"?>
<ValCurs Date="30.03.2026" name="Foreign Currency Market">
  <Valute ID="R01235">
    <NumCode>840</NumCode>
    <CharCode>USD</CharCode>
    <Nominal>1</Nominal>
    <Name>US Dollar</Name>
    <Value>84,7350</Value>
    <VunitRate>84,7350</VunitRate>
  </Valute>
  <!-- ... other currencies ... -->
</ValCurs>
```

**Note:** The `<Value>` field uses comma as decimal separator (Russian locale). The existing `FXRateService._parse_cbr_xml()` already handles this with `.replace(",", ".")`.

### Solution: Wire FXRateService as Fallback

The existing `FXRateService` and `CBRFetcher` already implement everything needed. The fix is wiring:

1. In `TradingLoop._strategy_cycle()`, if the gRPC-based FX rate is stale or unavailable, call `FXRateService.update_usdrub()` as fallback.
2. The `CBRFetcher` (in `data/fetchers/cbr.py`) also has `XML_dynamic.asp` for historical ranges -- already implemented and working.

**No new code for CBR parsing.** `FXRateService._parse_cbr_xml()` in `markets/fx_service.py` already parses `XML_daily.asp` correctly (lines 47-60).

### Existing Code Already Handles This

| Component | File | Status |
|-----------|------|--------|
| `FXRateService.update_usdrub()` | `markets/fx_service.py` | Implemented, uses `XML_daily.asp` |
| `FXRateService._parse_cbr_xml()` | `markets/fx_service.py` | Implemented, handles comma decimals |
| `CBRFetcher.fetch_fx_rates()` | `data/fetchers/cbr.py` | Implemented, uses `XML_dynamic.asp` |
| Fallback wiring in TradingLoop | `orchestration/trading_loop.py` | **NOT wired** -- this is the fix |

### Confidence: HIGH

The CBR API is stable (unchanged for 10+ years). The parsing code already exists and works. Only wiring is needed.

---

## Fix 6: Article Deduplication (Hash-Based Seen Cache)

### Problem

The news pipeline re-analyzes the same articles on every cycle, hitting OpenRouter rate limits. Need a seen-article cache to skip duplicates.

### Solution: In-Memory LRU + Content Hash

No new libraries needed. Use `functools.lru_cache` or a simple `dict` with TTL.

```python
import hashlib
from collections import OrderedDict

class ArticleDeduplicator:
    """Skip articles already analyzed. Hash-based, TTL-expiring."""

    def __init__(self, max_size: int = 5000, ttl_hours: int = 24) -> None:
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_hours * 3600

    def is_duplicate(self, article: NewsArticle) -> bool:
        key = self._hash(article)
        now = time.time()
        self._evict_stale(now)
        if key in self._seen:
            return True
        self._seen[key] = now
        if len(self._seen) > self._max_size:
            self._seen.popitem(last=False)  # evict oldest
        return False

    @staticmethod
    def _hash(article: NewsArticle) -> str:
        content = f"{article.title}|{article.source}|{article.url or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

**Why not Redis?** Redis is available but overkill for this. The cache only needs to survive within a single process lifecycle (articles are re-fetched on restart anyway). An in-memory `OrderedDict` with 5000 entries uses ~200KB. Simple, no network round-trips.

**Why SHA-256 truncated to 16 chars?** 16 hex chars = 64 bits of collision space. With 5000 entries, collision probability is negligible (~1.4e-12). Saves memory vs full 64-char hashes.

### Confidence: HIGH

Standard deduplication pattern. No external dependencies.

---

## What NOT to Add

| Temptation | Why Not |
|------------|---------|
| `grpclib` (pure-Python gRPC) | t-tech-investments SDK requires `grpcio`. Cannot swap. |
| `uvloop` | Would worsen the PollerCompletionQueue EAGAIN issue. Standard asyncio is safer. |
| `celery` for job scheduling | APScheduler already works. Adding Celery would require a broker and massive refactoring. |
| `tenacity` for retries | `RetryPolicy` already exists in `execution/retry.py`. No need for another retry lib. |
| `redis` for article dedup | Overkill. In-memory is sufficient for single-process news pipeline. |
| New logging framework | structlog + Promtail/Loki is the right stack. Fix the Promtail config, not the framework. |
| `cbrf` PyPI package | Already have `CBRFetcher` and `FXRateService` with CBR XML parsing. No need for a third-party wrapper. |

---

## Integration Points

### Shared gRPC Loop (Fix 1) Affects:

| Component | Current Pattern | New Pattern |
|-----------|----------------|-------------|
| `TinkoffFetcher` | Own `_loop` + `_loop_thread` | Use `grpc_loop.get_grpc_loop()` |
| `TinkoffBroker` | Own `_loop` + `_loop_thread` | Use `grpc_loop.get_grpc_loop()` |
| `FXRateService` | Own `httpx.AsyncClient` (no gRPC) | No change (HTTP, not gRPC) |

### DB Session Factory (Fix 3) Affects:

| Component | Current DB Usage | Change |
|-----------|-----------------|--------|
| `TradingLoop._strategy_cycle()` | No DB writes | Add signal/order persistence |
| `TradingLoop._news_cycle()` | No DB writes | Add article/sentiment persistence |
| `SandboxMonitorService` | Uses `get_async_session_factory()` | No change (already correct) |
| FastAPI endpoints | Use `get_db()` dependency | No change |

### Article Deduplicator (Fix 6) Affects:

| Component | Change |
|-----------|--------|
| `TradingLoop._news_cycle()` | Filter articles through deduplicator before LLM analysis |
| `RssNewsFetcher` | No change (fetcher returns all articles) |
| `TelegramChannelReader` | No change (reader returns all messages) |

---

## Summary: Zero New Dependencies

All six fixes use existing libraries:

| Fix | Libraries Used | New Code |
|-----|---------------|----------|
| gRPC loop isolation | `asyncio`, `threading` (stdlib) | `grpc_loop.py` singleton module |
| Channel reconnect | `grpc` (via t-tech-investments) | Error handling in TinkoffBroker |
| DB persistence | `sqlalchemy` 2.0.46 (existing) | Session factory calls in TradingLoop |
| Promtail pipeline | Promtail 3.4.2 (existing) | Config fix only (YAML) |
| CBR FX fallback | `httpx`, `xml.etree` (existing) | Wiring in TradingLoop |
| Article dedup | `hashlib`, `collections` (stdlib) | `ArticleDeduplicator` class |

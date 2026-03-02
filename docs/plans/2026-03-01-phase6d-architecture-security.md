# Phase 6D: Architecture & Security Fixes

**Date:** 2026-03-01
**Status:** NOT STARTED
**Scope:** 15 findings -- connection management, thread safety, security hardening, observability wiring, infrastructure pinning, FX service

---

## Executive Summary

Phase 6D addresses structural defects and security gaps found during the systems
architecture audit. The work spans six categories:

1. **Connection management** (6D.1, 6D.4, 6D.5) -- persistent Tinkoff client, DB pool sizing, Redis shutdown
2. **Thread safety** (6D.2, 6D.3) -- lock protection for stop-loss dict, lock scope reduction for sentiment writes
3. **Security hardening** (6D.6, 6D.7, 6D.12, 6D.13) -- CORS lockdown, timing-safe auth, no default passwords, model integrity
4. **Observability** (6D.8, 6D.9) -- structlog migration, MetricsCollector wiring
5. **Infrastructure** (6D.10, 6D.11) -- pinned image tags, TLS termination
6. **New service** (6D.14, 6D.15) -- portfolio caching, FX rate tracker

---

## Execution Order

Tasks are grouped into four waves. Within each wave, tasks have no mutual
dependencies and can be executed in parallel.

```
Wave 1 (security-critical, no cross-dependencies):
  6D.7  API key timing-safe comparison
  6D.12 Remove default DB password
  6D.6  CORS wildcard lockdown

Wave 2 (connection & thread safety):
  6D.1  Persistent Tinkoff client
  6D.4  DB connection pool sizing
  6D.5  Redis shutdown cleanup
  6D.2  Thread-safe stop-loss dict
  6D.3  Move Redis writes outside sentiment lock

Wave 3 (observability & ML safety):
  6D.8  structlog migration (8 modules)
  6D.9  MetricsCollector wiring into TradingLoop
  6D.13 HMAC verification for ML model files

Wave 4 (infrastructure & new services):
  6D.10 Pin TimescaleDB image tag
  6D.11 TLS termination in nginx
  6D.14 Portfolio caching per strategy cycle
  6D.15 FX rate tracking service
```

---

## Task Details

### 6D.1 -- Persistent Tinkoff async client with connection reuse

**Problem:** Every broker/fetcher call creates a new event loop via `asyncio.run()`
and opens a fresh gRPC channel (`async with AsyncClient(...) as client`). This is
both slow and leaks resources under load.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/execution/tinkoff_broker.py` | 36-217 | Store a persistent `AsyncClient` instance; lazily initialize on first call; add `close()` method |
| `src/finalayze/data/fetchers/tinkoff_data.py` | 39-139 | Same pattern: persistent client, lazy init, `close()` |

**Implementation:**

In `TinkoffBroker.__init__()` (line 44):
- Add `self._client: AsyncClient | AsyncSandboxClient | None = None`
- Add `self._client_lock = threading.Lock()` for thread-safe lazy init

Add a new method `_get_client()`:
```python
def _get_client(self) -> AsyncClient | AsyncSandboxClient:
    """Return the persistent async client, creating it lazily."""
    if self._client is None:
        with self._client_lock:
            if self._client is None:  # double-check
                cls = AsyncSandboxClient if self._sandbox else AsyncClient
                self._client = cls(self._token)
    return self._client
```

Replace all `async with client_cls(self._token) as client:` blocks (lines 78, 140,
153, 178, 208) with `client = self._get_client()` and remove the `async with`.

Add `close()` method:
```python
def close(self) -> None:
    """Close the persistent gRPC channel."""
    if self._client is not None:
        asyncio.run(self._client.__aexit__(None, None, None))
        self._client = None
```

Apply the same pattern to `TinkoffFetcher` (lines 89-105).

The `_run_async()` helper in `TradingLoop` already provides a persistent event loop
(line 160-175), so the Tinkoff clients will reuse the same loop + connection.

**Tests:**
- `tests/unit/execution/test_tinkoff_broker_persistent.py` -- verify that two
  consecutive `submit_order` calls reuse the same client instance (mock)
- `tests/unit/data/test_tinkoff_fetcher_persistent.py` -- same for fetcher
- `tests/unit/execution/test_tinkoff_broker_close.py` -- verify `close()` calls
  the client's `__aexit__`

**Dependencies:** None

---

### 6D.2 -- Thread-safe `_stop_loss_prices` dict

**Problem:** `_stop_loss_prices` is read and written from multiple APScheduler
threads (`_process_instrument` at line 447, `_check_stop_losses` at line 588,
`_submit_order` at line 568) without any synchronization.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/core/trading_loop.py` | 135, 447-448, 568-571, 588-610 | Add `_stop_loss_lock` and wrap all accesses |

**Implementation:**

In `__init__()` after line 135:
```python
self._stop_loss_lock = threading.Lock()
```

Wrap read in `_check_stop_losses()` (line 588):
```python
with self._stop_loss_lock:
    stop_price = self._stop_loss_prices.get(symbol)
```

Wrap write in `_submit_order()` (line 568):
```python
with self._stop_loss_lock:
    self._stop_loss_prices[order.symbol] = stop
```

Wrap delete in `_check_stop_losses()` (line 610):
```python
with self._stop_loss_lock:
    del self._stop_loss_prices[symbol]
```

Wrap pop in `_submit_order()` (line 571):
```python
with self._stop_loss_lock:
    self._stop_loss_prices.pop(order.symbol, None)
```

**Tests:**
- `tests/unit/core/test_trading_loop_stop_loss_lock.py` -- concurrent writes from
  two threads do not raise; verify final state is consistent

**Dependencies:** None

---

### 6D.3 -- Move Redis writes outside `_sentiment_lock`

**Problem:** In `_process_news_article()` (lines 273-282), the Redis cache write
(`self._run_async(self._cache.set_sentiment(...))`) happens while holding
`_sentiment_lock`. This blocks all other threads that need to read sentiment
(including strategy cycle threads) for the duration of the network round-trip.

Phase 5 already replaced `asyncio.run` with `_run_async`, but the lock scope
concern remains.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/core/trading_loop.py` | 273-282 | Collect updates under lock, write to Redis after releasing |

**Implementation:**

Replace the current block (lines 273-282):
```python
# Collect updates under lock
redis_updates: list[tuple[str, float]] = []
with self._sentiment_lock:
    for impact in impacts:
        existing = self._sentiment_cache.get(impact.segment_id, _DEFAULT_SENTIMENT)
        new_score = existing * 0.7 + impact.sentiment * 0.3
        self._sentiment_cache[impact.segment_id] = new_score
        redis_updates.append((impact.segment_id, new_score))

# Write to Redis outside the lock
if self._cache is not None:
    for segment_id, score in redis_updates:
        try:
            self._run_async(self._cache.set_sentiment(segment_id, score))
        except Exception:
            _log.debug("Failed to write sentiment to Redis cache")
```

**Tests:**
- `tests/unit/core/test_trading_loop_sentiment_lock.py` -- verify that the lock is
  not held during the Redis write (mock `_run_async` to assert lock is not acquired)

**Dependencies:** None

---

### 6D.4 -- Explicit DB connection pool sizing

**Problem:** `create_async_engine()` at `src/finalayze/core/db.py` line 55 uses
SQLAlchemy defaults (`pool_size=5`, `max_overflow=10`, no `pool_timeout`, no
`pool_recycle`). With 2 uvicorn workers and multiple APScheduler threads, the
default pool can be exhausted.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/core/db.py` | 55 | Add pool parameters |
| `config/settings.py` | 27 | Add pool config fields |

**Implementation:**

In `config/settings.py`, add after line 28:
```python
# DB pool
db_pool_size: int = 10
db_max_overflow: int = 5
db_pool_timeout: int = 30
db_pool_recycle: int = 1800
```

In `src/finalayze/core/db.py` line 55, change:
```python
engine = create_async_engine(url, echo=False, pool_pre_ping=True)
```
to:
```python
engine = create_async_engine(
    url,
    echo=False,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_recycle=settings.db_pool_recycle,
)
```

This requires passing `settings` into the function. Refactor
`get_async_session_factory()` to accept and use the full settings object
(it already imports `get_settings()` at line 49).

**Tests:**
- `tests/unit/core/test_db_pool_config.py` -- verify engine is created with the
  expected pool parameters (mock `create_async_engine`, assert kwargs)

**Dependencies:** None

---

### 6D.5 -- Redis connection cleanup on shutdown

**Problem:** Neither `EventBus` nor `RedisCache` connections are closed when the
`TradingLoop` shuts down. The `EventBus.close()` method exists (line 91 of
`core/events.py`) and `RedisCache.close()` exists (line 70 of `data/cache.py`),
but neither is called from `TradingLoop.stop()`.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/core/trading_loop.py` | 84-100, 222-230 | Accept `EventBus` in constructor; call `close()` on both in `stop()` |

**Implementation:**

Add `event_bus` parameter to `__init__()` (optional, defaulting to `None`):
```python
def __init__(self, ..., event_bus: EventBus | None = None) -> None:
    ...
    self._event_bus = event_bus
```

In `stop()` (after line 229), add cleanup:
```python
def stop(self) -> None:
    """Gracefully shut down scheduler, async loop, and connections."""
    if self._scheduler is not None:
        self._scheduler.shutdown(wait=True)
    if self._async_loop is not None and not self._async_loop.is_closed():
        # Close Redis connections on the async loop before stopping it
        if self._cache is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._cache.close(), self._async_loop
                ).result(timeout=5)
            except Exception:
                _log.debug("Failed to close RedisCache on shutdown")
        if self._event_bus is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._event_bus.close(), self._async_loop
                ).result(timeout=5)
            except Exception:
                _log.debug("Failed to close EventBus on shutdown")
        self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._async_thread is not None:
            self._async_thread.join(timeout=5)
    self._stop_event.set()
```

**Tests:**
- `tests/unit/core/test_trading_loop_shutdown.py` -- verify `cache.close()` and
  `event_bus.close()` are called during `stop()` (mock both)

**Dependencies:** None

---

### 6D.6 -- CORS wildcard lockdown

**Problem:** `src/finalayze/main.py` line 39 defaults CORS origins to `"*"`:
```python
allowed_origins = os.getenv("FINALAYZE_CORS_ORIGINS", "*").split(",")
```
This allows any origin to make authenticated requests in production.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/main.py` | 39 | Default to empty string (no origins allowed) |
| `config/settings.py` | -- | Add `cors_origins: list[str] = []` field |

**Implementation:**

In `config/settings.py`, add after line 86:
```python
cors_origins: list[str] = []  # FINALAYZE_CORS_ORIGINS (comma-separated)
```

In `src/finalayze/main.py` line 39, change to:
```python
settings = get_settings()
allowed_origins = settings.cors_origins if settings.cors_origins else []
```

Remove the `os.getenv` pattern entirely. When `cors_origins` is empty, the
`CORSMiddleware` will reject all cross-origin requests (same-origin still works).

**Tests:**
- `tests/unit/api/test_cors_default.py` -- verify that the default app has no
  CORS origins configured (assert `Access-Control-Allow-Origin` header is absent
  for cross-origin requests)
- `tests/unit/api/test_cors_configured.py` -- verify that setting
  `FINALAYZE_CORS_ORIGINS=https://app.example.com` allows that origin

**Dependencies:** None

---

### 6D.7 -- Timing-safe API key comparison

**Problem:** `src/finalayze/api/v1/auth.py` lines 35 and 56 use `!=` for API key
comparison:
```python
if key != expected:
```
This is vulnerable to timing attacks because Python's `!=` on strings short-circuits
on the first differing byte.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/api/v1/auth.py` | 35, 56 | Use `hmac.compare_digest()` |

**Implementation:**

Add `import hmac` at line 3.

Change line 35 (inside `require_api_key`):
```python
if not hmac.compare_digest(key, expected_key):
```

Change line 56 (inside `api_key_auth`):
```python
if not hmac.compare_digest(key, expected):
```

**Tests:**
- `tests/unit/api/test_auth_timing_safe.py` -- verify that the auth dependency
  calls `hmac.compare_digest` (mock `hmac.compare_digest` and assert it is called
  with the correct arguments)
- Existing auth tests in `tests/unit/api/` continue to pass (no behavior change)

**Dependencies:** None

---

### 6D.8 -- Migrate stdlib logging to structlog (8 modules)

**Problem:** Eight modules use `logging.getLogger(__name__)` instead of
`structlog.get_logger()`. This produces unstructured log lines that are not
captured by the structlog pipeline configured in `config/logging.py`.

**Affected modules:**

| # | File | Current (line) |
|---|------|----------------|
| 1 | `src/finalayze/core/trading_loop.py` | `logging.getLogger` (L22, L75) |
| 2 | `src/finalayze/core/alerts.py` | `logging.getLogger` (L13, L26) |
| 3 | `src/finalayze/ml/loader.py` | `logging.getLogger` (L5, L17) |
| 4 | `src/finalayze/strategies/ml_strategy.py` | `logging.getLogger` (L10, L25) |
| 5 | `src/finalayze/execution/retry.py` | `logging.getLogger` (L6, L18) |
| 6 | `src/finalayze/data/normalizer.py` | `logging.getLogger` (L5, L14) |
| 7 | `src/finalayze/api/v1/system.py` | `logging.getLogger` (L8, L25) |
| 8 | `src/finalayze/api/v1/portfolio.py` | `logging.getLogger` (L5, L15) |

**Implementation (per file):**

1. Replace `import logging` with `import structlog`
2. Replace `_log = logging.getLogger(__name__)` with `_log = structlog.get_logger()`
3. Replace `_log.exception("msg", arg1, arg2)` with `_log.exception("msg", arg1=arg1, arg2=arg2)` (structlog uses kwargs)
4. Replace `_log.warning("msg %s", val)` with `_log.warning("msg", val=val)` (same pattern)
5. Replace `_log.info("msg %s %s", a, b)` with `_log.info("msg", a=a, b=b)`
6. Replace `_log.debug("msg %s", val)` with `_log.debug("msg", val=val)`
7. Replace `_log.error("msg", extra={...})` with `_log.error("msg", **{...})`

For modules that call `_log.exception(...)` with positional format strings (e.g.
`trading_loop.py` lines 245-253), convert to structlog's keyword-based style:
```python
# Before
_log.exception("_news_cycle: failed to fetch news")
# After
_log.exception("news_cycle_fetch_failed")
```

**Tests:**
- `tests/unit/core/test_structlog_migration.py` -- for each migrated module,
  import it and verify that `_log` is a `structlog.BoundLogger` (or
  `structlog.stdlib.BoundLogger`), not a `logging.Logger`

**Dependencies:** None (but verify `config/logging.py` `setup_logging()` is called
before any module-level `structlog.get_logger()` in entry points)

---

### 6D.9 -- Wire MetricsCollector into TradingLoop

**Problem:** `MetricsCollector` in `src/finalayze/api/metrics.py` defines 18
Prometheus metrics but none are updated from the `TradingLoop`. The metrics
endpoints return zeros.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/core/trading_loop.py` | 548-575, 312-379, 743-768 | Add MetricsCollector calls |

**Implementation:**

Import at runtime (deferred, to avoid L6 import at module level):
```python
# In _submit_order, _strategy_cycle, _daily_reset:
from finalayze.api.metrics import MetricsCollector  # noqa: PLC0415
```

In `_submit_order()` after line 559 (filled branch):
```python
MetricsCollector.record_trade(
    market=market_id,
    side=order.side.lower(),
    slippage_bps=0.0,  # TODO: compute actual slippage when limit orders are added
    fill_latency_seconds=0.0,  # TODO: measure actual latency
)
```

In `_submit_order()` after line 573 (rejected branch):
```python
MetricsCollector.record_rejection(market=market_id, reason=result.reason or "unknown")
```

In `_strategy_cycle()`, after processing all instruments for a market (after line
379), add:
```python
MetricsCollector.set_portfolio_equity(market_id, float(equity))
MetricsCollector.set_circuit_breaker_level(market_id, level.value)
open_count = len([q for q in portfolio.positions.values() if q > _ZERO])
MetricsCollector.set_open_positions(market_id, open_count)
```

In `_daily_reset()` after line 756:
```python
MetricsCollector.set_daily_pnl(market_id, 0.0)  # simplified until P&L tracking
MetricsCollector.set_portfolio_equity(market_id, float(equity))
```

In `_process_instrument()` when a signal is generated (after line 451):
```python
MetricsCollector.record_signal(
    market=market_id,
    strategy=signal.strategy_name,
    direction=signal.direction.value,
)
```

**Tests:**
- `tests/unit/core/test_trading_loop_metrics.py` -- mock `MetricsCollector` static
  methods; run `_submit_order` with a filled result; assert `record_trade` was
  called with correct args
- Same for rejection path and signal recording

**Dependencies:** None (MetricsCollector is already defined, just unused)

---

### 6D.10 -- Pin TimescaleDB image tag

**Problem:** `docker/docker-compose.prod.yml` line 5 uses `latest-pg16`:
```yaml
image: timescale/timescaledb:latest-pg16
```
This means a `docker compose pull` could introduce breaking changes in production.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `docker/docker-compose.prod.yml` | 5 | Pin to `2.17.2-pg16` |

**Implementation:**

Change line 5:
```yaml
image: timescale/timescaledb:2.17.2-pg16
```

Also pin `redis:7-alpine` to `redis:7.4-alpine` (line 23) for the same reason.

**Tests:**
- `tests/infra/test_docker_compose_pinned.py` -- parse the YAML file and assert
  no image uses `latest` tag

**Dependencies:** None

---

### 6D.11 -- TLS termination in nginx

**Problem:** The nginx configuration at `docker/nginx/nginx.conf` only listens on
port 80 (HTTP). Production traffic is unencrypted.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `docker/nginx/nginx.conf` | 28-30 | Add HTTPS server block with TLS |
| `docker/docker-compose.prod.yml` | 73-74 | Expose port 443, mount cert volume |
| `docs/operations/DEPLOYMENT.md` | -- | Document cert provisioning |

**Implementation:**

Add a new `server` block in `docker/nginx/nginx.conf` for HTTPS:
```nginx
server {
    listen 443 ssl;
    server_name _;

    ssl_certificate     /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # ... same location blocks as port 80 ...
}
```

Modify the existing port-80 server to redirect to HTTPS:
```nginx
server {
    listen 80;
    server_name _;
    return 301 https://$host$request_uri;
}
```

In `docker/docker-compose.prod.yml`, update the nginx service:
```yaml
nginx:
  ...
  ports:
    - "${NGINX_HTTP_PORT:-80}:80"
    - "${NGINX_HTTPS_PORT:-443}:443"
  volumes:
    - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    - ${SSL_CERT_DIR:-./nginx/ssl}:/etc/nginx/ssl:ro
```

Add a `docker/nginx/ssl/.gitkeep` file and document that operators must mount
real certificates (e.g. from Let's Encrypt / certbot) or use a reverse proxy.

**Tests:**
- Manual verification: `docker compose -f docker/docker-compose.prod.yml config`
  succeeds
- `tests/infra/test_nginx_conf.py` -- parse nginx.conf and verify `listen 443 ssl`
  directive exists

**Dependencies:** None

---

### 6D.12 -- Remove default DB password

**Problem:** `config/settings.py` line 27 has a hardcoded default password:
```python
database_url: str = "postgresql+asyncpg://finalayze:secret@localhost:5432/finalayze"
```
`docker/docker-compose.prod.yml` line 9 also defaults to `secret`:
```yaml
POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secret}
```

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `config/settings.py` | 27 | Remove default; require via env var |
| `config/settings.py` | 90-105 | Add startup validation |
| `docker/docker-compose.prod.yml` | 9, 44 | Remove `:-secret` fallback |

**Implementation:**

In `config/settings.py` line 27, change to:
```python
database_url: str = ""
```

In the `validate_mode_requirements` validator (line 90), add after the DEBUG/TEST
early return:
```python
if not self.database_url:
    raise ValueError(
        "FINALAYZE_DATABASE_URL is required for non-DEBUG/TEST modes"
    )
```

For DEBUG/TEST modes, keep a fallback so local development still works:
```python
if self.mode in (WorkMode.DEBUG, WorkMode.TEST):
    if not self.database_url:
        self.database_url = (
            "postgresql+asyncpg://finalayze:secret@localhost:5432/finalayze"
        )
    return self
```

In `docker/docker-compose.prod.yml` lines 9 and 44, remove the `:-secret` fallback:
```yaml
POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}
```

The `${VAR:?msg}` syntax causes docker compose to fail immediately if the variable
is not set.

**Tests:**
- `tests/unit/config/test_settings_no_default_password.py` -- verify that
  instantiating `Settings(mode=WorkMode.SANDBOX)` without `database_url` raises
  `ValidationError`
- `tests/unit/config/test_settings_debug_fallback.py` -- verify that DEBUG mode
  still works without explicit `database_url`

**Dependencies:** None

---

### 6D.13 -- HMAC verification for ML model files before `joblib.load()`

**Problem:** `joblib.load()` deserializes arbitrary Python objects via pickle. A
tampered `.pkl` file can execute arbitrary code. Currently, model files are loaded
without any integrity check at:
- `src/finalayze/ml/models/xgboost_model.py` line 102
- `src/finalayze/ml/models/lightgbm_model.py` line 98

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/ml/loader.py` | 81-113 | Generate HMAC on save, verify on load |
| `src/finalayze/ml/models/xgboost_model.py` | 91-102 | Delegate save/load through loader |
| `src/finalayze/ml/models/lightgbm_model.py` | 87-98 | Same delegation |
| `config/settings.py` | -- | Add `ml_model_hmac_key: str = ""` |

**Implementation:**

Create `src/finalayze/ml/integrity.py`:
```python
"""HMAC integrity verification for serialized ML model files."""
from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from finalayze.core.exceptions import ModelIntegrityError

_DIGEST_SUFFIX = ".sha256"


def compute_hmac(path: Path, key: bytes) -> str:
    """Compute HMAC-SHA256 of a file."""
    h = hmac.new(key, digestmod=hashlib.sha256)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sign_model(path: Path, key: bytes) -> None:
    """Write an HMAC digest file alongside the model file."""
    digest = compute_hmac(path, key)
    digest_path = Path(str(path) + _DIGEST_SUFFIX)
    digest_path.write_text(digest)


def verify_model(path: Path, key: bytes) -> None:
    """Verify model file against its HMAC digest. Raises ModelIntegrityError."""
    digest_path = Path(str(path) + _DIGEST_SUFFIX)
    if not digest_path.exists():
        msg = f"No HMAC digest file for {path}"
        raise ModelIntegrityError(msg)
    expected = digest_path.read_text().strip()
    actual = compute_hmac(path, key)
    if not hmac.compare_digest(expected, actual):
        msg = f"HMAC verification failed for {path}"
        raise ModelIntegrityError(msg)
```

Add `ModelIntegrityError` to `src/finalayze/core/exceptions.py`.

In `_atomic_save()` (`src/finalayze/ml/loader.py` line 101), after `tmp_path.rename(target)`:
```python
from finalayze.ml.integrity import sign_model
key = _get_hmac_key()
if key:
    sign_model(target, key.encode())
```

In `XGBoostModel.load_from()` and `LightGBMModel.load_from()`, before `joblib.load()`:
```python
from finalayze.ml.integrity import verify_model
key = _get_hmac_key()
if key:
    verify_model(path, key.encode())
```

Where `_get_hmac_key()` reads from `settings.ml_model_hmac_key`. When the key is
empty (dev/test), verification is skipped.

**Tests:**
- `tests/unit/ml/test_integrity.py` -- test `sign_model` + `verify_model` round-trip
- `tests/unit/ml/test_integrity_tampered.py` -- modify file after signing; verify
  `ModelIntegrityError` is raised
- `tests/unit/ml/test_integrity_missing_digest.py` -- no digest file raises error

**Dependencies:** None

---

### 6D.14 -- Cache `get_portfolio()` per strategy cycle

**Problem:** `_strategy_cycle()` calls `broker.get_portfolio()` multiple times per
cycle:
- Once per market in `_get_market_equity()` (line 385)
- Once per instrument in `_process_instrument()` (line 465)
- Once per instrument in `_compute_total_equity_base()` (line 396, called from
  `_process_instrument` line 483)

For 10 instruments across 2 markets, this is ~22 API calls per cycle instead of 2.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/core/trading_loop.py` | 312-379, 381-401, 424-518 | Add per-cycle portfolio cache |

**Implementation:**

At the start of `_strategy_cycle()`, create a cycle-scoped cache:
```python
def _strategy_cycle(self) -> None:
    # Per-cycle portfolio cache: market_id -> PortfolioState
    self._cycle_portfolio_cache: dict[str, PortfolioState] = {}
    try:
        self._strategy_cycle_impl()
    finally:
        self._cycle_portfolio_cache.clear()
```

Add `_get_cached_portfolio()`:
```python
def _get_cached_portfolio(self, market_id: str) -> PortfolioState | None:
    """Return cached portfolio for this cycle, fetching once per market."""
    if market_id in self._cycle_portfolio_cache:
        return self._cycle_portfolio_cache[market_id]
    try:
        broker = self._broker_router.route(market_id)
        portfolio = broker.get_portfolio()
        self._cycle_portfolio_cache[market_id] = portfolio
        return portfolio
    except Exception:
        _log.exception("Failed to get portfolio for %s", market_id)
        return None
```

Replace `_get_market_equity()` to use the cache:
```python
def _get_market_equity(self, market_id: str) -> Decimal | None:
    portfolio = self._get_cached_portfolio(market_id)
    return portfolio.equity if portfolio is not None else None
```

In `_process_instrument()` line 465, replace `broker.get_portfolio()` with:
```python
portfolio = self._get_cached_portfolio(market_id)
if portfolio is None:
    return
```

In `_compute_total_equity_base()`, replace per-market `_get_market_equity` with the
cache (it already calls that method, so this is automatic).

**Tests:**
- `tests/unit/core/test_trading_loop_portfolio_cache.py` -- mock broker; run
  `_strategy_cycle` with 5 instruments; assert `get_portfolio()` is called exactly
  once per market (not once per instrument)

**Dependencies:** None

---

### 6D.15 -- FX rate tracking service

**Problem:** `CurrencyConverter` in `src/finalayze/markets/currency.py` uses a
hardcoded USD/RUB rate of 90.0 (line 15). There is no mechanism to update it from
a live feed.

**Files to modify:**

| File | Lines | Change |
|------|-------|--------|
| `src/finalayze/markets/fx_service.py` | NEW | Async FX rate fetcher service |
| `src/finalayze/markets/currency.py` | 30-35 | Accept injected rates |
| `src/finalayze/core/trading_loop.py` | 125, 179-220 | Schedule FX updates |
| `config/settings.py` | -- | Add `fx_update_interval_minutes: int = 60` |

**Implementation:**

Create `src/finalayze/markets/fx_service.py`:
```python
"""Periodic FX rate updater (Layer 2).

Fetches USD/RUB rate from the CBR (Central Bank of Russia) daily XML feed
or falls back to a configurable static rate.
"""
from __future__ import annotations

import structlog
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from finalayze.markets.currency import CurrencyConverter

_log = structlog.get_logger()
_CBR_DAILY_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
_USD_CHAR_CODE = "USD"


class FXRateService:
    """Fetches live FX rates and updates a CurrencyConverter."""

    def __init__(self, converter: CurrencyConverter) -> None:
        self._converter = converter
        self._client = httpx.AsyncClient(timeout=10.0)

    async def update_usdrub(self) -> Decimal | None:
        """Fetch USD/RUB from CBR and update the converter. Returns the rate."""
        try:
            response = await self._client.get(_CBR_DAILY_URL)
            response.raise_for_status()
            rate = self._parse_cbr_xml(response.text)
            if rate is not None:
                self._converter.set_rate("USDRUB", rate)
                _log.info("fx_rate_updated", pair="USDRUB", rate=float(rate))
            return rate
        except Exception:
            _log.exception("fx_rate_update_failed")
            return None

    @staticmethod
    def _parse_cbr_xml(xml_text: str) -> Decimal | None:
        """Parse CBR XML to extract USD rate."""
        import xml.etree.ElementTree as ET  # noqa: N817
        root = ET.fromstring(xml_text)  # noqa: S314
        for valute in root.findall("Valute"):
            char_code = valute.findtext("CharCode")
            if char_code == _USD_CHAR_CODE:
                nominal = int(valute.findtext("Nominal", "1"))
                value_str = valute.findtext("VCurs", "0").replace(",", ".")
                return Decimal(value_str) / nominal
        return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
```

In `TradingLoop.__init__()`, accept optional `FXRateService` and schedule it:
```python
self._fx_service = fx_service  # FXRateService | None
```

In `TradingLoop.start()`, add a scheduled job:
```python
if self._fx_service is not None:
    self._scheduler.add_job(
        self._fx_update_cycle,
        "interval",
        minutes=getattr(self._settings, "fx_update_interval_minutes", 60),
    )
```

Add `_fx_update_cycle()`:
```python
def _fx_update_cycle(self) -> None:
    """Fetch latest FX rates."""
    if self._fx_service is not None:
        rate = self._run_async(self._fx_service.update_usdrub())
        if rate is not None:
            from finalayze.api.metrics import MetricsCollector
            MetricsCollector.set_usd_rub_rate(float(rate))
```

In `TradingLoop.stop()`, close the FX service:
```python
if self._fx_service is not None:
    try:
        asyncio.run_coroutine_threadsafe(
            self._fx_service.close(), self._async_loop
        ).result(timeout=5)
    except Exception:
        _log.debug("Failed to close FXRateService on shutdown")
```

**Tests:**
- `tests/unit/markets/test_fx_service.py` -- mock httpx response with sample CBR
  XML; verify `CurrencyConverter.set_rate` is called with parsed rate
- `tests/unit/markets/test_fx_service_fallback.py` -- mock httpx to raise; verify
  converter retains old rate (no crash)
- `tests/unit/markets/test_fx_service_parse.py` -- test `_parse_cbr_xml` with real
  CBR XML snippet

**Dependencies:** 6D.9 (MetricsCollector wiring, for `set_usd_rub_rate`)

---

## File Summary Table

| File | Tasks |
|------|-------|
| `src/finalayze/api/v1/auth.py` | 6D.7 |
| `src/finalayze/api/v1/portfolio.py` | 6D.8 |
| `src/finalayze/api/v1/system.py` | 6D.8 |
| `src/finalayze/api/metrics.py` | 6D.9 (consumer, no changes) |
| `src/finalayze/core/alerts.py` | 6D.8 |
| `src/finalayze/core/db.py` | 6D.4 |
| `src/finalayze/core/exceptions.py` | 6D.13 (add `ModelIntegrityError`) |
| `src/finalayze/core/trading_loop.py` | 6D.2, 6D.3, 6D.5, 6D.9, 6D.14, 6D.15 |
| `src/finalayze/data/cache.py` | 6D.5 (consumer, no changes) |
| `src/finalayze/data/fetchers/tinkoff_data.py` | 6D.1 |
| `src/finalayze/data/normalizer.py` | 6D.8 |
| `src/finalayze/execution/retry.py` | 6D.8 |
| `src/finalayze/execution/tinkoff_broker.py` | 6D.1 |
| `src/finalayze/main.py` | 6D.6 |
| `src/finalayze/markets/currency.py` | 6D.15 (consumer, no changes) |
| `src/finalayze/markets/fx_service.py` | 6D.15 (NEW) |
| `src/finalayze/ml/integrity.py` | 6D.13 (NEW) |
| `src/finalayze/ml/loader.py` | 6D.8, 6D.13 |
| `src/finalayze/ml/models/lightgbm_model.py` | 6D.13 |
| `src/finalayze/ml/models/xgboost_model.py` | 6D.13 |
| `src/finalayze/strategies/ml_strategy.py` | 6D.8 |
| `config/settings.py` | 6D.4, 6D.6, 6D.12, 6D.13, 6D.15 |
| `docker/docker-compose.prod.yml` | 6D.10, 6D.11, 6D.12 |
| `docker/nginx/nginx.conf` | 6D.11 |
| `docker/Dockerfile.prod` | (no changes) |

---

## Test Plan Summary

| Task | Test File(s) | Type |
|------|-------------|------|
| 6D.1 | `tests/unit/execution/test_tinkoff_broker_persistent.py`, `tests/unit/data/test_tinkoff_fetcher_persistent.py` | unit |
| 6D.2 | `tests/unit/core/test_trading_loop_stop_loss_lock.py` | unit |
| 6D.3 | `tests/unit/core/test_trading_loop_sentiment_lock.py` | unit |
| 6D.4 | `tests/unit/core/test_db_pool_config.py` | unit |
| 6D.5 | `tests/unit/core/test_trading_loop_shutdown.py` | unit |
| 6D.6 | `tests/unit/api/test_cors_default.py`, `tests/unit/api/test_cors_configured.py` | unit |
| 6D.7 | `tests/unit/api/test_auth_timing_safe.py` | unit |
| 6D.8 | `tests/unit/core/test_structlog_migration.py` | unit |
| 6D.9 | `tests/unit/core/test_trading_loop_metrics.py` | unit |
| 6D.10 | `tests/infra/test_docker_compose_pinned.py` | infra |
| 6D.11 | `tests/infra/test_nginx_conf.py` | infra |
| 6D.12 | `tests/unit/config/test_settings_no_default_password.py` | unit |
| 6D.13 | `tests/unit/ml/test_integrity.py` | unit |
| 6D.14 | `tests/unit/core/test_trading_loop_portfolio_cache.py` | unit |
| 6D.15 | `tests/unit/markets/test_fx_service.py` | unit |

Total new test files: ~17

---

## Dependency Graph

```
6D.7  ──┐
6D.12 ──┤ Wave 1 (no deps)
6D.6  ──┘

6D.1  ──┐
6D.4  ──┤
6D.5  ──┤ Wave 2 (no deps)
6D.2  ──┤
6D.3  ──┘

6D.8  ──┐
6D.9  ──┤ Wave 3 (no deps)
6D.13 ──┘

6D.10 ──┐
6D.11 ──┤ Wave 4
6D.14 ──┤
6D.15 ──┘ (depends on 6D.9 for MetricsCollector.set_usd_rub_rate)
```

---

## Estimated Effort

| Complexity | Count | Estimated per task | Total |
|------------|-------|--------------------|-------|
| S (small)  | 8     | 1-2 hours          | 8-16h |
| M (medium) | 7     | 2-4 hours          | 14-28h |
| **Total**  | **15**|                    | **22-44h** |

---

## Acceptance Criteria

1. All 15 tasks implemented with tests passing
2. `uv run ruff check .` -- no new violations
3. `uv run mypy src/` -- no new type errors
4. `uv run pytest -v` -- all tests pass, coverage does not decrease
5. No `logging.getLogger` calls remain in `src/finalayze/` (6D.8 complete)
6. No `latest` Docker image tags in `docker/docker-compose.prod.yml` (6D.10 complete)
7. No default passwords in non-DEBUG settings paths (6D.12 complete)
8. `hmac.compare_digest` used for all secret comparisons (6D.7 complete)
9. CORS defaults to empty origins list (6D.6 complete)
10. ML model files have HMAC digests alongside them (6D.13 complete)

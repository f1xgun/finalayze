# Phase 6: Sandbox Validation - Research

**Researched:** 2026-03-15
**Domain:** Docker Compose deployment, gRPC error recovery, Grafana observability, APScheduler persistence, T-Invest sandbox operations
**Confidence:** HIGH

## Summary

Phase 6 validates that Finalayze can operate autonomously for 5+ consecutive trading days in the T-Invest sandbox environment. The system already has most building blocks in place: TinkoffBroker with sandbox endpoints, SandboxPortfolioTracker with shadow accounting, TelegramAlerter with priority queuing, TradingLoop with APScheduler and preflight checks, Prometheus metrics, and Docker infrastructure (Dockerfile.prod + docker-compose.prod.yml). The primary work involves hardening error recovery (gRPC reconnection in TinkoffBroker), wiring real health probes (replacing stubs in `/health`), adding Grafana to the Docker Compose stack with auto-provisioned dashboards, implementing APScheduler SQLAlchemyJobStore for job persistence across restarts, adding in-flight order reconciliation on startup, structured JSON cycle logging, and producing a final validation report.

The codebase is well-structured. TinkoffBroker already has lazy `_client` creation with `_client_lock`, a `close()` method, and RetryPolicy integration. The health endpoint has real DB/Redis probes but hardcoded `"ok"` for tinkoff/alpaca/llm. The monitoring stack (Prometheus + Alertmanager) exists but Grafana is not yet configured.

**Primary recommendation:** Work in 4 plans: (1) gRPC reconnection + health probes + staleness checks, (2) Docker Compose stack with Grafana provisioning, (3) APScheduler job store + in-flight order reconciliation + structured logging, (4) validation run orchestration + report generation.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Docker Compose stack: trading app + PostgreSQL/TimescaleDB + Redis + Prometheus + Grafana
- Run on local Mac for the 5-day validation
- Restart policy: `unless-stopped` (auto-restart on crash with backoff)
- Secrets via `env_file: .env` (already exists, in .gitignore)
- Telegram alerts (already built in Phase 5) for real-time notifications
- Grafana with one essential dashboard: equity curve (RUB), drawdown %, circuit breaker level, trade count, error rate
- Prometheus scrapes existing `/metrics` endpoint
- Real health probes replacing stubs: TinkoffBroker -> GetAccounts() ping, data feeds -> check last candle age
- `/health` endpoint returns accurate degraded/ok status for Grafana alerting
- Critical error = any exception that reaches the cycle-level catch (strict definition). Zero critical errors required for validation pass
- gRPC channel death: destroy and recreate AsyncClient with exponential backoff (30s, 60s, 120s, max 5min). Telegram alert on each retry. After 5 failed reconnects, halt trading and alert
- Market data gaps: staleness check before each cycle -- latest candle timestamp must be within expected freshness (~2 hours). Stale instruments skipped with Telegram alert
- Feed health probes wired to real checks (not stubs)
- Drawdown measured using SandboxPortfolioTracker shadow_portfolio()
- Minimum 10 round-trip trades across 5 days
- Both equity + bond cycles running together
- MOEX-only segments (no US segments)
- Starting capital: 1M RUB
- On 5% drawdown breach: continue in reduced mode (CAUTION -- halve position sizes), validation still counts if system recovers
- Daily reporting: Telegram P&L summary + Grafana dashboards + structured JSON validation log
- Final validation report after 5 days
- Continuous run for 5 days (no planned restarts)
- In-flight order handling on restart: query T-Invest GetOrders(), cancel stale orders, reconcile fills
- One deliberate kill test on day 2 or 3
- APScheduler SQLAlchemy job store: persist scheduled jobs to TimescaleDB

### Claude's Discretion
- Docker log rotation strategy (json-file driver vs mounted volume)
- Grafana dashboard layout and exact panel configuration
- Prometheus alerting rules (if any beyond Grafana visual monitoring)
- Structured JSON log schema (fields per cycle entry)
- Final report template format
- Docker Compose service health check intervals
- gRPC reconnection jitter and exact backoff parameters
- Candle staleness threshold (exact hours for "fresh" vs "stale")

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AUT-04 | T-Invest sandbox validation: 5+ days autonomous operation without critical errors | Docker Compose stack, gRPC reconnection, health probes, structured logging, validation report |
| AUT-06 | Graceful error recovery (network, API, market data gaps) | gRPC channel reconnection with backoff, candle staleness checks, in-flight order reconciliation, APScheduler job persistence |
</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| APScheduler | 3.11.2 | Job scheduling with SQLAlchemyJobStore | Already used in TradingLoop |
| prometheus-client | (installed) | Metrics collection | Already wired in api/metrics.py |
| prometheus-fastapi-instrumentator | (installed) | HTTP metrics | Already in main.py |
| structlog | (installed) | Structured JSON logging | Already project standard |
| httpx | (installed) | Telegram API, HTTP health checks | Already used |
| grpc/grpcio | (installed) | T-Invest gRPC API | Already used via t-tech-investments |

### Infrastructure (Docker images)
| Image | Version | Purpose | Why |
|-------|---------|---------|-----|
| timescale/timescaledb | 2.17.2-pg16 | Database + job store | Already in docker-compose.prod.yml |
| redis | 7.4-alpine | Cache | Already in docker-compose.prod.yml |
| prom/prometheus | v2.51.0 | Metrics scraping | Already in docker-compose.monitoring.yml |
| grafana/grafana-oss | 11.4.0 | Dashboard visualization | NEW -- needs adding |
| prom/alertmanager | v0.27.0 | Alert routing | Already in docker-compose.monitoring.yml |

### No New Dependencies Required
All code-level dependencies are already installed. Only Grafana Docker image is new.

## Architecture Patterns

### Recommended Changes to Existing Structure
```
docker/
  docker-compose.sandbox.yml   # NEW: sandbox-specific compose (extends prod)
  Dockerfile.prod              # EXISTS: multi-stage build
  entrypoint.sh                # EXISTS: needs trading loop startup addition
monitoring/
  prometheus.yml               # EXISTS
  alerts.yml                   # EXISTS
  grafana/
    provisioning/
      datasources/
        prometheus.yml         # NEW: auto-provision Prometheus datasource
      dashboards/
        dashboard.yml          # NEW: dashboard provider config
    dashboards/
      finalayze.json           # NEW: main dashboard JSON
scripts/
  run_sandbox_validation.py    # NEW: orchestrates 5-day run
  generate_validation_report.py # NEW: produces results/ markdown report
src/finalayze/
  execution/
    tinkoff_broker.py          # MODIFY: add reconnect_client() method
  api/v1/
    system.py                  # MODIFY: wire real broker/feed probes
  core/
    trading_loop.py            # MODIFY: SQLAlchemyJobStore, cycle JSON logging, order reconciliation
    validation_logger.py       # NEW: structured JSON cycle logger
```

### Pattern 1: gRPC Channel Reconnection
**What:** Destroy and recreate AsyncClient when gRPC channel dies
**When to use:** On grpc.RpcError or channel connectivity failure
**Example:**
```python
# In TinkoffBroker
def reconnect_client(self) -> bool:
    """Destroy current client and create a new one.

    Returns True on success, False if reconnection failed.
    """
    with self._client_lock:
        # Close existing client
        if self._client is not None:
            with contextlib.suppress(Exception):
                asyncio.run(self._client.__aexit__(None, None, None))
            self._client = None
        # Reset account ID (will be re-fetched on next call)
        self._account_id = ""
        # Create new client
        try:
            target = _TBANK_GRPC_SANDBOX_TARGET if self._sandbox else _TBANK_GRPC_TARGET
            self._client = AsyncClient(self._token, target=target)
            self._ensure_account_id()
            return True
        except Exception:
            self._client = None
            return False
```

### Pattern 2: Exponential Backoff Reconnection Loop
**What:** Wrap reconnection in escalating backoff with Telegram alerts
**When to use:** In TradingLoop when broker calls fail with gRPC errors
**Example:**
```python
_RECONNECT_DELAYS = [30, 60, 120, 240, 300]  # seconds, max 5min
_MAX_RECONNECT_ATTEMPTS = 5

def _attempt_grpc_reconnect(self, broker: TinkoffBroker) -> bool:
    """Try to reconnect gRPC channel with exponential backoff."""
    for attempt, delay in enumerate(_RECONNECT_DELAYS[:_MAX_RECONNECT_ATTEMPTS]):
        self._alerter.on_error(
            "gRPC", f"Channel dead, reconnect attempt {attempt + 1}/{_MAX_RECONNECT_ATTEMPTS}"
        )
        if broker.reconnect_client():
            _log.info("grpc_reconnected", attempt=attempt + 1)
            return True
        jitter = random.uniform(0.8, 1.2)
        time.sleep(delay * jitter)
    # All attempts failed -- halt trading
    self._alerter.on_error("gRPC", "All reconnect attempts failed -- halting trading")
    return False
```

### Pattern 3: APScheduler SQLAlchemy Job Store
**What:** Persist scheduled jobs so they survive container restarts
**When to use:** In TradingLoop.start() when configuring BackgroundScheduler
**Example:**
```python
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

# Use sync PostgreSQL URL (APScheduler 3.x doesn't support async)
sync_db_url = settings.database_url.replace("+asyncpg", "")
jobstores = {
    "default": SQLAlchemyJobStore(url=sync_db_url),
}
self._scheduler = BackgroundScheduler(
    timezone="UTC",
    executors=executors,
    jobstores=jobstores,
)
# Important: use replace_existing=True when adding jobs
self._scheduler.add_job(..., replace_existing=True, id="strategy_cycle")
```

### Pattern 4: Grafana Auto-Provisioning
**What:** Mount YAML config files so Grafana starts with Prometheus datasource and dashboard pre-loaded
**When to use:** Docker Compose setup
**Example provisioning/datasources/prometheus.yml:**
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

### Pattern 5: Structured JSON Cycle Logger
**What:** Append one JSON line per trading cycle for post-mortem analysis
**When to use:** At the end of each strategy_cycle and bond_cycle
**Schema (Claude's discretion):**
```python
@dataclass
class CycleLogEntry:
    timestamp: str           # ISO 8601
    cycle_type: str          # "equity" | "bond"
    duration_ms: int         # cycle wall time
    instruments_processed: int
    signals_generated: int
    orders_submitted: int
    orders_filled: int
    errors_caught: int       # non-critical errors caught within cycle
    equity_rub: float        # shadow portfolio equity
    drawdown_pct: float      # current drawdown
    circuit_breaker_level: str
```

### Pattern 6: Real Health Probes
**What:** Replace hardcoded `"ok"` stubs with actual liveness checks
**When to use:** In `_get_component_status()` in system.py
**Example:**
```python
async def _check_tinkoff() -> str:
    """Ping T-Invest via GetAccounts -- returns 'ok' or 'error'."""
    try:
        broker = _get_tinkoff_broker()  # application-scoped reference
        broker.get_portfolio()  # lightweight API call
        return "ok"
    except Exception:
        return "error"

async def _check_feed_freshness() -> str:
    """Check if latest candle data is within staleness threshold."""
    # Check last_candle_timestamp from data cache/fetcher
    # Return "ok" if within 2 hours, "stale" if older
```

### Anti-Patterns to Avoid
- **Creating new AsyncClient per API call:** T-Invest gRPC connections are expensive; keep the persistent client pattern
- **Using asyncpg URL with APScheduler 3.x:** APScheduler's SQLAlchemyJobStore requires a sync database URL; strip `+asyncpg`
- **Polling health probes on every /health call:** Keep the existing 30s cache TTL to avoid hammering broker APIs
- **Running US segments during validation:** Would add noise and is out of scope per user decision

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Job persistence | Custom file-based scheduler state | APScheduler SQLAlchemyJobStore | Built-in, handles schema, crash recovery |
| Dashboard provisioning | Manual Grafana setup via UI | Grafana provisioning YAML + JSON | Reproducible, version-controlled |
| Log rotation | Custom log management | Docker json-file driver with max-size/max-file | Built into Docker engine |
| Metrics collection | Custom metrics endpoints | prometheus-client Gauges/Counters (already exist) | Thread-safe, standard format |
| Reconnection backoff | Custom delay calculations | Simple list of delays + random jitter | Simpler than RetryPolicy for this use case |

## Common Pitfalls

### Pitfall 1: APScheduler SQLAlchemyJobStore with async URL
**What goes wrong:** `apscheduler.jobstores.sqlalchemy.SQLAlchemyJobStore` uses sync SQLAlchemy engine. Passing `postgresql+asyncpg://` URL causes immediate crash.
**Why it happens:** APScheduler 3.x predates async SQLAlchemy. The jobstore calls `create_engine()` not `create_async_engine()`.
**How to avoid:** Replace `+asyncpg` with empty string in the database URL before passing to jobstore: `url = settings.database_url.replace("+asyncpg", "")`. Install `psycopg2-binary` (or ensure `psycopg` is available) for the sync driver.
**Warning signs:** `ModuleNotFoundError: No module named 'asyncpg'` from SQLAlchemy engine.

### Pitfall 2: gRPC Channel Not Actually Dead
**What goes wrong:** A single RPC failure triggers full channel destruction when the channel is actually fine (transient network blip).
**Why it happens:** Conflating a single failed call with a dead channel.
**How to avoid:** Use RetryPolicy first (already wired, 3 retries). Only trigger channel reconnection after RetryPolicy exhausts all retries. The reconnection logic is a second line of defense.
**Warning signs:** Excessive reconnection events in logs.

### Pitfall 3: T-Invest Sandbox GetOrders() Not Available
**What goes wrong:** Attempting to call `get_orders()` to find in-flight orders but the method doesn't exist on TinkoffBroker.
**Why it happens:** TinkoffBroker currently has `get_order_state(order_id)` but no `get_orders()` (list all open orders). The T-Invest API has `GetOrders` but it's not wrapped.
**How to avoid:** Add a new `get_open_orders()` method to TinkoffBroker that wraps `client.orders.get_orders(account_id=...)`. Needed for in-flight order reconciliation on restart.
**Warning signs:** AttributeError on startup reconciliation.

### Pitfall 4: Docker Compose Network Isolation
**What goes wrong:** Prometheus can't scrape the app because they're on different Docker networks.
**Why it happens:** `docker-compose.monitoring.yml` uses `finalayze_finalayze_net` as external network, but if using a different compose file, the network name changes.
**How to avoid:** Use a single unified compose file or ensure all services share the same network name. The sandbox compose should define `finalayze_net` consistently.
**Warning signs:** Prometheus target showing "down" status.

### Pitfall 5: Shadow Portfolio Tracker Not Integrated into TradingLoop
**What goes wrong:** Drawdown is measured from raw sandbox equity (missing coupon/dividend adjustments), giving inaccurate drawdown readings.
**Why it happens:** SandboxPortfolioTracker exists but may not be wired into the TradingLoop's equity calculation path.
**How to avoid:** Ensure `_get_market_equity()` in TradingLoop uses SandboxPortfolioTracker.shadow_portfolio() when in SANDBOX mode.
**Warning signs:** Drawdown increases suddenly after a coupon payment date.

### Pitfall 6: TimescaleDB Container Memory on Mac
**What goes wrong:** PostgreSQL/TimescaleDB consumes excessive memory under Docker Desktop for Mac.
**Why it happens:** Default shared_buffers and work_mem settings are too high for a container.
**How to avoid:** Set `POSTGRES_SHARED_BUFFERS=128MB` and `POSTGRES_WORK_MEM=4MB` in compose environment, or use `command: postgres -c shared_buffers=128MB`.
**Warning signs:** Docker Desktop sluggishness, OOM kills.

### Pitfall 7: Entrypoint Only Starts API Server
**What goes wrong:** The Docker container starts uvicorn but not the TradingLoop.
**Why it happens:** Current `entrypoint.sh` only runs `uvicorn finalayze.main:app`. TradingLoop needs its own startup mechanism.
**How to avoid:** Either (a) start TradingLoop in the FastAPI lifespan handler (background thread), or (b) add a separate entrypoint/command for the trading loop process. Option (a) is simpler for a single-container setup.
**Warning signs:** Container starts, /health works, but no trading happens.

## Code Examples

### Unified Docker Compose for Sandbox Validation
```yaml
# docker/docker-compose.sandbox.yml
name: finalayze-sandbox

services:
  postgres:
    image: timescale/timescaledb:2.17.2-pg16
    container_name: finalayze-db
    environment:
      POSTGRES_USER: finalayze
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?must be set}
      POSTGRES_DB: finalayze
    volumes:
      - pgdata:/home/postgres/pgdata/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U finalayze"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - net
    restart: unless-stopped

  redis:
    image: redis:7.4-alpine
    container_name: finalayze-redis
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redisdata:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - net
    restart: unless-stopped

  app:
    build:
      context: ..
      dockerfile: docker/Dockerfile.prod
    container_name: finalayze-app
    env_file: ../.env
    environment:
      FINALAYZE_MODE: sandbox
      FINALAYZE_DATABASE_URL: postgresql+asyncpg://finalayze:${POSTGRES_PASSWORD}@postgres:5432/finalayze
      FINALAYZE_REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      FINALAYZE_TINKOFF_SANDBOX: "true"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - net
    restart: unless-stopped
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"

  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: finalayze-prometheus
    volumes:
      - ../monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ../monitoring/alerts.yml:/etc/prometheus/alerts.yml:ro
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=30d"
    networks:
      - net
    restart: unless-stopped

  grafana:
    image: grafana/grafana-oss:11.4.0
    container_name: finalayze-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:-admin}
      GF_PATHS_PROVISIONING: /etc/grafana/provisioning
    volumes:
      - ../monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ../monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - net
    restart: unless-stopped

volumes:
  pgdata:
  redisdata:
  prometheus_data:
  grafana_data:

networks:
  net:
```

### Log Rotation (Claude's Discretion Decision)
Use Docker `json-file` driver with `max-size: "50m"` and `max-file: "5"` (250MB total per container). This is the simplest approach -- no external log aggregation needed for a local 5-day test, logs stay accessible via `docker logs`, and 250MB is sufficient to not fill disk.

### Candle Staleness Threshold (Claude's Discretion Decision)
Use **2 hours** for equity candle staleness (MOEX intraday cycles are every 60 minutes, so 2 hours provides margin). For bonds, use **24 hours** (bond cycles are daily). These thresholds should be configurable via Settings.

### Docker Compose Health Check Intervals (Claude's Discretion Decision)
- PostgreSQL: `interval: 10s, timeout: 5s, retries: 5`
- Redis: `interval: 10s, timeout: 5s, retries: 5`
- App: Use existing Dockerfile.prod HEALTHCHECK (30s interval, 10s timeout, 15s start-period, 3 retries)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| APScheduler MemoryJobStore | SQLAlchemyJobStore for persistence | APScheduler 3.x (stable) | Jobs survive restarts |
| Hardcoded health stubs | Real broker/feed probes with caching | Phase 6 | Accurate degraded/ok status |
| Single docker-compose.prod.yml | Sandbox-specific compose file | Phase 6 | Isolated sandbox config |
| No cycle logging | Structured JSON per-cycle entries | Phase 6 | Post-mortem analysis |

## Open Questions

1. **Sync driver for APScheduler JobStore**
   - What we know: APScheduler 3.x requires sync SQLAlchemy. Project uses asyncpg.
   - What's unclear: Whether `psycopg2-binary` or `psycopg` is already installed.
   - Recommendation: Check `uv.lock` for psycopg; if absent, add `psycopg2-binary` to dev dependencies. Alternatively, use SQLite jobstore file (simpler, but less robust).

2. **TradingLoop startup mechanism in Docker**
   - What we know: Current entrypoint.sh only starts uvicorn (API server). TradingLoop.start() blocks forever (calls `self._stop_event.wait()`).
   - What's unclear: Whether TradingLoop is started somewhere in the FastAPI lifespan or as a separate process.
   - Recommendation: Start TradingLoop in a background thread from FastAPI lifespan. This keeps a single container with both API and trading.

3. **GetOrders API wrapper**
   - What we know: TinkoffBroker has `get_order_state(order_id)` but no `get_orders()` to list open orders.
   - What's unclear: Exact T-Invest SDK method signature for listing orders.
   - Recommendation: Add `get_open_orders() -> list[OrderStateResult]` wrapping `client.orders.get_orders(account_id=...)`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already configured) |
| Config file | pyproject.toml [tool.pytest] |
| Quick run command | `uv run pytest tests/unit/ -x -q` |
| Full suite command | `uv run pytest --cov` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AUT-04-a | gRPC reconnection logic | unit | `uv run pytest tests/unit/test_tinkoff_reconnect.py -x` | Wave 0 |
| AUT-04-b | Health probes return real status | unit | `uv run pytest tests/unit/test_api_health.py -x` | Exists (needs update) |
| AUT-04-c | APScheduler SQLAlchemyJobStore config | unit | `uv run pytest tests/unit/test_trading_loop_jobstore.py -x` | Wave 0 |
| AUT-04-d | Candle staleness check | unit | `uv run pytest tests/unit/test_candle_staleness.py -x` | Wave 0 |
| AUT-04-e | Structured cycle logger | unit | `uv run pytest tests/unit/test_validation_logger.py -x` | Wave 0 |
| AUT-04-f | Validation report generation | unit | `uv run pytest tests/unit/test_validation_report.py -x` | Wave 0 |
| AUT-06-a | In-flight order reconciliation | unit | `uv run pytest tests/unit/test_order_reconciliation.py -x` | Wave 0 |
| AUT-06-b | get_open_orders wrapper | unit | `uv run pytest tests/unit/test_tinkoff_broker.py -x` | Exists (needs new test) |
| AUT-04-g | Docker Compose stack starts cleanly | manual-only | `docker compose -f docker/docker-compose.sandbox.yml up -d` | N/A -- manual |
| AUT-04-h | 5-day sandbox run passes criteria | manual-only | Run validation script, check report | N/A -- manual |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/ -x -q`
- **Per wave merge:** `uv run pytest --cov`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_tinkoff_reconnect.py` -- covers AUT-04-a (gRPC reconnection)
- [ ] `tests/unit/test_trading_loop_jobstore.py` -- covers AUT-04-c (job store config)
- [ ] `tests/unit/test_candle_staleness.py` -- covers AUT-04-d (staleness check)
- [ ] `tests/unit/test_validation_logger.py` -- covers AUT-04-e (cycle JSON logger)
- [ ] `tests/unit/test_validation_report.py` -- covers AUT-04-f (report generation)
- [ ] `tests/unit/test_order_reconciliation.py` -- covers AUT-06-a (in-flight orders)

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `src/finalayze/execution/tinkoff_broker.py` -- current TinkoffBroker implementation with lazy client, close(), RetryPolicy
- Codebase inspection: `src/finalayze/core/trading_loop.py` -- APScheduler BackgroundScheduler setup, preflight checks, cycle structure
- Codebase inspection: `src/finalayze/api/v1/system.py` -- health endpoint with stub probes (tinkoff/alpaca/llm hardcoded "ok")
- Codebase inspection: `docker/docker-compose.prod.yml` -- existing production compose with TimescaleDB, Redis, app, nginx
- Codebase inspection: `docker-compose.monitoring.yml` -- existing Prometheus + Alertmanager stack
- Codebase inspection: `src/finalayze/api/metrics.py` -- 20+ Prometheus metrics already defined
- Codebase inspection: `src/finalayze/execution/sandbox_tracker.py` -- SandboxPortfolioTracker with shadow accounting
- [APScheduler 3.11.2 SQLAlchemyJobStore docs](https://apscheduler.readthedocs.io/en/3.x/modules/jobstores/sqlalchemy.html) -- jobstore configuration and parameters
- [APScheduler 3.x User Guide](https://apscheduler.readthedocs.io/en/3.x/userguide.html) -- scheduler configuration patterns

### Secondary (MEDIUM confidence)
- [Grafana provisioning docs](https://grafana.com/docs/grafana/latest/administration/provisioning/) -- datasource and dashboard YAML provisioning
- [Grafana Docker configuration](https://grafana.com/docs/grafana/latest/setup-grafana/configure-docker/) -- Docker-specific setup
- [gRPC Python reconnection patterns](https://blog.jeffli.me/blog/2017/08/02/keep-python-grpc-client-connection-truly-alive/) -- channel lifecycle management

### Tertiary (LOW confidence)
- T-Invest SDK `GetOrders` method signature -- needs verification at implementation time against `t-tech-investments` SDK

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, only Grafana Docker image is new
- Architecture: HIGH -- patterns are straightforward extensions of existing code
- Pitfalls: HIGH -- identified from direct codebase inspection (async URL, missing GetOrders wrapper, entrypoint gap)
- Docker/Grafana: MEDIUM -- provisioning patterns well-documented but exact Grafana JSON dashboard structure needs crafting at implementation time
- T-Invest sandbox behavior: MEDIUM -- sandbox may have quirks (latency, order behavior) that surface during the 5-day run

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (stable infrastructure, no rapidly moving dependencies)

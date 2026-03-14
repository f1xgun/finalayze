---
phase: 06-sandbox-validation
plan: 02
subsystem: infrastructure
tags: [docker, grafana, apscheduler, validation-logger, lifespan, observability]

# Dependency graph
requires:
  - phase: 06-sandbox-validation
    plan: 01
    provides: Error recovery hardening, health probes, reconnection
provides:
  - Docker Compose sandbox stack (postgres + redis + app + prometheus + grafana)
  - Grafana auto-provisioned dashboard with 5 panels
  - ValidationLogger with CycleLogEntry for structured JSONL cycle logging
  - APScheduler SQLAlchemyJobStore for persistent job scheduling
  - TradingLoop lifespan startup in FastAPI for sandbox/real modes
  - set_tinkoff_broker() health probe wiring in lifespan
affects: [06-sandbox-validation, sandbox-deployment, monitoring]

# Tech tracking
tech-stack:
  added: [grafana-oss-11.4.0, docker-compose-sandbox]
  patterns: [jsonl-cycle-logging, sqlalchemy-jobstore-fallback, lifespan-daemon-thread]

key-files:
  created:
    - docker/docker-compose.sandbox.yml
    - monitoring/grafana/provisioning/datasources/prometheus.yml
    - monitoring/grafana/provisioning/dashboards/dashboard.yml
    - monitoring/grafana/dashboards/finalayze.json
    - src/finalayze/core/validation_logger.py
    - tests/unit/test_validation_logger.py
    - tests/unit/test_trading_loop_jobstore.py
  modified:
    - src/finalayze/core/trading_loop.py
    - src/finalayze/main.py
    - docker/entrypoint.sh

key-decisions:
  - "SQLAlchemyJobStore with sync URL fallback to MemoryJobStore when psycopg2 unavailable"
  - "All APScheduler jobs have stable IDs with replace_existing=True for crash recovery"
  - "TradingLoop starts in daemon thread from FastAPI lifespan (sandbox and real modes)"
  - "Sandbox mode equity/drawdown sourced from SandboxPortfolioTracker.shadow_portfolio()"
  - "Single uvicorn worker forced in sandbox mode for TradingLoop thread safety"

patterns-established:
  - "ValidationLogger JSONL append-only pattern with threading.Lock"
  - "Lifespan-based background thread startup for trading automation"
  - "Graceful shutdown: stop trading loop + join thread in lifespan cleanup"

requirements-completed: [AUT-04]

# Metrics
duration: 6min
completed: 2026-03-15
---

# Phase 06 Plan 02: Docker Compose Sandbox Stack and Observability Summary

**Docker Compose sandbox stack with Grafana dashboard, structured JSONL cycle logging, APScheduler persistent job store, and TradingLoop lifespan startup**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-14T22:24:01Z
- **Completed:** 2026-03-14T22:30:27Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Docker Compose sandbox stack with all 5 services (app, postgres, redis, prometheus, grafana)
- Grafana auto-provisioned with Prometheus datasource and 5-panel dashboard (equity, drawdown, CB level, trades, errors)
- ValidationLogger writes/reads structured JSON cycle entries (thread-safe, JSONL format)
- APScheduler uses SQLAlchemyJobStore with sync DB URL, falls back to MemoryJobStore gracefully
- All scheduled jobs have stable IDs with replace_existing=True for crash recovery
- TradingLoop starts in background daemon thread from FastAPI lifespan in sandbox/real modes
- main.py wires set_tinkoff_broker() for real health probes during lifespan
- Cycle logging wired into both _strategy_cycle and _bond_cycle
- In sandbox mode, equity_rub and drawdown_pct sourced from SandboxPortfolioTracker

## Task Commits

Each task was committed atomically:

1. **Task 1: Structured JSON cycle logger and APScheduler job store persistence** - `ba6d52f` (feat, TDD)
2. **Task 2: Docker Compose sandbox stack with Grafana and TradingLoop lifespan startup** - `54e2597` (feat)

## Files Created/Modified
- `src/finalayze/core/validation_logger.py` - CycleLogEntry dataclass + ValidationLogger with JSONL I/O
- `src/finalayze/core/trading_loop.py` - SQLAlchemyJobStore, stable job IDs, cycle logging in strategy/bond cycles
- `src/finalayze/main.py` - Lifespan starts TradingLoop in daemon thread, wires health probes
- `docker/docker-compose.sandbox.yml` - Complete sandbox stack (5 services, health checks, volumes)
- `docker/entrypoint.sh` - Forces single worker in sandbox mode
- `monitoring/grafana/provisioning/datasources/prometheus.yml` - Prometheus datasource config
- `monitoring/grafana/provisioning/dashboards/dashboard.yml` - Dashboard provider config
- `monitoring/grafana/dashboards/finalayze.json` - 5-panel Grafana dashboard
- `tests/unit/test_validation_logger.py` - 8 tests for ValidationLogger
- `tests/unit/test_trading_loop_jobstore.py` - 4 tests for job store configuration

## Decisions Made
- SQLAlchemyJobStore uses sync URL (asyncpg stripped) since APScheduler 3.x requires sync driver
- Fallback to MemoryJobStore when psycopg2 is unavailable (graceful degradation)
- Single uvicorn worker forced in sandbox mode to prevent TradingLoop thread duplication
- TradingLoop built with minimal deps in _build_trading_loop(); full wiring deferred to runtime
- Peak equity tracked as instance variable for accurate drawdown computation across cycles

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## Next Phase Readiness
- Docker sandbox stack ready for `docker compose -f docker/docker-compose.sandbox.yml up`
- Grafana dashboard auto-provisions at http://localhost:3000
- Cycle logging produces results/validation/cycles.jsonl for report generation (Plan 03)
- Job persistence ready for crash recovery testing

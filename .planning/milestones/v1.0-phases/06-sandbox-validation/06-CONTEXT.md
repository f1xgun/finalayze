# Phase 6: Sandbox Validation - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove the system can run autonomously in T-Invest sandbox for 5+ consecutive trading days without critical errors. Deploy via Docker Compose with full observability (Telegram + Grafana). Harden error recovery (gRPC reconnection, market data staleness checks, real health probes). Validate state recovery after crashes (in-flight order reconciliation, APScheduler job store persistence). Generate a final validation report with trade log, equity curve, and error count.

</domain>

<decisions>
## Implementation Decisions

### Deployment & Runtime
- Docker Compose stack: trading app + PostgreSQL/TimescaleDB + Redis + Prometheus + Grafana
- Run on local Mac for the 5-day validation
- Restart policy: `unless-stopped` (auto-restart on crash with backoff)
- Secrets via `env_file: .env` (already exists, in .gitignore)
- Log strategy: Claude's discretion (prevent disk fill, keep logs accessible)

### Monitoring & Observability
- Telegram alerts (already built in Phase 5) for real-time notifications
- Grafana with one essential dashboard: equity curve (RUB), drawdown %, circuit breaker level, trade count, error rate
- Prometheus scrapes existing `/metrics` endpoint
- Real health probes replacing stubs: TinkoffBroker → GetAccounts() ping, data feeds → check last candle age
- `/health` endpoint returns accurate degraded/ok status for Grafana alerting

### Error Recovery
- Critical error = any exception that reaches the cycle-level catch (strict definition). Zero critical errors required for validation pass
- gRPC channel death: destroy and recreate AsyncClient with exponential backoff (30s, 60s, 120s, max 5min). Telegram alert on each retry. After 5 failed reconnects, halt trading and alert
- Market data gaps: staleness check before each cycle — latest candle timestamp must be within expected freshness (~2 hours). Stale instruments skipped with Telegram alert
- Feed health probes wired to real checks (not stubs)

### Success Criteria Details
- Drawdown measured using SandboxPortfolioTracker shadow_portfolio() (includes simulated coupon/dividend income)
- Minimum 10 round-trip trades across 5 days (proves pipeline works end-to-end)
- Both equity + bond cycles running together (full Phase 5 integration validation)
- MOEX-only segments (no US segments — already works, would add noise)
- Starting capital: 1M RUB (matches Phase 1 assumption and backtested parameters)
- On 5% drawdown breach: continue in reduced mode (CAUTION — halve position sizes), validation still counts if system recovers
- Daily reporting: Telegram P&L summary + Grafana dashboards + structured JSON validation log (one entry per cycle with metrics)
- Final validation report after 5 days: trade log, equity curve, max drawdown, error count, uptime %. Written to results/ as markdown

### Restart & State Recovery
- Continuous run for 5 days (no planned restarts) — proves long-running stability, catches memory leaks
- Docker auto-restarts only on crashes
- In-flight order handling on restart: query T-Invest GetOrders() for open/pending orders, cancel stale orders (older than fill timeout), reconcile fills that happened while down
- One deliberate kill test on day 2 or 3: kill container during market hours, verify restart + reconcile + resume
- APScheduler SQLAlchemy job store: persist scheduled jobs to TimescaleDB, missed jobs execute on restart

### Claude's Discretion
- Docker log rotation strategy (json-file driver vs mounted volume)
- Grafana dashboard layout and exact panel configuration
- Prometheus alerting rules (if any beyond Grafana visual monitoring)
- Structured JSON log schema (fields per cycle entry)
- Final report template format
- Docker Compose service health check intervals
- gRPC reconnection jitter and exact backoff parameters
- Candle staleness threshold (exact hours for "fresh" vs "stale")

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `SandboxPortfolioTracker` (`execution/sandbox_tracker.py`): Shadow accounting for coupons/dividends. Wraps TinkoffBroker. `shadow_portfolio()` returns corrected equity
- `TelegramAlerter` (`core/alerts.py`): 12 alert methods, priority queue from Phase 5. All trade/error/daily alerting wired
- `TradingLoop` (`core/trading_loop.py`): APScheduler with cycle-level exception isolation. Preflight checks (gRPC, macro, ledger). Bond/equity independent degradation
- `RetryPolicy` (`execution/retry.py`): 3 retries, exponential backoff 1-30s, jitter. Retryable vs non-retryable exception classification
- `CircuitBreaker` (`risk/circuit_breaker.py`): 4 levels (NORMAL/CAUTION/HALTED/LIQUIDATE) + cross-market breaker. Auto-reset and manual reset
- `LayerLedger` (`core/layer_ledger.py`): `reconcile_with_broker()` — diffs ledger vs broker, adds unknowns to Core, alerts on mismatches
- Health endpoint (`api/v1/system.py`): GET `/health` with DB+Redis probes, 30s cache TTL. Broker/feed probes are stubs (to be wired)
- Prometheus metrics (`api/metrics.py`): equity, drawdown, circuit breaker level, trade count, fill latency, slippage
- `docker-compose.yaml` exists with PostgreSQL, Redis, Prometheus, Alertmanager services
- `WorkMode.SANDBOX` / `WorkMode.TEST` — sandbox mode gating in `core/modes.py`

### Established Patterns
- TinkoffBroker lazy AsyncClient initialization with threading.Lock (thread-safe)
- `_run_async()` bridge: asyncio.run_coroutine_threadsafe to dedicated event loop thread
- APScheduler BackgroundScheduler with named executors
- Fire-and-forget Telegram via TelegramMessageQueue (Phase 5)
- ORM persistence via async SQLAlchemy (MacroSnapshot, LayerLedger patterns)

### Integration Points
- `docker-compose.yaml`: add Grafana service, trading app service, Grafana provisioning
- `TinkoffBroker`: add gRPC reconnection logic (destroy + recreate `_client`)
- `api/v1/system.py`: wire real broker/feed health probes replacing stubs
- `TradingLoop`: add structured JSON cycle logging, APScheduler SQLAlchemy job store
- `execution/sandbox_tracker.py`: integrate into TradingLoop startup for shadow accounting
- `scripts/`: add validation report generator script

</code_context>

<specifics>
## Specific Ideas

- T-Invest sandbox doesn't pay coupons or dividends — SandboxPortfolioTracker shadow accounting is essential for accurate drawdown measurement
- T-Invest sandbox has no rate limits but may have different latency characteristics than production
- MOEX trading hours 10:00-18:45 MSK — system should be idle outside these hours (already gated in Phase 5)
- The deliberate kill test should be documented: exact time, what was running, recovery time, any data loss
- Structured JSON log enables post-mortem analysis: one entry per cycle with timestamp, instruments processed, signals generated, orders submitted, errors caught, equity snapshot
- APScheduler SQLAlchemy job store uses `apscheduler.jobstores.sqlalchemy:SQLAlchemyJobStore` — compatible with existing async SQLAlchemy setup

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-sandbox-validation*
*Context gathered: 2026-03-15*

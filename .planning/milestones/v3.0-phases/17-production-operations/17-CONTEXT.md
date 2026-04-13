# Phase 17: Production Operations - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the kill switch (cancel orders + stop loop + escalate breakers within 30s), health monitoring (5min heartbeat with 2-miss alerting), 3-tier alert taxonomy wiring, and Telegram bot commands (/kill with confirmation, /gonogo). No dashboard UI (Phase 18). No sandbox metrics (Phase 16 — done).

</domain>

<decisions>
## Implementation Decisions

### Kill Switch Design
- `KillSwitch` class with `activate()` method — orchestrates: cancel all broker orders → stop TradingLoop scheduler → escalate CircuitBreakers to LIQUIDATE → send Telegram CRITICAL alert
- Order cancellation via `broker.cancel_all_orders()` for each active market (already exists in TinkoffBroker)
- Three triggers: Telegram `/kill` command, REST endpoint, programmatic `KillSwitch.activate()`
- Recovery requires full restart — kill switch sets persistent flag checked by `main.py` on startup

### Health Monitoring
- `HealthMonitor` class with APScheduler job every 5 minutes
- Checks: broker connectivity (API auth check), data feed freshness (last candle < 30min), TradingLoop alive (cycle count incrementing)
- Missed heartbeat detection: counter increments on check failure, resets on success; 2 consecutive failures → Telegram alert
- REST `/health/production` endpoint returns JSON with per-component status and overall pass/fail

### Telegram Bot Commands
- Extend existing `TelegramAlerter` with command handler dispatcher — reuse existing httpx client
- `/kill` requires confirmation reply within 30s ("Type CONFIRM to kill") to prevent accidental activation
- `/gonogo` runs `GoNoGoReporter.evaluate()` from Phase 16, formats as Telegram message with emoji pass/fail per criterion
- Authorization: restrict commands to `FINALAYZE_TELEGRAM_ADMIN_CHAT_ID` env var — only admin chat can trigger

### Claude's Discretion
- Internal data structures for health check results
- Kill switch persistent flag storage mechanism (file vs DB vs env)
- Telegram bot polling interval and webhook vs polling choice
- Health check timeout values for broker ping and feed freshness

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TelegramAlerter` (`core/alerts.py`) — 3-tier priority queue, rate limiting, `send_alert()` with async/sync bridge
- `TelegramAlerter.on_anomaly_detected()` and `on_go_nogo_decision()` — added in Phase 16
- `GoNoGoReporter` (`monitoring/go_no_go.py`) — 8-criterion gate evaluation, PROCEED/DEFER/ABORT
- `TinkoffBroker.cancel_all_orders()` — exists, cancels all pending orders for a market
- `CircuitBreaker` (`risk/circuit_breaker.py`) — `_level` attribute, escalation logic
- `TradingLoop` (`core/trading_loop.py`) — APScheduler-based, `_scheduler.shutdown()` to stop
- `MetricsCollector` (`api/metrics.py`) — Prometheus metrics facade

### Established Patterns
- APScheduler jobs in `TradingLoop.start()` for scheduled work
- `AlertPriority.CRITICAL` bypasses queue entirely — immediate send
- `send_alert(message, priority)` detects async context automatically
- Settings via Pydantic with `FINALAYZE_` env prefix

### Integration Points
- `main.py` — TradingLoop creation, broker initialization (kill switch needs both)
- `core/trading_loop.py` — scheduler reference for shutdown, cycle counter for health check
- `core/alerts.py` — command handler dispatch point
- `api/routes.py` — REST endpoint registration for `/health/production` and `/kill`

</code_context>

<specifics>
## Specific Ideas

- Kill switch 30-second SLA is critical — must be measured and tested
- Existing `AlertPriority.CRITICAL` already bypasses queue — perfect for kill switch alerts
- `/gonogo` reuses GoNoGoReporter from Phase 16 — no reimplementation needed
- Health monitor should be created in `main.py` alongside SandboxMonitorService

</specifics>

<deferred>
## Deferred Ideas

- Dashboard display of health status — Phase 18
- REST endpoint for `/sandbox/gonogo` — Phase 18
- Capital scaling automation — out of scope for v3.0

</deferred>

# Phase 16: Sandbox Monitoring and Go/No-Go Gate - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers sandbox metric collection (per-cycle persistence to TimescaleDB), slippage capture, a Go/No-Go gate reporter with backtest-derived thresholds, and anomaly detection with Telegram alerts. No dashboard UI (Phase 18). No kill switch (Phase 17).

</domain>

<decisions>
## Implementation Decisions

### Metric Collection Architecture
- SandboxMonitorService persists metrics to a new `sandbox_metrics` TimescaleDB hypertable (Alembic migration 005), following the existing `portfolio_snapshots` pattern
- Collection happens via post-cycle hook in `_strategy_cycle` finally block — every cycle, zero-config
- Slippage captured in `_submit_order` by computing `(fill_price - expected_price) / expected_price * 10000` bps using `candles[-1].close` as expected price
- Uptime tracked via heartbeat counter in SandboxMonitorService — increment per successful cycle, gap detection = downtime

### Go/No-Go Gate Design
- Gate evaluation is on-demand via `GoNoGoReporter.evaluate()` — called from REST endpoint and Telegram `/gonogo` command
- 3-tier result: PROCEED / DEFER / ABORT with per-criterion pass/fail breakdown
- Thresholds derived from walk-forward backtest stats in `results/iterations/history.jsonl`, computed as percentile bands, stored in `config/gate_thresholds.yaml`
- Minimum 5 trading days of sandbox data required before gate can return PROCEED

### Anomaly Detection & Alerting
- Rolling z-score (window=20 cycles) for drawdown anomalies; threshold-based for fill rate (<90%) and slippage (>50bps)
- 30-minute cooldown per metric to prevent alert fatigue from repeated threshold breaches
- Anomaly checking runs post-cycle in SandboxMonitorService after each metric persist
- Alerts via existing TelegramAlerter with `AlertPriority.CRITICAL`, new `on_anomaly_detected(metric, value, threshold)` method

### Claude's Discretion
- Internal data model for `SandboxMetricRow` (DB columns, indexes)
- GoNoGoReporter internal evaluation logic and criterion ordering
- Z-score window size tuning (20 suggested, can adjust)
- Exact gate threshold percentile values from backtest distribution

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CycleLogEntry` + `ValidationLogger` — extend for sandbox metrics, existing post-cycle logging pattern
- `TelegramAlerter.send_alert(msg, priority)` — 3-tier priority queue with rate limiting (20/min)
- `MetricsCollector` static facade — add `record_sandbox_slippage()`, `set_sandbox_fill_rate()` following existing static method pattern
- `_persist_equity_snapshots` async session factory pattern — reuse for sandbox metric persistence
- Migration 003 (`portfolio_snapshots`) — TimescaleDB hypertable creation pattern
- `results/validation/cycles.jsonl` — existing cycle log (timestamp, instruments, signals, orders, fills, errors, equity, drawdown, breaker level)
- `results/iterations/history.jsonl` — walk-forward iteration results (wf_sharpe, wf_max_drawdown, trade_count, verdict)

### Established Patterns
- APScheduler jobs in TradingLoop.start() for scheduled work
- Async session factory: `get_async_session_factory()` → `async with factory() as session`
- Prometheus metrics via module-level singletons in `api/metrics.py`
- JSONL logging via `ValidationLogger.log_cycle(entry)` in `_strategy_cycle` finally block

### Integration Points
- `trading_loop.py` `_strategy_cycle` finally block — post-cycle hook for SandboxMonitorService
- `trading_loop.py` `_submit_order` — slippage capture point (line ~1467, `candles[-1].close` vs `result.fill_price`)
- `alerts.py` `TelegramAlerter` — add `on_anomaly_detected()` and `on_go_nogo_decision()` methods
- `api/metrics.py` `MetricsCollector` — slippage_bps currently hardcoded to 0.0 (line ~1469)
- `_daily_reset` — natural trigger point for daily anomaly summary

</code_context>

<specifics>
## Specific Ideas

- Slippage in sandbox is synthetic (Tinkoff sandbox = 100% fill rate). Use MOEX ISS mid-price as reference for more realistic slippage measurement (per STATE.md blocker).
- GoNoGoReporter evaluates 8 criteria per success criteria: uptime, fill rate, max DD, trades, signal frequency, critical errors, slippage, signal divergence
- Gate result should be a Pydantic model for easy serialization to REST endpoint (Phase 18)

</specifics>

<deferred>
## Deferred Ideas

- REST endpoint `/sandbox/gonogo` — Phase 18 (Dashboard and API Integration)
- Streamlit sandbox dashboard page — Phase 18
- Telegram `/gonogo` command — Phase 17 (Production Operations)

</deferred>

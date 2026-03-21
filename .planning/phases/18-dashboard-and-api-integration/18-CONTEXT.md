# Phase 18: Dashboard and API Integration - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers a Streamlit sandbox dashboard page (trade log, equity curve, uptime, fill rate, slippage histogram) and a REST API endpoint for Go/No-Go gate reports. Consumes data from Phase 16 (SandboxMonitorService, GoNoGoReporter) and Phase 17 (HealthMonitor). No new data collection or monitoring logic.

</domain>

<decisions>
## Implementation Decisions

### Streamlit Sandbox Dashboard
- Single new page `src/finalayze/dashboard/pages/sandbox.py` with 5 sections: trade log table, equity curve chart, uptime %, fill rate gauge, slippage histogram
- Data sourced from `sandbox_metrics` TimescaleDB hypertable via async session queries
- Charts rendered with Plotly (already used in existing dashboard pages)
- Auto-refresh via `st.cache_data(ttl=60)` — 1 minute cache for all DB queries

### REST API Go/No-Go Endpoint
- `GET /api/v1/sandbox/gonogo` under existing v1 router
- Response: direct serialization of `GateReport` Pydantic model (verdict, criteria list, timestamp)
- Authentication: existing X-API-Key header auth (same as all other endpoints)
- When insufficient sandbox data: return `DEFER` verdict with message "Insufficient data — need 5 trading days"

### Claude's Discretion
- Dashboard layout and column widths
- Chart color scheme and styling
- SQL query optimization for TimescaleDB aggregations
- API client method for dashboard to call /sandbox/gonogo

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/finalayze/dashboard/` — existing Streamlit app with multiple pages
- `src/finalayze/dashboard/api_client.py` — API client used by dashboard to call REST endpoints
- `src/finalayze/api/v1/` — existing v1 router with portfolio, positions, health endpoints
- `GoNoGoReporter` (`monitoring/go_no_go.py`) — evaluate() returns GateReport Pydantic model
- `SandboxMetricRow` (`core/models.py`) — ORM model for sandbox_metrics table
- `get_async_session_factory()` — async DB session pattern

### Established Patterns
- Dashboard pages in `src/finalayze/dashboard/pages/` as separate .py files
- API routes registered in `src/finalayze/api/v1/` with FastAPI router
- Plotly charts in existing dashboard pages
- `st.cache_data(ttl=N)` for dashboard data caching

### Integration Points
- `src/finalayze/dashboard/app.py` — Streamlit app entry, auto-discovers pages
- `src/finalayze/api/v1/__init__.py` — router registration
- `monitoring/go_no_go.py` — GoNoGoReporter.evaluate() for gate endpoint

</code_context>

<specifics>
## Specific Ideas

- Dashboard should show last 7 days of sandbox metrics by default with date range selector
- Equity curve should include drawdown overlay
- Metrics table should be sortable with columns: timestamp, trade_count, pnl_rub, equity_rub, fill_rate, max_slippage_bps (sourced from SandboxMetricRow aggregate cycle data)
- Slippage histogram should show distribution with 50bps threshold line

</specifics>

<deferred>
## Deferred Ideas

None — this is the final phase of v3.0.

</deferred>

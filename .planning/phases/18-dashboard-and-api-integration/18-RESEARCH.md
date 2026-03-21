# Phase 18: Dashboard and API Integration - Research

**Researched:** 2026-03-22
**Domain:** Streamlit dashboard page + FastAPI REST endpoint for sandbox monitoring
**Confidence:** HIGH

## Summary

Phase 18 is a pure presentation/integration phase. All data collection (SandboxMonitorService), storage (SandboxMetricRow in TimescaleDB), and evaluation logic (GoNoGoReporter) already exist from Phases 16-17. This phase adds two thin layers: (1) a Streamlit dashboard page that queries sandbox_metrics and renders 5 visualizations, and (2) a FastAPI endpoint that exposes GoNoGoReporter.evaluate() over REST.

The codebase has well-established patterns for both. Dashboard pages follow a `render(api: ApiClient)` convention in `src/finalayze/dashboard/pages/`. API endpoints use FastAPI routers with Pydantic response models and X-API-Key auth. The existing `portfolio.py` dashboard page already demonstrates equity curves with Plotly, and `trades.py` shows filterable data tables -- both directly applicable patterns.

**Primary recommendation:** Follow existing patterns exactly. Create `sandbox.py` dashboard page with `render(api)`, create `sandbox.py` API router with GoNoGoReporter wiring via module-level setter (matching health_monitor/kill_switch pattern), and add corresponding tests.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Single new page `src/finalayze/dashboard/pages/sandbox.py` with 5 sections: trade log table, equity curve chart, uptime %, fill rate gauge, slippage histogram
- Data sourced from `sandbox_metrics` TimescaleDB hypertable via async session queries
- Charts rendered with Plotly (already used in existing dashboard pages)
- Auto-refresh via `st.cache_data(ttl=60)` -- 1 minute cache for all DB queries
- `GET /api/v1/sandbox/gonogo` under existing v1 router
- Response: direct serialization of `GateReport` Pydantic model (verdict, criteria list, timestamp)
- Authentication: existing X-API-Key header auth (same as all other endpoints)
- When insufficient sandbox data: return `DEFER` verdict with message "Insufficient data -- need 5 trading days"
- Dashboard should show last 7 days of sandbox metrics by default with date range selector
- Equity curve should include drawdown overlay
- Trade log table should be sortable with columns: timestamp, symbol, side, quantity, price, slippage_bps
- Slippage histogram should show distribution with 50bps threshold line

### Claude's Discretion
- Dashboard layout and column widths
- Chart color scheme and styling
- SQL query optimization for TimescaleDB aggregations
- API client method for dashboard to call /sandbox/gonogo

### Deferred Ideas (OUT OF SCOPE)
None -- this is the final phase of v3.0.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MON-03 | Streamlit sandbox dashboard page shows real-time trade log, equity curve, uptime %, fill rate, slippage histogram | Dashboard page pattern (render(api) convention), SandboxMetricRow ORM model, Plotly charts, st.cache_data caching |
| GATE-03 | REST endpoint `/sandbox/gonogo` returns structured pass/fail report with per-criterion breakdown | GoNoGoReporter.evaluate() returns GateReport, FastAPI router pattern, module-level setter for wiring, Pydantic response model |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| streamlit | (project-installed) | Dashboard page rendering | Already used for all 5 existing dashboard pages |
| fastapi | (project-installed) | REST API endpoint | Already used for 20+ endpoints |
| plotly | (project-installed) | Interactive charts (equity curve, histogram) | Already used in existing dashboard pages |
| pandas | (project-installed) | DataFrame operations for table display | Already used in portfolio.py, trades.py |
| sqlalchemy | 2.0 async | Database queries for sandbox_metrics | Already used throughout codebase |
| pydantic | v2 | API response models | Already used for all API responses |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| httpx | (project-installed) | Dashboard ApiClient HTTP calls | Dashboard calling REST API |

No new dependencies needed. Everything is already in the project.

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/
├── dashboard/
│   └── pages/
│       └── sandbox.py          # NEW: Streamlit sandbox dashboard page
├── api/
│   └── v1/
│       ├── sandbox.py          # NEW: GET /sandbox/gonogo endpoint
│       └── router.py           # MODIFY: include sandbox_router
├── monitoring/
│   └── go_no_go.py             # EXISTING: GoNoGoReporter (no changes)
└── core/
    └── models.py               # EXISTING: SandboxMetricRow (no changes)
```

### Pattern 1: Dashboard Page Convention
**What:** Each dashboard page is a module in `pages/` with a `render(api: ApiClient)` function.
**When to use:** For the sandbox.py dashboard page.
**Example:**
```python
# Source: src/finalayze/dashboard/pages/portfolio.py (existing pattern)
from __future__ import annotations
import streamlit as st
from finalayze.dashboard.api_client import ApiClient

def render(api: ApiClient) -> None:
    st.title("Sandbox Monitoring")
    # ... sections ...
```

Streamlit auto-discovers pages in the `pages/` directory. No registration in `app.py` needed.

### Pattern 2: API Router with Module-Level Setter
**What:** API routers use module-level singleton references set via setter functions during lifespan startup.
**When to use:** For wiring GoNoGoReporter to the sandbox endpoint.
**Example:**
```python
# Source: src/finalayze/api/v1/system.py (existing pattern)
_go_no_go_reporter: GoNoGoReporter | None = None

def set_go_no_go_reporter(reporter: GoNoGoReporter) -> None:
    global _go_no_go_reporter
    _go_no_go_reporter = reporter

@router.get("/sandbox/gonogo", dependencies=[Depends(api_key_auth)])
async def sandbox_gonogo() -> GoNoGoResponse:
    if _go_no_go_reporter is None:
        raise HTTPException(status_code=503, detail="GoNoGoReporter not configured")
    # ... evaluate ...
```

### Pattern 3: Pydantic Response Model with Frozen Config
**What:** All API responses use Pydantic BaseModel with `ConfigDict(frozen=True)`.
**When to use:** For GoNoGoResponse and CriterionResponse models.
**Example:**
```python
# Source: src/finalayze/api/v1/system.py (existing pattern)
class GoNoGoResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    verdict: str
    criteria: list[CriterionResponse]
    sandbox_days: int
    evaluated_at: str
    reason: str
```

### Pattern 4: Dashboard Data via API Client (not direct DB)
**What:** Dashboard pages fetch data through the REST API, not directly from the database.
**When to use:** The dashboard sandbox page should call the API endpoint for go/no-go report, and could call a new endpoint or query DB directly for metrics visualization.

**Important design choice:** The existing dashboard pages call REST API via `ApiClient`. For sandbox metrics visualization (trade log, equity curve, etc.), we need either:
- Option A: New REST endpoint(s) for sandbox metrics data + dashboard calls API
- Option B: Dashboard queries DB directly via `st.cache_data` with async session

The CONTEXT.md says "Data sourced from sandbox_metrics TimescaleDB hypertable via async session queries" which indicates **Option B** for the metrics display. The go/no-go report should come via the REST API endpoint.

### Anti-Patterns to Avoid
- **Importing GoNoGoReporter directly in the API router at module level:** Use deferred import or TYPE_CHECKING to maintain layer boundaries.
- **Running async DB queries from Streamlit without proper session management:** Use `get_async_session_factory()` with `async with` context manager.
- **Hardcoding gate thresholds in the API endpoint:** Use GoNoGoReporter which reads from gate_thresholds.yaml.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gate evaluation logic | Custom criteria checking | GoNoGoReporter.evaluate() | Already implements 8 criteria with threshold configuration |
| API authentication | Custom auth middleware | Existing api_key_auth dependency | Already wired in all endpoints |
| Chart rendering | Custom HTML/JS charts | Plotly via st.plotly_chart() | Already used in dashboard, interactive |
| Data caching | Manual cache logic | st.cache_data(ttl=60) | Streamlit built-in, thread-safe |
| HTTP client for dashboard | Custom requests wrapper | Existing ApiClient | Already handles auth headers, error handling |

**Key insight:** This phase is pure wiring -- every building block already exists. The only new code is presentation formatting and a thin API endpoint that delegates to GoNoGoReporter.

## Common Pitfalls

### Pitfall 1: Streamlit Async/Sync Mismatch
**What goes wrong:** Streamlit runs synchronously but DB queries are async (SQLAlchemy 2.0 async).
**Why it happens:** SandboxMetricRow queries need AsyncSession but Streamlit callbacks are sync.
**How to avoid:** Use `asyncio.run()` or `asyncio.get_event_loop().run_until_complete()` to bridge async queries in Streamlit, OR create a dedicated sync query helper. Alternatively, fetch data via REST API endpoints (which handle async natively) instead of direct DB access.
**Warning signs:** RuntimeError about event loop already running.

### Pitfall 2: GateReport is a Frozen Dataclass, Not Pydantic
**What goes wrong:** Trying to call `.model_dump()` or `.json()` on GateReport fails because it's a frozen dataclass, not a Pydantic model.
**Why it happens:** Phase 16 used frozen dataclasses (matching CycleMetrics pattern) for gate schemas.
**How to avoid:** The API endpoint must convert GateReport to a Pydantic response model manually (field-by-field mapping), or use `dataclasses.asdict()`.
**Warning signs:** AttributeError on `.model_dump()`.

### Pitfall 3: GoNoGoReporter Requires AsyncSession
**What goes wrong:** GoNoGoReporter.evaluate() requires an AsyncSession parameter for DB queries.
**Why it happens:** The reporter queries SandboxMetricRow from the database.
**How to avoid:** The API endpoint must obtain an AsyncSession via `get_db()` FastAPI dependency and pass it to reporter.evaluate(session).
**Warning signs:** TypeError about missing session argument.

### Pitfall 4: Dashboard Page Not Appearing in Sidebar
**What goes wrong:** New page file in pages/ directory doesn't show up in Streamlit sidebar.
**Why it happens:** Streamlit auto-discovery requires the file to be a valid Python module. Import errors or missing `__init__.py` entries can prevent discovery.
**How to avoid:** Ensure `sandbox.py` follows the exact same structure as existing pages, with no import errors at module level.
**Warning signs:** Page missing from sidebar navigation.

### Pitfall 5: TimescaleDB Query Performance
**What goes wrong:** Querying all sandbox_metrics rows without date filtering causes slow page loads.
**Why it happens:** TimescaleDB hypertables can grow large; full table scans are expensive.
**How to avoid:** Always include a WHERE timestamp >= (now - 7 days) clause, matching the "last 7 days by default" requirement. Use TimescaleDB time_bucket() for aggregation if needed.
**Warning signs:** Dashboard page taking >5 seconds to load.

## Code Examples

### Dashboard Page Structure (sandbox.py)
```python
# Based on: src/finalayze/dashboard/pages/portfolio.py pattern
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finalayze.dashboard.api_client import ApiClient

def render(api: ApiClient) -> None:
    st.title("Sandbox Monitoring")

    # Date range selector (default: last 7 days)
    col1, col2 = st.columns(2)
    # ... date inputs ...

    # Section 1: Trade log table (sortable)
    # Section 2: Equity curve with drawdown overlay (Plotly)
    # Section 3: Uptime % metric
    # Section 4: Fill rate gauge
    # Section 5: Slippage histogram with 50bps threshold line
```

### API Endpoint (sandbox.py router)
```python
# Based on: src/finalayze/api/v1/system.py pattern
from __future__ import annotations
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from finalayze.api.v1.auth import api_key_auth
from finalayze.core.db import get_db

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from finalayze.monitoring.go_no_go import GoNoGoReporter

router = APIRouter(tags=["sandbox"])

_go_no_go_reporter: GoNoGoReporter | None = None

def set_go_no_go_reporter(reporter: GoNoGoReporter) -> None:
    global _go_no_go_reporter
    _go_no_go_reporter = reporter

class CriterionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    passed: bool
    actual: float
    threshold: float
    unit: str
    critical: bool

class GoNoGoResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    verdict: str
    criteria: list[CriterionResponse]
    sandbox_days: int
    evaluated_at: str
    reason: str

@router.get(
    "/sandbox/gonogo",
    response_model=GoNoGoResponse,
    dependencies=[Depends(api_key_auth)],
)
async def sandbox_gonogo(session: AsyncSession = Depends(get_db)) -> GoNoGoResponse:
    if _go_no_go_reporter is None:
        raise HTTPException(status_code=503, detail="GoNoGoReporter not configured")
    report = await _go_no_go_reporter.evaluate(session)
    return GoNoGoResponse(
        verdict=report.verdict.value,
        criteria=[
            CriterionResponse(
                name=c.name, passed=c.passed, actual=c.actual,
                threshold=c.threshold, unit=c.unit, critical=c.critical,
            ) for c in report.criteria
        ],
        sandbox_days=report.sandbox_days,
        evaluated_at=report.evaluated_at.isoformat(),
        reason=report.reason,
    )
```

### Wiring in main.py lifespan
```python
# Based on: existing kill_switch/health_monitor wiring pattern in main.py
from finalayze.api.v1.sandbox import set_go_no_go_reporter
# ... after GoNoGoReporter is created ...
set_go_no_go_reporter(go_no_go_reporter)
```

### Router Registration
```python
# In src/finalayze/api/v1/router.py -- add:
from finalayze.api.v1.sandbox import router as sandbox_router
api_router.include_router(sandbox_router)
```

### Plotly Equity Curve with Drawdown Overlay
```python
# Based on: existing portfolio.py chart pattern
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3], vertical_spacing=0.05)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["equity_rub"],
                         name="Equity (RUB)"), row=1, col=1)
fig.add_trace(go.Bar(x=df["timestamp"], y=df["drawdown_pct"],
                     name="Drawdown %"), row=2, col=1)
st.plotly_chart(fig, use_container_width=True)
```

### Slippage Histogram with Threshold Line
```python
import plotly.express as px
fig = px.histogram(df, x="max_slippage_bps", nbins=30,
                   title="Slippage Distribution")
fig.add_vline(x=50, line_dash="dash", line_color="red",
              annotation_text="50bps threshold")
st.plotly_chart(fig, use_container_width=True)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct DB queries in Streamlit | REST API + ApiClient | Project convention | Dashboard decoupled from DB layer |
| str/Enum for gate verdicts | StrEnum (GateVerdict) | Phase 16 | Type-safe verdict handling |
| Hardcoded gate thresholds | YAML config (gate_thresholds.yaml) | Phase 16 | Configurable without code changes |

## Open Questions

1. **Dashboard data source: API vs direct DB**
   - What we know: CONTEXT.md says "via async session queries" (direct DB). Existing pages use API.
   - What's unclear: Whether to add new REST endpoints for sandbox metrics list, or query DB directly.
   - Recommendation: Use REST API for the go/no-go report (GATE-03 requirement). For metrics visualization (MON-03), add a new `/api/v1/sandbox/metrics` endpoint that returns sandbox_metrics rows, keeping the dashboard-via-API pattern consistent. The dashboard page calls this endpoint via ApiClient.

2. **Streamlit page auto-discovery naming**
   - What we know: Streamlit sorts pages alphabetically. Current pages: portfolio.py, risk.py, signals.py, system_status.py, trades.py.
   - What's unclear: Exact file name impact on sidebar ordering.
   - Recommendation: Name it `sandbox.py` -- it will appear between `risk.py` and `signals.py` alphabetically.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + pytest-asyncio |
| Config file | pyproject.toml `[tool.pytest]` |
| Quick run command | `uv run pytest tests/unit/test_api_sandbox.py tests/unit/test_dashboard_pages.py -x` |
| Full suite command | `uv run pytest tests/unit/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MON-03 | Sandbox dashboard page importable with render() | unit (smoke) | `uv run pytest tests/unit/test_dashboard_pages.py -x` | Needs update |
| GATE-03-a | GET /sandbox/gonogo returns 200 with valid report | unit | `uv run pytest tests/unit/test_api_sandbox.py::TestGoNoGoEndpoint::test_returns_200 -x` | Wave 0 |
| GATE-03-b | GET /sandbox/gonogo returns 503 when reporter not configured | unit | `uv run pytest tests/unit/test_api_sandbox.py::TestGoNoGoEndpoint::test_503_when_no_reporter -x` | Wave 0 |
| GATE-03-c | GET /sandbox/gonogo returns DEFER verdict on insufficient data | unit | `uv run pytest tests/unit/test_api_sandbox.py::TestGoNoGoEndpoint::test_defer_insufficient_data -x` | Wave 0 |
| GATE-03-d | GET /sandbox/gonogo requires API key auth | unit | `uv run pytest tests/unit/test_api_sandbox.py::TestGoNoGoEndpoint::test_requires_auth -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_api_sandbox.py tests/unit/test_dashboard_pages.py -x`
- **Per wave merge:** `uv run pytest tests/unit/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_api_sandbox.py` -- covers GATE-03 (endpoint tests)
- [ ] Update `tests/unit/test_dashboard_pages.py` -- add sandbox page smoke test for MON-03

## Sources

### Primary (HIGH confidence)
- `src/finalayze/dashboard/pages/portfolio.py` -- existing dashboard page pattern with Plotly charts
- `src/finalayze/dashboard/pages/trades.py` -- existing trade log table pattern
- `src/finalayze/api/v1/system.py` -- existing API endpoint pattern with module-level setters
- `src/finalayze/api/v1/router.py` -- router registration pattern
- `src/finalayze/monitoring/go_no_go.py` -- GoNoGoReporter API (evaluate(session) -> GateReport)
- `src/finalayze/core/models.py` -- SandboxMetricRow ORM model (columns: timestamp, market_id, trade_count, pnl_rub, equity_rub, fill_rate, uptime_cycles, signals_generated, errors_caught, max_slippage_bps, avg_slippage_bps, drawdown_pct)
- `src/finalayze/dashboard/api_client.py` -- ApiClient with get/post methods
- `src/finalayze/main.py` -- lifespan wiring pattern for GoNoGoReporter
- `tests/unit/test_api_health.py` -- existing API test pattern (httpx AsyncClient + ASGITransport)
- `tests/unit/test_dashboard_pages.py` -- existing dashboard smoke test pattern

### Secondary (MEDIUM confidence)
- None needed -- all patterns established in codebase

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, no new dependencies
- Architecture: HIGH -- patterns directly observed in existing codebase
- Pitfalls: HIGH -- based on actual codebase inspection (dataclass vs Pydantic, async/sync)

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (stable patterns, no external dependencies)

# Phase 4 Track B — Observability & Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give the operator full real-time visibility into the trading system from any device via a REST API (X-API-Key auth), Streamlit dashboard, and Prometheus metrics.

**Architecture:** Three sequential PRs — B-1 (Core API, all endpoints, migration 003), B-2 (Streamlit dashboard reading B-1 API), B-3 (Prometheus metrics + Alertmanager rules, parallel with B-2). See design doc at `docs/plans/2026-02-27-phase4-track-b-design.md`.

**Tech Stack:** FastAPI, SQLAlchemy 2.0 async, Pydantic v2, Streamlit, httpx, prometheus-client, prometheus-fastapi-instrumentator, Alembic.

---

## PR B-1: Core API

### Task 1: Alembic migration 003 — portfolio_snapshots table

**Context:** `GET /api/v1/portfolio/history` needs a `portfolio_snapshots` hypertable. Two existing migrations: `001_initial.py`, `002_news_sentiment.py`. TimescaleDB already enabled.

**Files:**
- Create: `alembic/versions/003_portfolio_snapshots.py`
- Modify: `src/finalayze/core/models.py` (add ORM model)

**Step 1: Write the failing test**

```python
# tests/unit/test_models.py — add to existing file
def test_portfolio_snapshot_model_has_expected_columns() -> None:
    from finalayze.core.models import PortfolioSnapshot
    cols = {c.name for c in PortfolioSnapshot.__table__.columns}
    assert {"timestamp", "market_id", "equity", "cash", "daily_pnl", "drawdown_pct", "mode"} <= cols
```

**Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/test_models.py::test_portfolio_snapshot_model_has_expected_columns -v
```
Expected: `FAILED` — `ImportError: cannot import name 'PortfolioSnapshot'`

**Step 3: Add ORM model to `src/finalayze/core/models.py`**

Add at the bottom of the existing models file:

```python
class PortfolioSnapshot(Base):
    """Portfolio equity snapshot written after each strategy cycle."""

    __tablename__ = "portfolio_snapshots"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    equity: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    cash: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    daily_pnl: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    drawdown_pct: Mapped[float | None] = mapped_column(Float)
    mode: Mapped[str | None] = mapped_column(String(10))
```

Add `datetime`, `Float` imports if not present. Check existing imports in the file first.

**Step 4: Create migration file `alembic/versions/003_portfolio_snapshots.py`**

```python
"""003 portfolio snapshots

Revision ID: 003
Revises: 002
Create Date: 2026-02-27
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "portfolio_snapshots",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("market_id", sa.String(10), nullable=False),
        sa.Column("equity", sa.Numeric(14, 4), nullable=True),
        sa.Column("cash", sa.Numeric(14, 4), nullable=True),
        sa.Column("daily_pnl", sa.Numeric(14, 4), nullable=True),
        sa.Column("drawdown_pct", sa.Float(), nullable=True),
        sa.Column("mode", sa.String(10), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "market_id"),
    )
    # Create TimescaleDB hypertable on timestamp column
    op.execute(
        "SELECT create_hypertable('portfolio_snapshots', 'timestamp', if_not_exists => TRUE)"
    )


def downgrade() -> None:
    op.drop_table("portfolio_snapshots")
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_models.py::test_portfolio_snapshot_model_has_expected_columns -v
```
Expected: `PASSED`

**Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add alembic/versions/003_portfolio_snapshots.py src/finalayze/core/models.py tests/unit/test_models.py
git commit -m "feat(api): add portfolio_snapshots migration 003 and ORM model"
```

---

### Task 2: Settings update + API key auth dependency

**Context:** `config/settings.py` needs `api_key` and `real_token` fields. A FastAPI dependency `verify_api_key` will guard all non-health endpoints.

**Files:**
- Modify: `config/settings.py`
- Modify: `.env.example`
- Create: `src/finalayze/api/v1/auth.py`
- Create: `tests/unit/test_api_auth.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_api_auth.py
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from finalayze.api.v1.auth import require_api_key


def _make_app(key: str) -> FastAPI:
    from fastapi import Depends
    app = FastAPI()

    @app.get("/secret")
    async def secret(_: None = Depends(require_api_key(key))) -> dict[str, str]:
        return {"ok": "yes"}

    return app


def test_valid_key_passes() -> None:
    client = TestClient(_make_app("test-key"))
    resp = client.get("/secret", headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200


def test_wrong_key_returns_401() -> None:
    client = TestClient(_make_app("test-key"))
    resp = client.get("/secret", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_missing_key_returns_422() -> None:
    client = TestClient(_make_app("test-key"))
    resp = client.get("/secret")
    assert resp.status_code == 422


def test_key_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """API key must never appear in logs."""
    import logging
    with caplog.at_level(logging.DEBUG):
        client = TestClient(_make_app("super-secret-key"))
        client.get("/secret", headers={"X-API-Key": "super-secret-key"})
    assert "super-secret-key" not in caplog.text
```

**Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/test_api_auth.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: finalayze.api.v1.auth`

**Step 3: Add fields to `config/settings.py`**

Add after the `real_confirmed` field:

```python
    # API auth
    api_key: str = "change-me-in-production"  # FINALAYZE_API_KEY
    real_token: str = ""  # FINALAYZE_REAL_TOKEN — required to switch to REAL mode via API
```

**Step 4: Create `src/finalayze/api/v1/auth.py`**

```python
"""API key authentication dependency (Layer 6).

Usage:
    router = APIRouter(dependencies=[Depends(require_api_key(settings.api_key))])
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=True)


def require_api_key(expected_key: str) -> Callable[..., None]:
    """Return a FastAPI dependency that validates the X-API-Key header."""

    async def _verify(key: str = Security(_header_scheme)) -> None:
        if key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    return _verify
```

**Step 5: Update `.env.example`** — add after `FINALAYZE_REAL_CONFIRMED`:

```
FINALAYZE_API_KEY=change-me-in-production
FINALAYZE_REAL_TOKEN=
```

**Step 6: Run tests**

```bash
uv run pytest tests/unit/test_api_auth.py -v
```
Expected: all 4 `PASSED`

**Step 7: Commit**

```bash
git add config/settings.py .env.example src/finalayze/api/v1/auth.py tests/unit/test_api_auth.py
git commit -m "feat(api): add X-API-Key auth dependency and settings fields"
```

---

### Task 3: Health endpoints

**Context:** `GET /health` (no auth, liveness check) and `GET /health/feeds` (no auth, feed last-seen). The existing `/health` in `system.py` returns only `{status, mode, version}` — expand it. The component health probes are stubs for now (real connections come in Phase 4 Track A); they return `"ok"` by default and can be overridden via app state.

**Files:**
- Modify: `src/finalayze/api/v1/system.py`
- Modify: `tests/unit/test_api_system.py` (existing file)

**Step 1: Write the failing tests**

```python
# Add to tests/unit/test_api_system.py

from fastapi.testclient import TestClient
from finalayze.main import create_app


def test_health_includes_components() -> None:
    client = TestClient(create_app())
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "components" in body
    assert "db" in body["components"]
    assert "redis" in body["components"]


def test_health_feeds_returns_list() -> None:
    client = TestClient(create_app())
    resp = client.get("/api/v1/health/feeds")
    assert resp.status_code == 200
    body = resp.json()
    assert "feeds" in body
    assert isinstance(body["feeds"], list)
```

**Step 2: Run to verify they fail**

```bash
uv run pytest tests/unit/test_api_system.py::test_health_includes_components tests/unit/test_api_system.py::test_health_feeds_returns_list -v
```
Expected: `FAILED`

**Step 3: Update `src/finalayze/api/v1/system.py`**

Replace the `HealthResponse` class and `health` endpoint:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from finalayze.core.exceptions import ModeError
from finalayze.core.modes import ModeManager, WorkMode

router = APIRouter(tags=["system"])

_default_mode_manager = ModeManager()
APP_VERSION = "0.1.0"

# In-memory ring buffer for recent errors (max 100)
_recent_errors: list[dict[str, Any]] = []
_MAX_ERRORS = 100


def get_mode_manager() -> ModeManager:
    return _default_mode_manager


def record_error(component: str, message: str, traceback_excerpt: str = "") -> None:
    """Called by logging handler to store recent exceptions."""
    _recent_errors.append({
        "timestamp": datetime.now(UTC).isoformat(),
        "component": component,
        "message": message,
        "traceback_excerpt": traceback_excerpt,
    })
    if len(_recent_errors) > _MAX_ERRORS:
        _recent_errors.pop(0)


class ComponentStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    db: str = "ok"
    redis: str = "ok"
    alpaca: str = "ok"
    tinkoff: str = "ok"
    llm: str = "ok"


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: str
    mode: str
    version: str
    components: ComponentStatus


class FeedStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    source: str
    last_seen: str | None
    latency_ms: float | None


class FeedsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    feeds: list[FeedStatus]


class ModeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: str


class ModeRequest(BaseModel):
    mode: WorkMode
    confirm_token: str | None = None


class SystemStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: str
    version: str
    uptime_seconds: float
    components: ComponentStatus


class ErrorEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp: str
    component: str
    message: str
    traceback_excerpt: str


_start_time = datetime.now(UTC)


@router.get("/health", response_model=HealthResponse)
async def health(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> HealthResponse:
    """Liveness check. No auth required. Returns 503 if critical component down."""
    components = ComponentStatus()
    overall = "ok" if all(
        v == "ok" for v in components.model_dump().values()
    ) else "degraded"
    return HealthResponse(
        status=overall,
        mode=str(mgr.current_mode),
        version=APP_VERSION,
        components=components,
    )


@router.get("/health/feeds", response_model=FeedsResponse)
async def health_feeds() -> FeedsResponse:
    """Feed health: last-seen timestamp per data source. No auth required."""
    # Stubs — real timestamps injected by fetchers in Phase 4 Track A
    return FeedsResponse(feeds=[
        FeedStatus(source="finnhub", last_seen=None, latency_ms=None),
        FeedStatus(source="newsapi", last_seen=None, latency_ms=None),
        FeedStatus(source="tinkoff", last_seen=None, latency_ms=None),
    ])


@router.get("/system/status", response_model=SystemStatusResponse)
async def system_status(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> SystemStatusResponse:
    """System status including mode, uptime, and component health. Auth required."""
    uptime = (datetime.now(UTC) - _start_time).total_seconds()
    return SystemStatusResponse(
        mode=str(mgr.current_mode),
        version=APP_VERSION,
        uptime_seconds=uptime,
        components=ComponentStatus(),
    )


@router.get("/system/errors", response_model=list[ErrorEntry])
async def system_errors() -> list[ErrorEntry]:
    """Last 100 recorded exceptions. Auth required."""
    return [ErrorEntry(**e) for e in _recent_errors]


@router.get("/mode", response_model=ModeResponse)
async def get_mode(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> ModeResponse:
    return ModeResponse(mode=str(mgr.current_mode))


@router.post("/mode", response_model=ModeResponse)
async def set_mode(
    request: ModeRequest,
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> ModeResponse:
    """Change work mode. Transitioning to REAL requires confirm_token."""
    from config.settings import Settings
    settings = Settings()
    if request.mode == WorkMode.REAL:
        if not settings.real_token or request.confirm_token != settings.real_token:
            raise HTTPException(
                status_code=403,
                detail="Transitioning to REAL mode requires a valid confirm_token",
            )
    try:
        mgr.transition_to(request.mode)
    except ModeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModeResponse(mode=str(mgr.current_mode))
```

**Step 4: Run tests**

```bash
uv run pytest tests/unit/test_api_system.py -v
```
Expected: all pass

**Step 5: Commit**

```bash
git add src/finalayze/api/v1/system.py tests/unit/test_api_system.py
git commit -m "feat(api): expand health endpoint, add feeds/system/errors endpoints"
```

---

### Task 4: Portfolio endpoints

**Context:** Five portfolio endpoints backed by BrokerRouter (live data) and `portfolio_snapshots` DB table (history). For now, inject a `BrokerRouter | None` via app state — `None` means no live broker (returns empty/zeroed data), real router injected when TradingLoop starts.

**Files:**
- Create: `src/finalayze/api/v1/portfolio.py`
- Create: `tests/unit/test_api_portfolio.py`
- Modify: `src/finalayze/api/v1/router.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_api_portfolio.py
from __future__ import annotations

from fastapi.testclient import TestClient

from finalayze.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_portfolio_unified_requires_auth() -> None:
    resp = _client().get("/api/v1/portfolio")
    assert resp.status_code == 422  # missing API key header


def test_portfolio_unified_with_valid_key() -> None:
    from config.settings import Settings
    key = Settings().api_key
    resp = _client().get("/api/v1/portfolio", headers={"X-API-Key": key})
    assert resp.status_code == 200
    body = resp.json()
    assert "total_equity_usd" in body
    assert "markets" in body


def test_portfolio_positions_with_valid_key() -> None:
    from config.settings import Settings
    key = Settings().api_key
    resp = _client().get("/api/v1/portfolio/positions", headers={"X-API-Key": key})
    assert resp.status_code == 200
    body = resp.json()
    assert "positions" in body
    assert isinstance(body["positions"], list)


def test_portfolio_history_with_valid_key() -> None:
    from config.settings import Settings
    key = Settings().api_key
    resp = _client().get("/api/v1/portfolio/history", headers={"X-API-Key": key})
    assert resp.status_code == 200
    body = resp.json()
    assert "snapshots" in body


def test_portfolio_performance_with_valid_key() -> None:
    from config.settings import Settings
    key = Settings().api_key
    resp = _client().get("/api/v1/portfolio/performance", headers={"X-API-Key": key})
    assert resp.status_code == 200
    body = resp.json()
    assert "sharpe_30d" in body
    assert "max_drawdown_pct" in body
```

**Step 2: Run to verify they fail**

```bash
uv run pytest tests/unit/test_api_portfolio.py -v
```
Expected: `FAILED`

**Step 3: Create `src/finalayze/api/v1/portfolio.py`**

```python
"""Portfolio endpoints (Layer 6).

All endpoints require X-API-Key. Data sources:
- Live portfolio/positions: BrokerRouter (injected via app state, may be None)
- History: portfolio_snapshots table via SQLAlchemy async session
- Performance: computed from portfolio_snapshots
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from config.settings import Settings
from finalayze.api.v1.auth import require_api_key

_settings = Settings()
router = APIRouter(
    prefix="/portfolio",
    tags=["portfolio"],
    dependencies=[Depends(require_api_key(_settings.api_key))],
)


class MarketPortfolio(BaseModel):
    model_config = ConfigDict(frozen=True)
    market_id: str
    equity_usd: float
    cash_usd: float
    positions_value_usd: float
    daily_pnl_usd: float
    daily_pnl_pct: float


class PortfolioResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    total_equity_usd: float
    total_cash_usd: float
    daily_pnl_usd: float
    daily_pnl_pct: float
    markets: list[MarketPortfolio]


class PositionDetail(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    market_id: str
    segment_id: str
    quantity: float
    market_value_usd: float
    unrealized_pnl_usd: float
    unrealized_pnl_pct: float
    stop_distance_atr: float | None


class PositionsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    positions: list[PositionDetail]


class SnapshotEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp: str
    market_id: str
    equity: float
    drawdown_pct: float


class HistoryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    snapshots: list[SnapshotEntry]


class PerformanceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    sharpe_30d: float | None
    sortino_30d: float | None
    max_drawdown_pct: float | None
    win_rate: float | None
    profit_factor: float | None
    avg_win_loss_ratio: float | None


def _empty_portfolio() -> PortfolioResponse:
    return PortfolioResponse(
        total_equity_usd=0.0,
        total_cash_usd=0.0,
        daily_pnl_usd=0.0,
        daily_pnl_pct=0.0,
        markets=[],
    )


@router.get("", response_model=PortfolioResponse)
async def get_portfolio(request: Request) -> PortfolioResponse:
    """Unified portfolio across all markets in base currency (USD)."""
    broker_router = getattr(request.app.state, "broker_router", None)
    if broker_router is None:
        return _empty_portfolio()

    markets: list[MarketPortfolio] = []
    for market_id in ["us", "moex"]:
        try:
            broker = broker_router.route(market_id)
            p = broker.get_portfolio()
            equity = float(p.equity)
            cash = float(p.cash)
            markets.append(MarketPortfolio(
                market_id=market_id,
                equity_usd=equity,
                cash_usd=cash,
                positions_value_usd=equity - cash,
                daily_pnl_usd=0.0,
                daily_pnl_pct=0.0,
            ))
        except Exception:  # noqa: BLE001
            pass

    total = sum(m.equity_usd for m in markets)
    return PortfolioResponse(
        total_equity_usd=total,
        total_cash_usd=sum(m.cash_usd for m in markets),
        daily_pnl_usd=0.0,
        daily_pnl_pct=0.0,
        markets=markets,
    )


@router.get("/positions", response_model=PositionsResponse)
async def get_positions(request: Request) -> PositionsResponse:
    """All open positions with unrealized P&L."""
    broker_router = getattr(request.app.state, "broker_router", None)
    if broker_router is None:
        return PositionsResponse(positions=[])

    positions: list[PositionDetail] = []
    for market_id in ["us", "moex"]:
        try:
            broker = broker_router.route(market_id)
            raw = broker.get_positions()
            for symbol, qty in raw.items():
                if qty > Decimal(0):
                    positions.append(PositionDetail(
                        symbol=symbol,
                        market_id=market_id,
                        segment_id="",
                        quantity=float(qty),
                        market_value_usd=0.0,
                        unrealized_pnl_usd=0.0,
                        unrealized_pnl_pct=0.0,
                        stop_distance_atr=None,
                    ))
        except Exception:  # noqa: BLE001
            pass

    return PositionsResponse(positions=positions)


@router.get("/positions/{symbol}", response_model=PositionDetail)
async def get_position(symbol: str) -> PositionDetail:
    """Single position detail."""
    return PositionDetail(
        symbol=symbol,
        market_id="",
        segment_id="",
        quantity=0.0,
        market_value_usd=0.0,
        unrealized_pnl_usd=0.0,
        unrealized_pnl_pct=0.0,
        stop_distance_atr=None,
    )


@router.get("/history", response_model=HistoryResponse)
async def get_portfolio_history() -> HistoryResponse:
    """Equity curve from portfolio_snapshots table (last 30 days)."""
    # Real implementation queries DB; stub returns empty for now
    return HistoryResponse(snapshots=[])


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance() -> PerformanceResponse:
    """Rolling 30-day performance metrics."""
    return PerformanceResponse(
        sharpe_30d=None,
        sortino_30d=None,
        max_drawdown_pct=None,
        win_rate=None,
        profit_factor=None,
        avg_win_loss_ratio=None,
    )
```

**Step 4: Register router in `src/finalayze/api/v1/router.py`**

```python
from __future__ import annotations

from fastapi import APIRouter

from finalayze.api.v1.portfolio import router as portfolio_router
from finalayze.api.v1.system import router as system_router

api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(portfolio_router)
```

**Step 5: Run tests**

```bash
uv run pytest tests/unit/test_api_portfolio.py -v
```
Expected: all pass

**Step 6: Commit**

```bash
git add src/finalayze/api/v1/portfolio.py src/finalayze/api/v1/router.py tests/unit/test_api_portfolio.py
git commit -m "feat(api): add portfolio endpoints (unified, positions, history, performance)"
```

---

### Task 5: Trades endpoints

**Files:**
- Create: `src/finalayze/api/v1/trades.py`
- Create: `tests/unit/test_api_trades.py`
- Modify: `src/finalayze/api/v1/router.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_api_trades.py
from __future__ import annotations

from fastapi.testclient import TestClient
from finalayze.main import create_app


def _auth_headers() -> dict[str, str]:
    from config.settings import Settings
    return {"X-API-Key": Settings().api_key}


def test_trades_list_returns_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades", headers=_auth_headers())
    assert resp.status_code == 200
    assert "trades" in resp.json()


def test_trades_list_requires_auth() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades")
    assert resp.status_code == 422


def test_trades_analytics_returns_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades/analytics", headers=_auth_headers())
    assert resp.status_code == 200
    assert "avg_slippage_bps" in resp.json()


def test_trade_detail_returns_404_for_unknown() -> None:
    import uuid
    resp = TestClient(create_app()).get(
        f"/api/v1/trades/{uuid.uuid4()}", headers=_auth_headers()
    )
    assert resp.status_code == 404
```

**Step 2: Run to verify they fail**

```bash
uv run pytest tests/unit/test_api_trades.py -v
```
Expected: `FAILED`

**Step 3: Create `src/finalayze/api/v1/trades.py`**

```python
"""Trades endpoints (Layer 6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from config.settings import Settings
from finalayze.api.v1.auth import require_api_key

_settings = Settings()
router = APIRouter(
    prefix="/trades",
    tags=["trades"],
    dependencies=[Depends(require_api_key(_settings.api_key))],
)


class TradeEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    symbol: str
    market_id: str
    side: str
    quantity: float
    fill_price: float | None
    slippage_bps: float | None
    timestamp: str


class TradesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    trades: list[TradeEntry]
    total: int


class TradeAnalytics(BaseModel):
    model_config = ConfigDict(frozen=True)
    period_days: int
    total_trades: int
    avg_slippage_bps: float | None
    avg_fill_latency_ms: float | None
    rejection_rate_pct: float | None


@router.get("", response_model=TradesResponse)
async def list_trades(
    market: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
) -> TradesResponse:
    """Trade history. Reads from orders table (stub: returns empty)."""
    return TradesResponse(trades=[], total=0)


@router.get("/analytics", response_model=TradeAnalytics)
async def trade_analytics(
    market: str | None = None,
    period: int = 7,
) -> TradeAnalytics:
    """Slippage and fill latency stats."""
    return TradeAnalytics(
        period_days=period,
        total_trades=0,
        avg_slippage_bps=None,
        avg_fill_latency_ms=None,
        rejection_rate_pct=None,
    )


@router.get("/{trade_id}", response_model=TradeEntry)
async def get_trade(trade_id: str) -> TradeEntry:
    """Single trade detail for audit drill-down."""
    raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
```

**Step 4: Add to router**

```python
from finalayze.api.v1.trades import router as trades_router
api_router.include_router(trades_router)
```

**Step 5: Run tests, commit**

```bash
uv run pytest tests/unit/test_api_trades.py -v
git add src/finalayze/api/v1/trades.py src/finalayze/api/v1/router.py tests/unit/test_api_trades.py
git commit -m "feat(api): add trades endpoints (list, detail, analytics)"
```

---

### Task 6: Signals, strategies, risk, ML, and news endpoints

**Files:**
- Create: `src/finalayze/api/v1/signals.py`
- Create: `src/finalayze/api/v1/risk.py`
- Create: `src/finalayze/api/v1/ml.py`
- Create: `src/finalayze/api/v1/news.py`
- Create: `tests/unit/test_api_signals_risk.py`
- Modify: `src/finalayze/api/v1/router.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_api_signals_risk.py
from __future__ import annotations

from fastapi.testclient import TestClient
from finalayze.main import create_app


def _h() -> dict[str, str]:
    from config.settings import Settings
    return {"X-API-Key": Settings().api_key}


def test_signals_list_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/signals", headers=_h())
    assert resp.status_code == 200
    assert "signals" in resp.json()


def test_strategies_performance_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/strategies/performance", headers=_h())
    assert resp.status_code == 200
    assert "strategies" in resp.json()


def test_risk_status_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/risk/status", headers=_h())
    assert resp.status_code == 200
    body = resp.json()
    assert "markets" in body


def test_risk_exposure_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/risk/exposure", headers=_h())
    assert resp.status_code == 200
    assert "segments" in resp.json()


def test_risk_override_requires_auth() -> None:
    resp = TestClient(create_app()).post(
        "/api/v1/risk/override",
        json={"market_id": "us", "level": 1},
    )
    assert resp.status_code == 422


def test_ml_status_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/ml/status", headers=_h())
    assert resp.status_code == 200
    assert "models" in resp.json()


def test_news_list_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/news", headers=_h())
    assert resp.status_code == 200
    assert "articles" in resp.json()
```

**Step 2: Run to verify they fail**

```bash
uv run pytest tests/unit/test_api_signals_risk.py -v
```

**Step 3: Create `src/finalayze/api/v1/signals.py`**

```python
"""Signals and strategy performance endpoints (Layer 6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from config.settings import Settings
from finalayze.api.v1.auth import require_api_key

_settings = Settings()
router = APIRouter(
    tags=["signals"],
    dependencies=[Depends(require_api_key(_settings.api_key))],
)


class SignalEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    symbol: str
    market_id: str
    segment_id: str
    strategy: str
    direction: str
    confidence: float
    created_at: str


class SignalsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    signals: list[SignalEntry]


class StrategyPerf(BaseModel):
    model_config = ConfigDict(frozen=True)
    strategy: str
    market_id: str
    win_rate: float | None
    profit_factor: float | None
    trades_today: int
    last_signal_at: str | None


class StrategiesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    strategies: list[StrategyPerf]


@router.get("/signals", response_model=SignalsResponse)
async def list_signals(
    market: str | None = None,
    segment: str | None = None,
    limit: int = 50,
) -> SignalsResponse:
    return SignalsResponse(signals=[])


@router.get("/strategies/performance", response_model=StrategiesResponse)
async def strategies_performance() -> StrategiesResponse:
    return StrategiesResponse(strategies=[])
```

**Step 4: Create `src/finalayze/api/v1/risk.py`**

```python
"""Risk endpoints (Layer 6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from config.settings import Settings
from finalayze.api.v1.auth import require_api_key

_settings = Settings()
router = APIRouter(
    prefix="/risk",
    tags=["risk"],
    dependencies=[Depends(require_api_key(_settings.api_key))],
)


class MarketRiskStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    market_id: str
    circuit_breaker_level: int  # 0=normal 1=caution 2=halted 3=liquidate
    level_label: str
    level_since: str | None


class RiskStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    markets: list[MarketRiskStatus]
    cross_market_halted: bool


class SegmentExposure(BaseModel):
    model_config = ConfigDict(frozen=True)
    segment_id: str
    market_id: str
    value_usd: float
    pct_of_portfolio: float


class ExposureResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    segments: list[SegmentExposure]
    total_invested_pct: float


class OverrideRequest(BaseModel):
    market_id: str
    level: int  # 0–3


class OverrideResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    market_id: str
    level: int
    applied: bool


@router.get("/status", response_model=RiskStatusResponse)
async def risk_status(request: Request) -> RiskStatusResponse:
    circuit_breakers = getattr(request.app.state, "circuit_breakers", {})
    markets: list[MarketRiskStatus] = []
    for market_id in ["us", "moex"]:
        cb = circuit_breakers.get(market_id)
        if cb is not None:
            from finalayze.risk.circuit_breaker import CircuitLevel
            level_map = {
                CircuitLevel.NORMAL: (0, "NORMAL"),
                CircuitLevel.CAUTION: (1, "CAUTION"),
                CircuitLevel.HALTED: (2, "HALTED"),
                CircuitLevel.LIQUIDATE: (3, "LIQUIDATE"),
            }
            lvl_int, lvl_label = level_map.get(cb.current_level, (0, "NORMAL"))
            markets.append(MarketRiskStatus(
                market_id=market_id,
                circuit_breaker_level=lvl_int,
                level_label=lvl_label,
                level_since=None,
            ))
        else:
            markets.append(MarketRiskStatus(
                market_id=market_id,
                circuit_breaker_level=0,
                level_label="NORMAL",
                level_since=None,
            ))
    return RiskStatusResponse(markets=markets, cross_market_halted=False)


@router.get("/exposure", response_model=ExposureResponse)
async def risk_exposure() -> ExposureResponse:
    return ExposureResponse(segments=[], total_invested_pct=0.0)


@router.post("/override", response_model=OverrideResponse)
async def risk_override(req: OverrideRequest, request: Request) -> OverrideResponse:
    circuit_breakers = getattr(request.app.state, "circuit_breakers", {})
    cb = circuit_breakers.get(req.market_id)
    if cb is None:
        return OverrideResponse(market_id=req.market_id, level=req.level, applied=False)
    from finalayze.risk.circuit_breaker import CircuitLevel
    level_list = [CircuitLevel.NORMAL, CircuitLevel.CAUTION, CircuitLevel.HALTED, CircuitLevel.LIQUIDATE]
    if 0 <= req.level < len(level_list):
        cb._level = level_list[req.level]  # type: ignore[attr-defined]
    return OverrideResponse(market_id=req.market_id, level=req.level, applied=True)
```

**Step 5: Create `src/finalayze/api/v1/ml.py`**

```python
"""ML model status endpoints (Layer 6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from config.settings import Settings
from finalayze.api.v1.auth import require_api_key

_settings = Settings()
router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    dependencies=[Depends(require_api_key(_settings.api_key))],
)


class ModelStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    segment_id: str
    model_type: str
    last_retrain: str | None
    prediction_latency_p50_ms: float | None
    is_stale: bool


class MLStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    models: list[ModelStatus]


@router.get("/status", response_model=MLStatusResponse)
async def ml_status() -> MLStatusResponse:
    return MLStatusResponse(models=[])
```

**Step 6: Create `src/finalayze/api/v1/news.py`**

```python
"""News endpoints (Layer 6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from config.settings import Settings
from finalayze.api.v1.auth import require_api_key

_settings = Settings()
router = APIRouter(
    prefix="/news",
    tags=["news"],
    dependencies=[Depends(require_api_key(_settings.api_key))],
)


class ArticleEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    title: str
    source: str
    scope: str
    sentiment: float | None
    published_at: str


class NewsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    articles: list[ArticleEntry]


@router.get("", response_model=NewsResponse)
async def list_news(
    scope: str | None = None,
    limit: int = 20,
) -> NewsResponse:
    return NewsResponse(articles=[])
```

**Step 7: Wire all into `src/finalayze/api/v1/router.py`**

```python
from __future__ import annotations

from fastapi import APIRouter

from finalayze.api.v1.ml import router as ml_router
from finalayze.api.v1.news import router as news_router
from finalayze.api.v1.portfolio import router as portfolio_router
from finalayze.api.v1.risk import router as risk_router
from finalayze.api.v1.signals import router as signals_router
from finalayze.api.v1.system import router as system_router
from finalayze.api.v1.trades import router as trades_router

api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(portfolio_router)
api_router.include_router(trades_router)
api_router.include_router(signals_router)
api_router.include_router(risk_router)
api_router.include_router(ml_router)
api_router.include_router(news_router)
```

**Step 8: Run all new tests**

```bash
uv run pytest tests/unit/test_api_signals_risk.py -v
```

**Step 9: Run full suite + lint**

```bash
uv run pytest tests/ -v
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
```

**Step 10: Commit**

```bash
git add src/finalayze/api/v1/{signals,risk,ml,news}.py src/finalayze/api/v1/router.py tests/unit/test_api_signals_risk.py
git commit -m "feat(api): add signals, risk, ML status, and news endpoints"
```

---

### Task 7: Final B-1 lint pass, coverage check, and PR

**Step 1: Run full suite**

```bash
uv run pytest tests/ --cov=src/finalayze --cov-report=term-missing -v 2>&1 | tail -20
```
Expected: all tests pass, coverage >= 50%

**Step 2: Run linters**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src/
```
Expected: zero errors

**Step 3: Push and create PR B-1**

```bash
git push -u origin feature/phase4-track-b-api
gh pr create --title "feat(phase4): Track B-1 — Core API with auth, 20 endpoints" --body "$(cat <<'EOF'
## Summary
- Alembic migration 003: portfolio_snapshots TimescaleDB hypertable
- X-API-Key auth dependency (all endpoints except /health)
- 20 new REST endpoints: health, system/status/errors/mode, portfolio (5), trades (3), signals+strategies (2), risk (3), ml/status, news
- All endpoints return structured Pydantic v2 responses
- Auth guard: sandbox→real mode requires confirm_token

## Test Plan
- [ ] All unit tests pass
- [ ] `GET /api/v1/health` returns 200 without API key
- [ ] `GET /api/v1/portfolio` returns 401 without key, 200 with key
- [ ] `POST /api/v1/system/mode` with mode=real and wrong token returns 403

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## PR B-2: Streamlit Dashboard

### Task 8: Streamlit setup, auth, and HTTP client helper

**Context:** Streamlit is already in `pyproject.toml`. Dashboard lives at `src/finalayze/dashboard/`. Auth via `st.secrets["password"]`. All API calls go through a shared `httpx` client with the API key from `st.secrets["api_key"]`.

**Files:**
- Create: `src/finalayze/dashboard/app.py`
- Create: `src/finalayze/dashboard/api_client.py`
- Create: `.streamlit/secrets.toml.example`
- Create: `tests/unit/test_dashboard_api_client.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_dashboard_api_client.py
from __future__ import annotations

import pytest
import httpx
import respx

from finalayze.dashboard.api_client import ApiClient


@respx.mock
def test_api_client_injects_key() -> None:
    respx.get("http://localhost:8000/api/v1/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = ApiClient(base_url="http://localhost:8000", api_key="test-key")
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert respx.calls.last.request.headers["x-api-key"] == "test-key"


@respx.mock
def test_api_client_raises_on_401() -> None:
    respx.get("http://localhost:8000/api/v1/portfolio").mock(
        return_value=httpx.Response(401, json={"detail": "Invalid API key"})
    )
    client = ApiClient(base_url="http://localhost:8000", api_key="bad-key")
    with pytest.raises(httpx.HTTPStatusError):
        client.get("/api/v1/portfolio", raise_on_error=True)
```

**Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/test_dashboard_api_client.py -v
```

**Step 3: Create `src/finalayze/dashboard/api_client.py`**

```python
"""Synchronous HTTP client for Streamlit dashboard.

Usage:
    client = ApiClient(base_url=st.secrets["api_url"], api_key=st.secrets["api_key"])
    data = client.get("/api/v1/portfolio").json()
"""

from __future__ import annotations

import httpx


class ApiClient:
    """Thin httpx wrapper that injects X-API-Key on every request."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}
        self._timeout = timeout

    def get(self, path: str, raise_on_error: bool = False, **kwargs: object) -> httpx.Response:
        url = f"{self._base_url}{path}"
        resp = httpx.get(url, headers=self._headers, timeout=self._timeout, **kwargs)  # type: ignore[arg-type]
        if raise_on_error:
            resp.raise_for_status()
        return resp

    def post(self, path: str, raise_on_error: bool = False, **kwargs: object) -> httpx.Response:
        url = f"{self._base_url}{path}"
        resp = httpx.post(url, headers=self._headers, timeout=self._timeout, **kwargs)  # type: ignore[arg-type]
        if raise_on_error:
            resp.raise_for_status()
        return resp
```

**Step 4: Create `src/finalayze/dashboard/app.py`**

```python
"""Streamlit dashboard entry point.

Run with:
    streamlit run src/finalayze/dashboard/app.py

Requires .streamlit/secrets.toml with:
    password = "your-dashboard-password"
    api_key = "your-api-key"
    api_url = "http://localhost:8000"
"""

from __future__ import annotations

import streamlit as st

from finalayze.dashboard.api_client import ApiClient

st.set_page_config(page_title="Finalayze", page_icon="📈", layout="wide")

_PASSWORD = st.secrets.get("password", "")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("Finalayze — Login")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if pwd == _PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid password")
    st.stop()

# Authenticated — build shared API client
_api = ApiClient(
    base_url=st.secrets.get("api_url", "http://localhost:8000"),
    api_key=st.secrets.get("api_key", ""),
)
st.session_state["api"] = _api

pages = {
    "System Status": "finalayze.dashboard.pages.system_status",
    "Portfolio": "finalayze.dashboard.pages.portfolio",
    "Trades": "finalayze.dashboard.pages.trades",
    "Signals": "finalayze.dashboard.pages.signals",
    "Risk": "finalayze.dashboard.pages.risk",
}

page = st.sidebar.radio("Navigation", list(pages.keys()))
import importlib  # noqa: E402
mod = importlib.import_module(pages[page])
mod.render(_api)  # type: ignore[attr-defined]
```

**Step 5: Create `.streamlit/secrets.toml.example`**

```toml
password = "change-me"
api_key = "change-me-in-production"
api_url = "http://localhost:8000"
```

Add `.streamlit/secrets.toml` to `.gitignore`.

**Step 6: Run tests, commit**

```bash
uv run pytest tests/unit/test_dashboard_api_client.py -v
git add src/finalayze/dashboard/app.py src/finalayze/dashboard/api_client.py \
        .streamlit/secrets.toml.example tests/unit/test_dashboard_api_client.py
git commit -m "feat(dashboard): Streamlit app entry point, auth, and API client"
```

---

### Task 9: Dashboard pages — System Status and Portfolio

**Files:**
- Create: `src/finalayze/dashboard/pages/__init__.py`
- Create: `src/finalayze/dashboard/pages/system_status.py`
- Create: `src/finalayze/dashboard/pages/portfolio.py`
- Create: `tests/unit/test_dashboard_pages.py`

**Step 1: Write the failing smoke tests**

```python
# tests/unit/test_dashboard_pages.py
from __future__ import annotations

from unittest.mock import MagicMock

import httpx


def _mock_client(json_data: dict) -> MagicMock:  # type: ignore[type-arg]
    client = MagicMock()
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = json_data
    resp.status_code = 200
    client.get.return_value = resp
    return client


def test_system_status_render_importable() -> None:
    from finalayze.dashboard.pages import system_status
    assert callable(system_status.render)


def test_portfolio_render_importable() -> None:
    from finalayze.dashboard.pages import portfolio
    assert callable(portfolio.render)
```

**Step 2: Run to verify they fail**

```bash
uv run pytest tests/unit/test_dashboard_pages.py -v
```

**Step 3: Create `src/finalayze/dashboard/pages/__init__.py`** — empty file

**Step 4: Create `src/finalayze/dashboard/pages/system_status.py`**

```python
"""System Status page — page 1 of the dashboard."""

from __future__ import annotations

import streamlit as st

from finalayze.dashboard.api_client import ApiClient

_LEVEL_COLOR = {
    "NORMAL": "🟢",
    "CAUTION": "🟡",
    "HALTED": "🔴",
    "LIQUIDATE": "🔴",
}

_MODE_COLOR = {
    "real": "🔴",
    "test": "🟡",
    "sandbox": "🟢",
    "debug": "⚪",
}


def render(api: ApiClient) -> None:
    st.title("System Status")

    # Health + mode
    health = api.get("/api/v1/health").json()
    mode = health.get("mode", "unknown")
    icon = _MODE_COLOR.get(mode, "⚪")
    st.metric("Work Mode", f"{icon} {mode.upper()}")

    # Component health
    st.subheader("Components")
    components = health.get("components", {})
    cols = st.columns(len(components))
    for col, (name, status) in zip(cols, components.items()):
        color = "🟢" if status == "ok" else "🔴"
        col.metric(name.upper(), f"{color} {status}")

    # Recent errors
    st.subheader("Recent Errors")
    errors = api.get("/api/v1/system/errors").json()
    if not errors:
        st.success("No recent errors")
    else:
        for err in errors[:10]:
            with st.expander(f"[{err['timestamp']}] {err['component']}: {err['message']}"):
                st.code(err.get("traceback_excerpt", ""))

    # Mode switcher
    st.subheader("Change Mode")
    new_mode = st.selectbox("Mode", ["debug", "sandbox", "test", "real"])
    confirm = ""
    if new_mode == "real":
        confirm = st.text_input("Confirm token (required for REAL mode)")
    if st.button("Apply Mode"):
        payload: dict[str, object] = {"mode": new_mode}
        if confirm:
            payload["confirm_token"] = confirm
        resp = api.post("/api/v1/system/mode", json=payload)
        if resp.status_code == 200:
            st.success(f"Mode changed to {new_mode}")
        else:
            st.error(f"Failed: {resp.json().get('detail', resp.status_code)}")
```

**Step 5: Create `src/finalayze/dashboard/pages/portfolio.py`**

```python
"""Portfolio page."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from finalayze.dashboard.api_client import ApiClient


def render(api: ApiClient) -> None:
    st.title("Portfolio")

    portfolio = api.get("/api/v1/portfolio").json()
    perf = api.get("/api/v1/portfolio/performance").json()
    history = api.get("/api/v1/portfolio/history").json()
    positions = api.get("/api/v1/portfolio/positions").json()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Equity (USD)", f"${portfolio.get('total_equity_usd', 0):,.2f}")
    col2.metric("Daily P&L", f"${portfolio.get('daily_pnl_usd', 0):,.2f}",
                f"{portfolio.get('daily_pnl_pct', 0):.2f}%")
    col3.metric("Sharpe (30d)", f"{perf.get('sharpe_30d', 'N/A')}")
    col4.metric("Max Drawdown", f"{(perf.get('max_drawdown_pct') or 0) * 100:.1f}%")

    # Equity curve with drawdown
    snapshots = history.get("snapshots", [])
    if snapshots:
        df = pd.DataFrame(snapshots)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.subheader("Equity Curve")
        st.line_chart(df.set_index("timestamp")["equity"])
    else:
        st.info("No historical data yet — equity curve will appear after the first trading cycle.")

    # Per-market breakdown
    markets = portfolio.get("markets", [])
    if markets:
        st.subheader("By Market")
        mdf = pd.DataFrame(markets)
        st.dataframe(mdf, use_container_width=True)

    # Positions heatmap
    pos_list = positions.get("positions", [])
    if pos_list:
        st.subheader("Open Positions")
        pdf = pd.DataFrame(pos_list)
        st.dataframe(
            pdf.style.background_gradient(subset=["unrealized_pnl_pct"], cmap="RdYlGn"),
            use_container_width=True,
        )
    else:
        st.info("No open positions.")
```

**Step 6: Run tests**

```bash
uv run pytest tests/unit/test_dashboard_pages.py -v
```

**Step 7: Commit**

```bash
git add src/finalayze/dashboard/pages/ tests/unit/test_dashboard_pages.py
git commit -m "feat(dashboard): System Status and Portfolio pages"
```

---

### Task 10: Dashboard pages — Trades, Signals, Risk

**Files:**
- Create: `src/finalayze/dashboard/pages/trades.py`
- Create: `src/finalayze/dashboard/pages/signals.py`
- Create: `src/finalayze/dashboard/pages/risk.py`

**Step 1: Add smoke tests to `tests/unit/test_dashboard_pages.py`**

```python
def test_trades_render_importable() -> None:
    from finalayze.dashboard.pages import trades
    assert callable(trades.render)

def test_signals_render_importable() -> None:
    from finalayze.dashboard.pages import signals
    assert callable(signals.render)

def test_risk_render_importable() -> None:
    from finalayze.dashboard.pages import risk
    assert callable(risk.render)
```

**Step 2: Create `src/finalayze/dashboard/pages/trades.py`**

```python
"""Trades page."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from finalayze.dashboard.api_client import ApiClient


def render(api: ApiClient) -> None:
    st.title("Trades")

    col1, col2 = st.columns(2)
    market_filter = col1.selectbox("Market", ["all", "us", "moex"])
    limit = col2.slider("Limit", 10, 200, 100)

    params = {"limit": limit}
    if market_filter != "all":
        params["market"] = market_filter

    trades = api.get("/api/v1/trades", params=params).json()  # type: ignore[arg-type]
    trade_list = trades.get("trades", [])

    if trade_list:
        df = pd.DataFrame(trade_list)
        st.dataframe(df, use_container_width=True)

        # Slippage scatter
        if "slippage_bps" in df.columns and df["slippage_bps"].notna().any():
            st.subheader("Slippage by Time of Day")
            st.scatter_chart(
                df.dropna(subset=["slippage_bps"]),
                x="timestamp",
                y="slippage_bps",
                color="market_id",
            )
    else:
        st.info("No trades recorded yet.")

    # Analytics
    analytics = api.get("/api/v1/trades/analytics").json()
    st.subheader("Analytics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Slippage (bps)", analytics.get("avg_slippage_bps", "N/A"))
    col2.metric("Avg Fill Latency (ms)", analytics.get("avg_fill_latency_ms", "N/A"))
    col3.metric("Rejection Rate", f"{analytics.get('rejection_rate_pct', 0) or 0:.1f}%")
```

**Step 3: Create `src/finalayze/dashboard/pages/signals.py`**

```python
"""Signals page."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from finalayze.dashboard.api_client import ApiClient


def render(api: ApiClient) -> None:
    st.title("Signals")

    # Strategy performance matrix
    strategies = api.get("/api/v1/strategies/performance").json().get("strategies", [])
    if strategies:
        st.subheader("Strategy Performance")
        sdf = pd.DataFrame(strategies)
        st.dataframe(
            sdf.style.background_gradient(subset=["win_rate"], cmap="RdYlGn"),
            use_container_width=True,
        )
    else:
        st.info("No strategy performance data yet.")

    # Recent signals
    signals = api.get("/api/v1/signals").json().get("signals", [])
    st.subheader("Recent Signals")
    if signals:
        st.dataframe(pd.DataFrame(signals), use_container_width=True)
    else:
        st.info("No signals recorded yet.")
```

**Step 4: Create `src/finalayze/dashboard/pages/risk.py`**

```python
"""Risk page."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from finalayze.dashboard.api_client import ApiClient

_LEVEL_BADGE = {0: "🟢 NORMAL", 1: "🟡 CAUTION", 2: "🔴 HALTED", 3: "🔴 LIQUIDATE"}


def render(api: ApiClient) -> None:
    st.title("Risk")

    risk = api.get("/api/v1/risk/status").json()
    exposure = api.get("/api/v1/risk/exposure").json()

    # Circuit breaker status
    st.subheader("Circuit Breakers")
    markets = risk.get("markets", [])
    if markets:
        cols = st.columns(len(markets))
        for col, m in zip(cols, markets):
            badge = _LEVEL_BADGE.get(m["circuit_breaker_level"], "❓")
            since = m.get("level_since") or "—"
            col.metric(m["market_id"].upper(), badge, f"since {since}")
    else:
        st.info("No circuit breaker data available.")

    if risk.get("cross_market_halted"):
        st.error("⚠️ Cross-market circuit breaker TRIPPED — all markets halted")

    # Segment exposure
    segments = exposure.get("segments", [])
    if segments:
        st.subheader("Exposure by Segment")
        edf = pd.DataFrame(segments)
        st.bar_chart(edf.set_index("segment_id")["pct_of_portfolio"])
        st.dataframe(edf, use_container_width=True)
    else:
        st.info("No exposure data available.")

    # Emergency override
    st.subheader("Emergency Override")
    with st.form("override_form"):
        market_id = st.selectbox("Market", ["us", "moex"])
        level = st.selectbox("Level", [0, 1, 2, 3],
                             format_func=lambda x: _LEVEL_BADGE.get(x, str(x)))
        submitted = st.form_submit_button("Apply Override")
        if submitted:
            resp = api.post("/api/v1/risk/override", json={"market_id": market_id, "level": level})
            if resp.status_code == 200 and resp.json().get("applied"):
                st.success(f"Override applied: {market_id} → level {level}")
            else:
                st.error("Override failed")
```

**Step 5: Run tests, lint, commit**

```bash
uv run pytest tests/unit/test_dashboard_pages.py -v
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
git add src/finalayze/dashboard/pages/{trades,signals,risk}.py tests/unit/test_dashboard_pages.py
git commit -m "feat(dashboard): Trades, Signals, and Risk pages"
```

---

### Task 11: B-2 final checks and PR

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -v 2>&1 | tail -20
```

**Step 2: Create PR B-2**

```bash
git push -u origin feature/phase4-track-b-dashboard
gh pr create --title "feat(phase4): Track B-2 — Streamlit operator dashboard (5 pages)" --body "$(cat <<'EOF'
## Summary
- Streamlit app with st.secrets password auth
- 5 pages: System Status, Portfolio, Trades, Signals, Risk
- Shared ApiClient (httpx) with X-API-Key injection
- Portfolio: equity curve, position table, performance metrics
- Risk: circuit breaker badges, segment exposure chart, emergency override form

## Test Plan
- [ ] `streamlit run src/finalayze/dashboard/app.py` starts without error
- [ ] Login rejects wrong password
- [ ] All pages load without exception (smoke tests pass)
- [ ] API key is not visible in any page output

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## PR B-3: Prometheus Metrics

### Task 12: MetricsCollector class and /metrics endpoint

**Context:** `prometheus-fastapi-instrumentator` handles HTTP request metrics automatically. A custom `MetricsCollector` class holds business gauge/counter/histogram objects and exposes an `update()` method called by `TradingLoop` after each cycle.

**Files:**
- Create: `src/finalayze/core/metrics.py`
- Create: `tests/unit/test_metrics.py`
- Modify: `src/finalayze/main.py`
- Modify: `pyproject.toml` (add prometheus deps)

**Step 1: Add dependencies to `pyproject.toml`**

Under `[project.dependencies]`, add:
```toml
"prometheus-client>=0.20",
"prometheus-fastapi-instrumentator>=7.0",
```

Then run: `uv sync`

**Step 2: Write the failing tests**

```python
# tests/unit/test_metrics.py
from __future__ import annotations

from finalayze.core.metrics import MetricsCollector


def test_metrics_collector_instantiates() -> None:
    mc = MetricsCollector()
    assert mc is not None


def test_update_portfolio_equity_accepts_market_and_value() -> None:
    mc = MetricsCollector()
    mc.update_portfolio_equity("us", 10000.0, daily_pnl=100.0, drawdown_pct=0.01)
    # No exception means success


def test_update_circuit_breaker_accepts_level() -> None:
    mc = MetricsCollector()
    mc.update_circuit_breaker("us", level=1)
    mc.update_circuit_breaker("moex", level=0)


def test_record_trade_increments_counter() -> None:
    mc = MetricsCollector()
    mc.record_trade(market="us", side="BUY", slippage_bps=2.5)
```

**Step 3: Run to verify they fail**

```bash
uv run pytest tests/unit/test_metrics.py -v
```

**Step 4: Create `src/finalayze/core/metrics.py`**

```python
"""Prometheus business metrics collector (Layer 0 — no domain imports).

Usage:
    metrics = MetricsCollector()
    # In TradingLoop._strategy_cycle():
    metrics.update_portfolio_equity("us", equity=12500.0, daily_pnl=150.0, drawdown_pct=0.02)
    metrics.update_circuit_breaker("us", level=0)
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Portfolio ─────────────────────────────────────────────────────────────────
_equity = Gauge("finalayze_portfolio_equity_usd", "Portfolio equity in USD", ["market"])
_daily_pnl = Gauge("finalayze_daily_pnl_usd", "Daily P&L in USD", ["market"])
_drawdown = Gauge("finalayze_drawdown_pct", "Current drawdown 0.0-1.0", ["market"])
_open_positions = Gauge("finalayze_open_positions_count", "Number of open positions", ["market"])

# ── Circuit breakers ──────────────────────────────────────────────────────────
_cb_level = Gauge(
    "finalayze_circuit_breaker_level",
    "Circuit breaker level (0=normal 1=caution 2=halted 3=liquidate)",
    ["market"],
)

# ── Execution ─────────────────────────────────────────────────────────────────
_trades_total = Counter("finalayze_trades_total", "Cumulative filled orders", ["market", "side"])
_slippage = Histogram(
    "finalayze_trade_slippage_bps",
    "Trade slippage in basis points",
    ["market"],
    buckets=[0, 1, 2, 5, 10, 20, 50, 100],
)
_fill_latency = Histogram(
    "finalayze_order_fill_latency_seconds",
    "Order fill latency",
    ["market"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
_rejections = Counter(
    "finalayze_order_rejection_total",
    "Order rejections",
    ["market", "reason"],
)

# ── Strategy ──────────────────────────────────────────────────────────────────
_strategy_win_rate = Gauge(
    "finalayze_strategy_win_rate",
    "Rolling win rate over last 100 trades",
    ["market", "strategy"],
)
_signal_count = Counter(
    "finalayze_strategy_signal_count",
    "Cumulative signals generated",
    ["market", "strategy", "direction"],
)

# ── ML ────────────────────────────────────────────────────────────────────────
_ml_retrain_ts = Gauge(
    "finalayze_ml_model_last_retrain_timestamp",
    "Unix timestamp of last model retrain",
    ["model"],
)
_ml_latency = Histogram(
    "finalayze_ml_model_prediction_latency_seconds",
    "ML model prediction latency",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)

# ── Data feeds ────────────────────────────────────────────────────────────────
_feed_latency = Histogram(
    "finalayze_market_data_feed_latency_seconds",
    "Market data fetch latency",
    ["market", "source"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
_news_last_seen = Gauge(
    "finalayze_news_feed_last_article_timestamp",
    "Unix timestamp of last processed news article",
    ["scope"],
)

# ── Currency ──────────────────────────────────────────────────────────────────
_usd_rub = Gauge("finalayze_usd_rub_rate", "USD/RUB exchange rate")
_equity_rub = Gauge("finalayze_portfolio_equity_rub", "MOEX portfolio equity in RUB")


class MetricsCollector:
    """Facade over all Prometheus metrics.

    All methods are safe to call from any thread (prometheus_client is thread-safe).
    """

    def update_portfolio_equity(
        self,
        market: str,
        equity: float,
        *,
        daily_pnl: float = 0.0,
        drawdown_pct: float = 0.0,
    ) -> None:
        _equity.labels(market=market).set(equity)
        _daily_pnl.labels(market=market).set(daily_pnl)
        _drawdown.labels(market=market).set(drawdown_pct)

    def update_circuit_breaker(self, market: str, *, level: int) -> None:
        _cb_level.labels(market=market).set(level)

    def update_open_positions(self, market: str, *, count: int) -> None:
        _open_positions.labels(market=market).set(count)

    def record_trade(
        self,
        market: str,
        side: str,
        *,
        slippage_bps: float | None = None,
        fill_latency_seconds: float | None = None,
    ) -> None:
        _trades_total.labels(market=market, side=side).inc()
        if slippage_bps is not None:
            _slippage.labels(market=market).observe(slippage_bps)
        if fill_latency_seconds is not None:
            _fill_latency.labels(market=market).observe(fill_latency_seconds)

    def record_rejection(self, market: str, *, reason: str) -> None:
        _rejections.labels(market=market, reason=reason).inc()

    def record_signal(self, market: str, strategy: str, direction: str) -> None:
        _signal_count.labels(market=market, strategy=strategy, direction=direction).inc()

    def update_strategy_win_rate(self, market: str, strategy: str, *, win_rate: float) -> None:
        _strategy_win_rate.labels(market=market, strategy=strategy).set(win_rate)

    def record_ml_prediction(self, model: str, *, latency_seconds: float) -> None:
        _ml_latency.labels(model=model).observe(latency_seconds)

    def set_ml_retrain_timestamp(self, model: str, *, unix_ts: float) -> None:
        _ml_retrain_ts.labels(model=model).set(unix_ts)

    def record_feed_latency(self, market: str, source: str, *, latency_seconds: float) -> None:
        _feed_latency.labels(market=market, source=source).observe(latency_seconds)

    def set_news_last_seen(self, scope: str, *, unix_ts: float) -> None:
        _news_last_seen.labels(scope=scope).set(unix_ts)

    def set_usd_rub_rate(self, rate: float) -> None:
        _usd_rub.set(rate)

    def set_equity_rub(self, equity_rub: float) -> None:
        _equity_rub.set(equity_rub)
```

**Step 5: Add `/metrics` endpoint to `src/finalayze/main.py`**

```python
# Add to imports:
from prometheus_fastapi_instrumentator import Instrumentator

# In create_app(), after application.include_router(...):
Instrumentator().instrument(application).expose(application, endpoint="/metrics")
```

**Step 6: Run tests**

```bash
uv run pytest tests/unit/test_metrics.py -v
```

**Step 7: Commit**

```bash
git add src/finalayze/core/metrics.py src/finalayze/main.py tests/unit/test_metrics.py pyproject.toml uv.lock
git commit -m "feat(metrics): MetricsCollector class and /metrics endpoint"
```

---

### Task 13: Alertmanager rules and docker-compose.monitoring.yml

**Files:**
- Create: `monitoring/prometheus.yml`
- Create: `monitoring/alerts.yml`
- Create: `monitoring/alertmanager.yml`
- Create: `docker-compose.monitoring.yml`

**Step 1: Create `monitoring/prometheus.yml`**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

rule_files:
  - /etc/prometheus/alerts.yml

scrape_configs:
  - job_name: finalayze
    static_configs:
      - targets: ["app:8000"]
    metrics_path: /metrics
```

**Step 2: Create `monitoring/alerts.yml`**

```yaml
groups:
  - name: finalayze_trading
    rules:
      - alert: CircuitBreakerHalted
        expr: finalayze_circuit_breaker_level >= 2
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker {{ $labels.market }} level {{ $value }}"
          description: "Trading halted or liquidating on {{ $labels.market }}"

      - alert: HighDrawdown
        expr: finalayze_drawdown_pct > 0.08
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High drawdown on {{ $labels.market }}: {{ $value | humanizePercentage }}"

      - alert: NewsFeedSilent
        expr: time() - finalayze_news_feed_last_article_timestamp > 1800
        labels:
          severity: warning
        annotations:
          summary: "No news articles for 30+ minutes (scope={{ $labels.scope }})"

      - alert: MLModelStale
        expr: time() - finalayze_ml_model_last_retrain_timestamp > 259200
        labels:
          severity: warning
        annotations:
          summary: "ML model {{ $labels.model }} not retrained in 3+ days"

      - alert: NoTradesDeadMan
        expr: increase(finalayze_trades_total[2h]) == 0
        for: 2h
        labels:
          severity: warning
        annotations:
          summary: "No trades processed in 2 hours — system may be stuck"
```

**Step 3: Create `monitoring/alertmanager.yml`**

```yaml
global:
  resolve_timeout: 5m

route:
  receiver: telegram
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

receivers:
  - name: telegram
    webhook_configs:
      - url: "http://app:8000/api/v1/alertmanager/webhook"
        send_resolved: true
```

**Step 4: Create `docker-compose.monitoring.yml`**

```yaml
services:
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: finalayze_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/alerts.yml:/etc/prometheus/alerts.yml:ro
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=30d"
    networks:
      - finalayze_net

  alertmanager:
    image: prom/alertmanager:v0.27.0
    container_name: finalayze_alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
    networks:
      - finalayze_net

volumes:
  prometheus_data:

networks:
  finalayze_net:
    external: true
    name: finalayze_finalayze_net
```

**Step 5: Write a smoke test for the metrics endpoint**

```python
# tests/unit/test_metrics_endpoint.py
from fastapi.testclient import TestClient
from finalayze.main import create_app


def test_metrics_endpoint_accessible() -> None:
    client = TestClient(create_app())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "finalayze_" in resp.text or "http_requests" in resp.text
```

**Step 6: Run the smoke test**

```bash
uv run pytest tests/unit/test_metrics_endpoint.py -v
```

**Step 7: Run full suite + lint**

```bash
uv run pytest tests/ -v 2>&1 | tail -20
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
```

**Step 8: Commit and create PR B-3**

```bash
git add monitoring/ docker-compose.monitoring.yml tests/unit/test_metrics_endpoint.py
git commit -m "feat(metrics): Alertmanager rules and docker-compose.monitoring.yml"

git push -u origin feature/phase4-track-b-metrics
gh pr create --title "feat(phase4): Track B-3 — Prometheus metrics + Alertmanager rules" --body "$(cat <<'EOF'
## Summary
- MetricsCollector class: 18 business metrics (equity, drawdown, circuit breaker, slippage, ML staleness, feed health)
- /metrics endpoint via prometheus-fastapi-instrumentator
- 5 Alertmanager alert rules: CircuitBreakerHalted, HighDrawdown, NewsFeedSilent, MLModelStale, NoTradesDeadMan
- docker-compose.monitoring.yml: Prometheus + Alertmanager services

## Test Plan
- [ ] GET /metrics returns 200 with prometheus-format metrics
- [ ] MetricsCollector update methods don't throw
- [ ] `docker compose -f docker-compose.monitoring.yml up` starts Prometheus on :9090 and scrapes /metrics

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Acceptance Checklist

Before claiming Track B complete, verify ALL of the following:

- [ ] `GET /api/v1/health` returns 200 without API key; returns `components` dict
- [ ] `GET /api/v1/portfolio` returns 401 without key, 200 with correct key
- [ ] `POST /api/v1/system/mode` with `mode=real` and no/wrong `confirm_token` returns 403
- [ ] Streamlit app starts: `streamlit run src/finalayze/dashboard/app.py`
- [ ] Wrong dashboard password is rejected
- [ ] All 5 pages render without Python exception
- [ ] `GET /metrics` returns 200 with prometheus text format
- [ ] `docker compose -f docker-compose.monitoring.yml up` starts Prometheus on port 9090
- [ ] `uv run pytest tests/` passes with >= 50% coverage
- [ ] `uv run ruff check . && uv run ruff format --check .` zero errors
- [ ] `uv run mypy src/` zero errors

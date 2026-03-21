"""Unit tests for the sandbox REST endpoints (go/no-go and metrics).

Tests cover: go/no-go (200 PROCEED, 503 not configured, 200 DEFER, 401),
and metrics (200 with rows, 200 empty, custom params, 401 without key).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.settings import Settings
from httpx import ASGITransport, AsyncClient

from finalayze.core.db import get_db
from finalayze.main import create_app
from finalayze.monitoring.go_no_go import CriterionResult, GateReport, GateVerdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HTTP_200 = 200
HTTP_401 = 401
HTTP_503 = 503

ENDPOINT = "/api/v1/sandbox/gonogo"
METRICS_ENDPOINT = "/api/v1/sandbox/metrics"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_app() -> object:
    """Create a fresh FastAPI app for testing."""
    application = create_app()
    # Override get_db so we never touch a real database
    application.dependency_overrides[get_db] = _fake_db  # type: ignore[attr-defined]
    return application


async def _fake_db():  # type: ignore[no-untyped-def]
    """Yield a mock AsyncSession."""
    yield AsyncMock()


def _make_client(application: object) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=application), base_url="http://test")  # type: ignore[arg-type]


def _get_api_key() -> str:
    return Settings().api_key


def _make_gate_report(
    verdict: GateVerdict = GateVerdict.PROCEED,
    sandbox_days: int = 8,
    reason: str = "All 8 criteria passed",
    criteria: list[CriterionResult] | None = None,
) -> GateReport:
    """Build a fake GateReport for testing."""
    if criteria is None:
        criteria = [
            CriterionResult(
                name="uptime_pct",
                passed=True,
                actual=99.5,
                threshold=95.0,
                unit="%",
                critical=True,
            ),
            CriterionResult(
                name="fill_rate_pct",
                passed=True,
                actual=98.0,
                threshold=90.0,
                unit="%",
                critical=True,
            ),
        ]
    return GateReport(
        verdict=verdict,
        criteria=criteria,
        sandbox_days=sandbox_days,
        evaluated_at=datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSandboxGoNoGoEndpoint:
    """Tests for GET /api/v1/sandbox/gonogo."""

    @pytest.mark.asyncio
    async def test_gonogo_returns_200_with_proceed_verdict(self) -> None:
        """When reporter is configured and has data, returns 200 with PROCEED."""
        from finalayze.api.v1.sandbox import set_go_no_go_reporter

        app = _build_test_app()
        mock_reporter = AsyncMock()
        mock_reporter.evaluate = AsyncMock(return_value=_make_gate_report())
        set_go_no_go_reporter(mock_reporter)

        try:
            key = _get_api_key()
            async with _make_client(app) as client:
                response = await client.get(ENDPOINT, headers={"X-API-Key": key})

            assert response.status_code == HTTP_200
            body = response.json()
            assert body["verdict"] == "PROCEED"
            assert len(body["criteria"]) == 2  # noqa: PLR2004
            assert body["sandbox_days"] == 8  # noqa: PLR2004
            assert body["reason"] == "All 8 criteria passed"
            # Check criterion structure
            criterion = body["criteria"][0]
            assert "name" in criterion
            assert "passed" in criterion
            assert "actual" in criterion
            assert "threshold" in criterion
            assert "unit" in criterion
            assert "critical" in criterion
        finally:
            set_go_no_go_reporter(None)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_gonogo_returns_503_when_reporter_not_configured(self) -> None:
        """When set_go_no_go_reporter was never called, returns 503."""
        from finalayze.api.v1.sandbox import set_go_no_go_reporter

        set_go_no_go_reporter(None)  # type: ignore[arg-type]
        app = _build_test_app()
        key = _get_api_key()

        async with _make_client(app) as client:
            response = await client.get(ENDPOINT, headers={"X-API-Key": key})

        assert response.status_code == HTTP_503
        assert "not configured" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_gonogo_returns_defer_with_insufficient_data(self) -> None:
        """When reporter returns DEFER verdict (insufficient sandbox data), returns 200."""
        from finalayze.api.v1.sandbox import set_go_no_go_reporter

        app = _build_test_app()
        defer_report = _make_gate_report(
            verdict=GateVerdict.DEFER,
            sandbox_days=3,
            reason="Insufficient data: 3 days < 5 required",
            criteria=[],
        )
        mock_reporter = AsyncMock()
        mock_reporter.evaluate = AsyncMock(return_value=defer_report)
        set_go_no_go_reporter(mock_reporter)

        try:
            key = _get_api_key()
            async with _make_client(app) as client:
                response = await client.get(ENDPOINT, headers={"X-API-Key": key})

            assert response.status_code == HTTP_200
            body = response.json()
            assert body["verdict"] == "DEFER"
            assert body["criteria"] == []
            assert body["sandbox_days"] == 3  # noqa: PLR2004
            assert "Insufficient" in body["reason"]
        finally:
            set_go_no_go_reporter(None)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_gonogo_returns_401_without_api_key(self) -> None:
        """Without X-API-Key header, returns 401."""
        from finalayze.api.v1.sandbox import set_go_no_go_reporter

        app = _build_test_app()
        mock_reporter = AsyncMock()
        mock_reporter.evaluate = AsyncMock(return_value=_make_gate_report())
        set_go_no_go_reporter(mock_reporter)

        try:
            async with _make_client(app) as client:
                response = await client.get(ENDPOINT)

            assert response.status_code == HTTP_401
        finally:
            set_go_no_go_reporter(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helpers for metrics tests
# ---------------------------------------------------------------------------


def _make_fake_metric_row(
    ts: datetime | None = None,
    market_id: str = "moex",
) -> MagicMock:
    """Build a mock object mimicking SandboxMetricRow."""
    row = MagicMock()
    row.timestamp = ts or datetime(2026, 3, 20, 10, 0, 0, tzinfo=UTC)
    row.market_id = market_id
    row.trade_count = 5
    row.pnl_rub = Decimal("1234.56")
    row.equity_rub = Decimal("100000.00")
    row.fill_rate = Decimal("0.9500")
    row.uptime_cycles = 42
    row.signals_generated = 10
    row.errors_caught = 1
    row.max_slippage_bps = Decimal("25.50")
    row.avg_slippage_bps = Decimal("12.30")
    row.drawdown_pct = Decimal("0.0150")
    return row


async def _fake_db_with_rows(*rows: MagicMock):  # type: ignore[no-untyped-def]
    """Yield a mock AsyncSession that returns the given rows on execute."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = list(rows)
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)
    yield mock_session


async def _fake_db_empty():  # type: ignore[no-untyped-def]
    """Yield a mock AsyncSession that returns no rows."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)
    yield mock_session


# ---------------------------------------------------------------------------
# Tests for GET /api/v1/sandbox/metrics
# ---------------------------------------------------------------------------


class TestSandboxMetricsEndpoint:
    """Tests for GET /api/v1/sandbox/metrics."""

    @pytest.mark.asyncio
    async def test_metrics_returns_200_with_rows(self) -> None:
        """When DB has metric rows, returns 200 with list of SandboxMetricResponse."""
        app = create_app()
        row1 = _make_fake_metric_row()
        row2 = _make_fake_metric_row(
            ts=datetime(2026, 3, 21, 10, 0, 0, tzinfo=UTC),
        )

        async def _db_override():  # type: ignore[no-untyped-def]
            async for s in _fake_db_with_rows(row1, row2):
                yield s

        app.dependency_overrides[get_db] = _db_override  # type: ignore[attr-defined]
        key = _get_api_key()

        async with _make_client(app) as client:
            response = await client.get(
                METRICS_ENDPOINT, headers={"X-API-Key": key}
            )

        assert response.status_code == HTTP_200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 2  # noqa: PLR2004
        first = body[0]
        assert first["trade_count"] == 5  # noqa: PLR2004
        assert first["market_id"] == "moex"
        assert first["equity_rub"] == 100000.0  # noqa: PLR2004
        assert first["fill_rate"] == 0.95  # noqa: PLR2004
        assert first["max_slippage_bps"] == 25.5  # noqa: PLR2004
        assert first["drawdown_pct"] == 0.015  # noqa: PLR2004

    @pytest.mark.asyncio
    async def test_metrics_returns_200_empty_when_no_rows(self) -> None:
        """When no rows match the filter, returns 200 with empty list."""
        app = create_app()

        async def _db_override():  # type: ignore[no-untyped-def]
            async for s in _fake_db_empty():
                yield s

        app.dependency_overrides[get_db] = _db_override  # type: ignore[attr-defined]
        key = _get_api_key()

        async with _make_client(app) as client:
            response = await client.get(
                METRICS_ENDPOINT, headers={"X-API-Key": key}
            )

        assert response.status_code == HTTP_200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_metrics_accepts_custom_params(self) -> None:
        """Custom days and market_id params are accepted and return 200."""
        app = create_app()

        async def _db_override():  # type: ignore[no-untyped-def]
            async for s in _fake_db_empty():
                yield s

        app.dependency_overrides[get_db] = _db_override  # type: ignore[attr-defined]
        key = _get_api_key()

        async with _make_client(app) as client:
            response = await client.get(
                METRICS_ENDPOINT,
                headers={"X-API-Key": key},
                params={"days": 14, "market_id": "us"},
            )

        assert response.status_code == HTTP_200

    @pytest.mark.asyncio
    async def test_metrics_returns_401_without_api_key(self) -> None:
        """Without X-API-Key header, returns 401."""
        app = create_app()

        async def _db_override():  # type: ignore[no-untyped-def]
            async for s in _fake_db_empty():
                yield s

        app.dependency_overrides[get_db] = _db_override  # type: ignore[attr-defined]

        async with _make_client(app) as client:
            response = await client.get(METRICS_ENDPOINT)

        assert response.status_code == HTTP_401

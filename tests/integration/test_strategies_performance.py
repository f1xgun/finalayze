"""Integration test for /api/v1/strategies/performance shape (Plan 55-04-04).

Verifies against a live DB that the response JSON carries all the Plan-55
D-16 fields (`segment_id`, `trades_count`, `signal_count`) so the Plan 05
heatmap can render a Strategy x Segment grid.

Skipped when FINALAYZE_DATABASE_URL is not set (the usual CI gating pattern
matched by tests/integration/test_stop_history_endpoint.py).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def _db_url() -> str:
    url = os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("FINALAYZE_DATABASE_URL not set")
    return url


def test_strategies_performance_http_shape() -> None:
    """GET /api/v1/strategies/performance?period=30 returns the D-16 shape."""
    _db_url()

    from config.settings import Settings  # noqa: PLC0415
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from finalayze.main import create_app  # noqa: PLC0415

    resp = TestClient(create_app()).get(
        "/api/v1/strategies/performance?period=30",
        headers={"X-API-Key": Settings().api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "strategies" in data
    assert isinstance(data["strategies"], list)
    for row in data["strategies"]:
        for key in (
            "strategy",
            "market_id",
            "segment_id",
            "win_rate",
            "profit_factor",
            "trades_count",
            "signal_count",
            "last_signal_at",
        ):
            assert key in row, f"missing {key} in {row}"
        assert "trades_today" not in row, "legacy trades_today field must be removed per Plan 55-04"

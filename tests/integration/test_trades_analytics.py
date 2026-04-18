"""Integration test for TRAD-02 analytics endpoint shape.

After ``alembic upgrade head`` (adds ``signals.signal_price`` via migration 008),
the /trades/analytics response must include the new Phase 55 keys (win_rate,
avg_win, avg_loss, profit_factor) plus the retained scaffold keys. Default
period is 30 (D-13).

Skips when no test DB is configured (FINALAYZE_DATABASE_URL / DATABASE_URL).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def _db_url() -> str:
    url = os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("FINALAYZE_DATABASE_URL not set; integration DB unavailable")
    return url


def test_trades_analytics_http_shape() -> None:
    """GET /trades/analytics must expose the TRAD-02 keys after migration 008."""
    from config.settings import Settings  # noqa: PLC0415
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from finalayze.main import create_app  # noqa: PLC0415

    _db_url()  # skip gate only
    resp = TestClient(create_app()).get(
        "/api/v1/trades/analytics",
        headers={"X-API-Key": Settings().api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "period_days",
        "total_trades",
        "win_rate",
        "avg_win",
        "avg_loss",
        "profit_factor",
        "avg_slippage_bps",
        "avg_fill_latency_ms",
        "rejection_rate_pct",
    ):
        assert key in data, f"missing key {key}"
    assert data["period_days"] == 30  # D-13

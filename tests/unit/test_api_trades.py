from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from finalayze.main import create_app


def _auth() -> dict[str, str]:
    from config.settings import Settings

    return {"X-API-Key": Settings().api_key}


def test_trades_list_returns_empty_without_db() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["trades"] == []
    assert resp.json()["total"] == 0


def test_trades_list_requires_auth() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades")
    assert resp.status_code == 401


def test_trades_analytics_returns_empty_without_db() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades/analytics", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_trades"] == 0


def test_trade_detail_returns_500_without_db() -> None:
    resp = TestClient(create_app()).get(f"/api/v1/trades/{uuid.uuid4()}", headers=_auth())
    assert resp.status_code == 500

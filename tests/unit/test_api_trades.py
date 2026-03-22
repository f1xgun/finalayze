from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from finalayze.main import create_app


def _auth() -> dict[str, str]:
    from config.settings import Settings

    return {"X-API-Key": Settings().api_key}


def test_trades_list_returns_501() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades", headers=_auth())
    assert resp.status_code == 501
    assert resp.json()["detail"] == "Not yet implemented"


def test_trades_list_requires_auth() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades")
    assert resp.status_code == 401


def test_trades_analytics_returns_501() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades/analytics", headers=_auth())
    assert resp.status_code == 501
    assert resp.json()["detail"] == "Not yet implemented"


def test_trade_detail_returns_501() -> None:
    resp = TestClient(create_app()).get(f"/api/v1/trades/{uuid.uuid4()}", headers=_auth())
    assert resp.status_code == 501
    assert resp.json()["detail"] == "Not yet implemented"

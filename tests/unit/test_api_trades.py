from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from finalayze.main import create_app


def _auth() -> dict[str, str]:
    from config.settings import Settings

    return {"X-API-Key": Settings().api_key}


def test_trades_list_returns_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades", headers=_auth())
    assert resp.status_code == 200
    assert "trades" in resp.json()


def test_trades_list_requires_auth() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades")
    assert resp.status_code == 422


def test_trades_analytics_returns_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/trades/analytics", headers=_auth())
    assert resp.status_code == 200
    assert "avg_slippage_bps" in resp.json()


def test_trade_detail_returns_404_for_unknown() -> None:
    resp = TestClient(create_app()).get(f"/api/v1/trades/{uuid.uuid4()}", headers=_auth())
    assert resp.status_code == 404

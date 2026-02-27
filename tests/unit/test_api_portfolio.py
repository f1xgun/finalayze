from __future__ import annotations

from fastapi.testclient import TestClient

from finalayze.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _auth() -> dict[str, str]:
    from config.settings import Settings

    return {"X-API-Key": Settings().api_key}


def test_portfolio_unified_requires_auth() -> None:
    resp = _client().get("/api/v1/portfolio")
    assert resp.status_code == 422


def test_portfolio_unified_with_valid_key() -> None:
    resp = _client().get("/api/v1/portfolio", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert "total_equity_usd" in body
    assert "markets" in body


def test_portfolio_positions_with_valid_key() -> None:
    resp = _client().get("/api/v1/portfolio/positions", headers=_auth())
    assert resp.status_code == 200
    assert "positions" in resp.json()
    assert isinstance(resp.json()["positions"], list)


def test_portfolio_history_with_valid_key() -> None:
    resp = _client().get("/api/v1/portfolio/history", headers=_auth())
    assert resp.status_code == 200
    assert "snapshots" in resp.json()


def test_portfolio_performance_with_valid_key() -> None:
    resp = _client().get("/api/v1/portfolio/performance", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert "sharpe_30d" in body
    assert "max_drawdown_pct" in body

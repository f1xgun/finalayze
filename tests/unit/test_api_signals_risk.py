from __future__ import annotations

from unittest.mock import MagicMock

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
    assert "markets" in resp.json()


def test_risk_exposure_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/risk/exposure", headers=_h())
    assert resp.status_code == 200
    assert "segments" in resp.json()


def test_risk_override_requires_auth() -> None:
    resp = TestClient(create_app()).post(
        "/api/v1/risk/override",
        json={"market_id": "us", "level": 1},
    )
    assert resp.status_code == 401


def test_ml_status_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/ml/status", headers=_h())
    assert resp.status_code == 200
    assert "models" in resp.json()


def test_news_list_200() -> None:
    resp = TestClient(create_app()).get("/api/v1/news", headers=_h())
    assert resp.status_code == 200
    assert "articles" in resp.json()


def test_risk_override_valid_applies() -> None:
    app = create_app()
    mock_cb = MagicMock()
    app.state.circuit_breakers = {"us": mock_cb}
    client = TestClient(app)
    resp = client.post("/api/v1/risk/override", json={"market_id": "us", "level": 0}, headers=_h())
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"] is True
    mock_cb.override_level.assert_called_once()


def test_risk_override_out_of_range_level_returns_422() -> None:
    resp = TestClient(create_app()).post(
        "/api/v1/risk/override", json={"market_id": "us", "level": 99}, headers=_h()
    )
    assert resp.status_code == 422


def test_risk_override_unknown_market_returns_404() -> None:
    resp = TestClient(create_app()).post(
        "/api/v1/risk/override", json={"market_id": "unknown", "level": 1}, headers=_h()
    )
    assert resp.status_code == 404

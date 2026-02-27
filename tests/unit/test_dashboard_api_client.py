"""Smoke tests for dashboard API client — httpx responses mocked via respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from finalayze.dashboard.api_client import (
    ApiClient,
    get_health,
    get_portfolio,
    get_risk_status,
    get_signals,
    get_strategies_performance,
    get_system_errors,
    get_trades,
    post_risk_override,
)

_BASE = "http://localhost:8000"
_KEY = "test-key"


@respx.mock
def test_api_client_injects_key() -> None:
    respx.get(f"{_BASE}/api/v1/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = ApiClient(base_url=_BASE, api_key=_KEY)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert respx.calls.last.request.headers["x-api-key"] == _KEY


@respx.mock
def test_api_client_raises_on_401_when_requested() -> None:
    respx.get(f"{_BASE}/api/v1/portfolio").mock(
        return_value=httpx.Response(401, json={"detail": "Invalid API key"})
    )
    client = ApiClient(base_url=_BASE, api_key="bad-key")
    with pytest.raises(httpx.HTTPStatusError):
        client.get("/api/v1/portfolio", raise_on_error=True)


@respx.mock
def test_api_client_does_not_raise_by_default() -> None:
    respx.get(f"{_BASE}/api/v1/portfolio").mock(
        return_value=httpx.Response(401, json={"detail": "Invalid API key"})
    )
    client = ApiClient(base_url=_BASE, api_key="bad-key")
    resp = client.get("/api/v1/portfolio")
    assert resp.status_code == 401


@respx.mock
def test_get_health_returns_dict() -> None:
    respx.get(f"{_BASE}/api/v1/health").mock(
        return_value=httpx.Response(200, json={"status": "ok", "mode": "sandbox"})
    )
    result = get_health(_BASE, _KEY)
    assert result["status"] == "ok"
    assert result["mode"] == "sandbox"


@respx.mock
def test_get_system_errors_returns_list() -> None:
    errors = [
        {
            "timestamp": "2026-02-27T10:00:00",
            "component": "db",
            "message": "timeout",
            "traceback_excerpt": "",
        },
    ]
    respx.get(f"{_BASE}/api/v1/system/errors").mock(return_value=httpx.Response(200, json=errors))
    result = get_system_errors(_BASE, _KEY)
    assert isinstance(result, list)
    assert result[0]["component"] == "db"


@respx.mock
def test_get_portfolio_returns_dict() -> None:
    payload = {"total_equity_usd": 10000.0, "markets": []}
    respx.get(f"{_BASE}/api/v1/portfolio").mock(return_value=httpx.Response(200, json=payload))
    result = get_portfolio(_BASE, _KEY)
    assert result["total_equity_usd"] == 10000.0  # noqa: PLR2004


@respx.mock
def test_get_risk_status_returns_dict() -> None:
    payload = {
        "markets": [
            {
                "market_id": "us",
                "circuit_breaker_level": 0,
                "level_label": "NORMAL",
                "level_since": None,
            },  # noqa: E501
        ],
        "cross_market_halted": False,
    }
    respx.get(f"{_BASE}/api/v1/risk/status").mock(return_value=httpx.Response(200, json=payload))
    result = get_risk_status(_BASE, _KEY)
    assert "markets" in result
    assert result["cross_market_halted"] is False


@respx.mock
def test_post_risk_override_sends_payload() -> None:
    respx.post(f"{_BASE}/api/v1/risk/override").mock(
        return_value=httpx.Response(200, json={"market_id": "us", "level": 1, "applied": True})
    )
    result = post_risk_override(_BASE, _KEY, "us", 1)
    assert result["applied"] is True
    sent_body = respx.calls.last.request.content
    assert b'"market_id"' in sent_body
    assert b'"level"' in sent_body


@respx.mock
def test_get_trades_returns_list() -> None:
    respx.get(f"{_BASE}/api/v1/trades").mock(
        return_value=httpx.Response(200, json={"trades": [], "total": 0})
    )
    result = get_trades(_BASE, _KEY)
    assert isinstance(result, list)


@respx.mock
def test_get_signals_returns_list() -> None:
    respx.get(f"{_BASE}/api/v1/signals").mock(
        return_value=httpx.Response(200, json={"signals": []})
    )
    result = get_signals(_BASE, _KEY)
    assert isinstance(result, list)


@respx.mock
def test_get_strategies_performance_returns_list() -> None:
    respx.get(f"{_BASE}/api/v1/strategies/performance").mock(
        return_value=httpx.Response(200, json={"strategies": []})
    )
    result = get_strategies_performance(_BASE, _KEY)
    assert isinstance(result, list)

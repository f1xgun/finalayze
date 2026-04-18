from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from finalayze.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _auth() -> dict[str, str]:
    from config.settings import Settings

    return {"X-API-Key": Settings().api_key}


def test_portfolio_unified_requires_auth() -> None:
    resp = _client().get("/api/v1/portfolio")
    assert resp.status_code == 401


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


def test_get_single_position_returns_404() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/v1/portfolio/positions/AAPL", headers=_auth())
    assert resp.status_code == 404


# ---------- run_in_executor tests ----------


def _make_mock_portfolio() -> MagicMock:
    """Create a mock PortfolioState with equity and cash."""
    p = MagicMock()
    p.equity = Decimal(100000)
    p.cash = Decimal(50000)
    return p


def test_get_portfolio_uses_run_in_executor() -> None:
    """Verify that broker.get_portfolio() is called via run_in_executor, not directly."""
    import inspect

    from finalayze.api.v1.portfolio import get_portfolio as _endpoint  # noqa: F401

    source = inspect.getsource(_endpoint)
    assert "run_in_executor" in source, (
        "get_portfolio endpoint must use run_in_executor to avoid blocking the event loop"
    )


def test_get_portfolio_returns_broker_data_via_executor() -> None:
    """Portfolio response correctly assembled from broker data through executor."""
    mock_portfolio = _make_mock_portfolio()
    mock_broker = MagicMock()
    mock_broker.get_portfolio.return_value = mock_portfolio

    mock_broker_router = MagicMock()
    mock_broker_router.route.return_value = mock_broker
    mock_broker_router.registered_markets = ["moex"]

    mock_market = MagicMock()
    mock_market.id = "moex"

    mock_registry = MagicMock()
    mock_registry.list_markets.return_value = [mock_market]

    app = create_app()
    app.state.broker_router = mock_broker_router

    with patch("finalayze.api.v1.portfolio.default_registry", return_value=mock_registry):
        client = TestClient(app)
        resp = client.get("/api/v1/portfolio", headers=_auth())

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_equity_usd"] == 100000.0
    assert body["total_cash_usd"] == 50000.0
    assert len(body["markets"]) == 1
    assert body["markets"][0]["market_id"] == "moex"


def test_get_portfolio_skips_market_on_broker_error() -> None:
    """When broker.get_portfolio() raises in executor, market is skipped gracefully."""
    mock_broker = MagicMock()
    mock_broker.get_portfolio.side_effect = RuntimeError("gRPC channel closed")

    mock_broker_router = MagicMock()
    mock_broker_router.route.return_value = mock_broker
    mock_broker_router.registered_markets = ["moex"]

    mock_market = MagicMock()
    mock_market.id = "moex"

    mock_registry = MagicMock()
    mock_registry.list_markets.return_value = [mock_market]

    app = create_app()
    app.state.broker_router = mock_broker_router

    with patch("finalayze.api.v1.portfolio.default_registry", return_value=mock_registry):
        client = TestClient(app)
        resp = client.get("/api/v1/portfolio", headers=_auth())

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_equity_usd"] == 0.0
    assert len(body["markets"]) == 0

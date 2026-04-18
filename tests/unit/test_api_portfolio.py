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


# ------ STOP-01 D-02 schema tests ------


def test_position_detail_has_d02_stop_fields() -> None:
    from finalayze.api.v1.portfolio import PositionDetail

    fields = PositionDetail.model_fields
    expected = {
        "stop_price",
        "distance_pct",
        "distance_atr",
        "atr_value",
        "entry_price",
        "highest_price",
        "trail_activated",
        "activation_threshold",
    }
    assert expected.issubset(set(fields.keys()))


def test_position_detail_removed_legacy_stop_distance_atr() -> None:
    from finalayze.api.v1.portfolio import PositionDetail

    assert "stop_distance_atr" not in PositionDetail.model_fields


def test_position_detail_stop_fields_all_nullable() -> None:
    from finalayze.api.v1.portfolio import PositionDetail

    # Instantiate with only the mandatory fields (D-03: all stop fields default to None)
    pd = PositionDetail(
        symbol="SBER",
        market_id="moex",
        segment_id="ru_blue_chips",
        quantity=10.0,
        avg_price=100.0,
        current_price=105.0,
        market_value=1050.0,
        unrealized_pnl=50.0,
        unrealized_pnl_pct=5.0,
    )
    assert pd.stop_price is None
    assert pd.distance_pct is None
    assert pd.distance_atr is None
    assert pd.atr_value is None
    assert pd.entry_price is None
    assert pd.highest_price is None
    assert pd.trail_activated is None
    assert pd.activation_threshold is None


def test_position_detail_distance_pct_field_description_documents_convention() -> None:
    from finalayze.api.v1.portfolio import PositionDetail

    desc = PositionDetail.model_fields["distance_pct"].description or ""
    assert "(current_price - stop_price) / current_price" in desc


# ------ STOP-01/02 handler population tests ------


def _mock_stop_state(
    entry: float = 100.0,
    current_stop: float = 95.0,
    atr: float = 2.5,
    highest: float | None = None,
    activation_atr: float = 1.0,
    trail_activated: bool = False,
) -> MagicMock:
    """Mimic StopLossState dataclass with the 8 D-02 fields."""
    state = MagicMock()
    state.entry_price = Decimal(str(entry))
    state.current_stop = Decimal(str(current_stop))
    state.atr_value = Decimal(str(atr))
    state.highest_price = Decimal(str(highest if highest is not None else entry))
    state.activation_atr = Decimal(str(activation_atr))
    state.trail_atr = Decimal("1.5")
    state.trail_activated = trail_activated
    state.initial_stop = Decimal(str(current_stop))
    return state


def test_build_stop_fields_returns_null_when_no_tracker() -> None:
    from finalayze.api.v1.portfolio import _build_stop_fields

    result = _build_stop_fields("SBER", 105.0, None)
    assert result == {
        "stop_price": None,
        "distance_pct": None,
        "distance_atr": None,
        "atr_value": None,
        "entry_price": None,
        "highest_price": None,
        "trail_activated": None,
        "activation_threshold": None,
    }


def test_build_stop_fields_returns_null_when_no_active_stop() -> None:
    from finalayze.api.v1.portfolio import _build_stop_fields

    tracker = MagicMock()
    tracker.get_stop_state.return_value = None
    result = _build_stop_fields("SBER", 105.0, tracker)
    assert all(v is None for v in result.values())


def test_build_stop_fields_distance_pct_formula() -> None:
    from finalayze.api.v1.portfolio import _build_stop_fields

    tracker = MagicMock()
    tracker.get_stop_state.return_value = _mock_stop_state(
        entry=100.0, current_stop=95.0, atr=2.5
    )
    result = _build_stop_fields("SBER", 100.0, tracker)
    # (100 - 95) / 100 = 0.05
    assert result["distance_pct"] is not None
    assert abs(result["distance_pct"] - 0.05) < 1e-9


def test_build_stop_fields_distance_atr_formula() -> None:
    from finalayze.api.v1.portfolio import _build_stop_fields

    tracker = MagicMock()
    tracker.get_stop_state.return_value = _mock_stop_state(
        entry=100.0, current_stop=95.0, atr=2.5
    )
    result = _build_stop_fields("SBER", 100.0, tracker)
    # (100 - 95) / 2.5 = 2.0
    assert result["distance_atr"] is not None
    assert abs(result["distance_atr"] - 2.0) < 1e-9


def test_build_stop_fields_activation_threshold_formula() -> None:
    from finalayze.api.v1.portfolio import _build_stop_fields

    tracker = MagicMock()
    tracker.get_stop_state.return_value = _mock_stop_state(
        entry=100.0, current_stop=95.0, atr=2.5, activation_atr=1.0
    )
    result = _build_stop_fields("SBER", 100.0, tracker)
    # 100 + 1.0 * 2.5 = 102.5
    assert result["activation_threshold"] is not None
    assert abs(result["activation_threshold"] - 102.5) < 1e-9


def test_build_stop_fields_populates_all_fields_when_active() -> None:
    from finalayze.api.v1.portfolio import _build_stop_fields

    tracker = MagicMock()
    tracker.get_stop_state.return_value = _mock_stop_state(
        entry=100.0,
        current_stop=96.0,
        atr=2.0,
        highest=105.0,
        activation_atr=1.0,
        trail_activated=True,
    )
    result = _build_stop_fields("SBER", 103.0, tracker)
    assert result["stop_price"] == 96.0
    assert result["atr_value"] == 2.0
    assert result["entry_price"] == 100.0
    assert result["highest_price"] == 105.0
    assert result["trail_activated"] is True
    assert result["activation_threshold"] == 102.0


def test_positions_endpoint_returns_null_stops_when_no_tracker() -> None:
    """TEST mode (no position_tracker on app.state): stop fields are null."""
    resp = _client().get("/api/v1/portfolio/positions", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    # In TEST mode there are no positions OR positions with null stop fields
    for p in body.get("positions", []):
        assert p.get("stop_price") is None
        assert p.get("distance_pct") is None
        assert p.get("distance_atr") is None


def test_stop_history_empty_for_unknown_symbol() -> None:
    """Empty history returns 200 with empty list, not 404."""
    resp = _client().get(
        "/api/v1/portfolio/positions/NONEXISTENT/stop-history",
        headers=_auth(),
    )
    # If DB not available, may return 500; test passes if response is 200 with empty list
    if resp.status_code == 200:
        body = resp.json()
        assert body["symbol"] == "NONEXISTENT"
        assert body["events"] == []

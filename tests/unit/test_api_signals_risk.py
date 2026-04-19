from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from finalayze.main import create_app


def _h() -> dict[str, str]:
    from config.settings import Settings

    return {"X-API-Key": Settings().api_key}


# --- Helpers for /strategies/performance (Plan 55-04) ------------------------


def _make_signal_row(
    *,
    strategy_name: str,
    market_id: str,
    segment_id: str,
    sig_count: int,
    last_signal: datetime,
) -> Any:
    """Return a duck-typed row matching the signals group-by `select()` labels."""
    return SimpleNamespace(
        strategy_name=strategy_name,
        market_id=market_id,
        segment_id=segment_id,
        sig_count=sig_count,
        last_signal=last_signal,
    )


def _make_order(
    *,
    signal_id: uuid.UUID | None,
    strategy_name: str | None,
    segment_id: str,
    market_id: str,
    symbol: str,
    side: str,
    qty: Decimal,
    price: Decimal,
    filled_at: datetime,
) -> Any:
    """Return a duck-typed OrderModel-shaped row with `.signal` relationship pre-populated."""
    signal_obj = None
    if signal_id is not None and strategy_name is not None:
        signal_obj = SimpleNamespace(
            id=signal_id,
            strategy_name=strategy_name,
            market_id=market_id,
            segment_id=segment_id,
        )
    return SimpleNamespace(
        id=uuid.uuid4(),
        signal_id=signal_id,
        signal=signal_obj,
        symbol=symbol,
        market_id=market_id,
        side=side,
        status="filled",
        filled_quantity=qty,
        filled_avg_price=price,
        filled_at=filled_at,
    )


def _seed_patched_factory(sig_rows: list[Any], order_rows: list[Any]) -> Any:
    """Return an async_session_factory mock wired to deliver sig_rows then order_rows.

    The /strategies/performance handler issues two `session.execute()` calls:
      1. Signals group-by → consumed via `.all()`
      2. Orders select     → consumed via `.scalars().all()`
    """

    def _sig_result() -> MagicMock:
        r = MagicMock()
        r.all = MagicMock(return_value=sig_rows)
        return r

    def _order_result() -> MagicMock:
        r = MagicMock()
        scalars = MagicMock()
        scalars.all = MagicMock(return_value=order_rows)
        r.scalars = MagicMock(return_value=scalars)
        return r

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute = AsyncMock(side_effect=[_sig_result(), _order_result()])

    return MagicMock(return_value=mock_session)


def _call_strategies_performance(
    sig_rows: list[Any],
    order_rows: list[Any],
    *,
    query: str = "",
) -> Any:
    """Issue GET /strategies/performance with mocked session factory."""
    factory = _seed_patched_factory(sig_rows, order_rows)
    with patch(
        "finalayze.core.db.get_async_session_factory",
        return_value=factory,
    ):
        url = "/api/v1/strategies/performance"
        if query:
            url = f"{url}?{query}"
        return TestClient(create_app()).get(url, headers=_h())


def test_signals_list_returns_empty_without_db() -> None:
    resp = TestClient(create_app()).get("/api/v1/signals", headers=_h())
    assert resp.status_code == 200
    assert resp.json()["signals"] == []


def test_strategies_performance_returns_empty_without_db() -> None:
    resp = TestClient(create_app()).get("/api/v1/strategies/performance", headers=_h())
    assert resp.status_code == 200
    assert resp.json()["strategies"] == []


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


def test_ml_status_501() -> None:
    resp = TestClient(create_app()).get("/api/v1/ml/status", headers=_h())
    assert resp.status_code == 501
    assert resp.json()["detail"] == "Not yet implemented"


def test_news_list_501() -> None:
    resp = TestClient(create_app()).get("/api/v1/news", headers=_h())
    assert resp.status_code == 501
    assert resp.json()["detail"] == "Not yet implemented"


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


# --- /strategies/performance (Plan 55-04) ------------------------------------
# Closes VALIDATION task IDs 55-04-01..55-04-03 (unit layer).


def test_strategies_performance_credits_closing_order_strategy() -> None:
    """55-04-01 / D-04: P&L credited to the CLOSING order's signal.strategy_name.

    Five paired trades where momentum opens and mean_reversion closes → only
    "mean_reversion" appears with trades_count=5 (not "momentum").
    """
    sig_rows: list[Any] = []  # irrelevant for attribution here — orders dominate
    order_rows: list[Any] = []
    base = datetime.now(UTC) - timedelta(days=1)
    for i in range(5):
        mom_sig_id = uuid.uuid4()
        mr_sig_id = uuid.uuid4()
        buy_at = base + timedelta(hours=i * 2)
        sell_at = buy_at + timedelta(minutes=30)
        order_rows.append(
            _make_order(
                signal_id=mom_sig_id,
                strategy_name="momentum",
                segment_id="ru_blue_chips",
                market_id="moex",
                symbol="SBER",
                side="BUY",
                qty=Decimal(100),
                price=Decimal(280),
                filled_at=buy_at,
            )
        )
        order_rows.append(
            _make_order(
                signal_id=mr_sig_id,
                strategy_name="mean_reversion",
                segment_id="ru_blue_chips",
                market_id="moex",
                symbol="SBER",
                side="SELL",
                qty=Decimal(100),
                price=Decimal(290),
                filled_at=sell_at,
            )
        )

    resp = _call_strategies_performance(sig_rows, order_rows)
    assert resp.status_code == 200
    data = resp.json()["strategies"]

    mr_row = next(
        (
            r
            for r in data
            if r["strategy"] == "mean_reversion" and r["segment_id"] == "ru_blue_chips"
        ),
        None,
    )
    assert mr_row is not None, f"expected mean_reversion row, got {data}"
    assert mr_row["trades_count"] == 5
    # Profit on (290-280)*100 = 1000, cost = 285 * 100 * (4+5)/10000 = 25.65
    # So all 5 are wins after commissions. win_rate==1.0, profit_factor==None (no losses).
    assert mr_row["win_rate"] == 1.0
    assert mr_row["profit_factor"] is None

    mom_row = next(
        (r for r in data if r["strategy"] == "momentum" and r["segment_id"] == "ru_blue_chips"),
        None,
    )
    assert mom_row is None or int(mom_row["trades_count"]) == 0


def test_strategies_performance_sample_gate_below_5_returns_null_metrics() -> None:
    """55-04-02: trades_count < 5 → win_rate/profit_factor forced to None (D-15)."""
    order_rows: list[Any] = []
    base = datetime.now(UTC) - timedelta(hours=6)
    sig_id_buy = uuid.uuid4()
    sig_id_sell = uuid.uuid4()
    order_rows.append(
        _make_order(
            signal_id=sig_id_buy,
            strategy_name="momentum",
            segment_id="ru_blue_chips",
            market_id="moex",
            symbol="SBER",
            side="BUY",
            qty=Decimal(100),
            price=Decimal(280),
            filled_at=base,
        )
    )
    order_rows.append(
        _make_order(
            signal_id=sig_id_sell,
            strategy_name="mean_reversion",
            segment_id="ru_blue_chips",
            market_id="moex",
            symbol="SBER",
            side="SELL",
            qty=Decimal(100),
            price=Decimal(290),
            filled_at=base + timedelta(minutes=30),
        )
    )

    resp = _call_strategies_performance(sig_rows=[], order_rows=order_rows)
    assert resp.status_code == 200
    data = resp.json()["strategies"]
    row = next(
        (
            r
            for r in data
            if r["strategy"] == "mean_reversion" and r["segment_id"] == "ru_blue_chips"
        ),
        None,
    )
    assert row is not None
    assert row["trades_count"] == 1
    assert row["win_rate"] is None
    assert row["profit_factor"] is None


def test_strategies_performance_sample_gate_at_5_returns_numeric_metrics() -> None:
    """55-04-02: trades_count == 5 with all wins → win_rate=1.0, profit_factor=None (Pitfall 1)."""
    order_rows: list[Any] = []
    base = datetime.now(UTC) - timedelta(days=1)
    for i in range(5):
        buy_at = base + timedelta(hours=i * 2)
        sell_at = buy_at + timedelta(minutes=30)
        sig_buy = uuid.uuid4()
        sig_sell = uuid.uuid4()
        order_rows.append(
            _make_order(
                signal_id=sig_buy,
                strategy_name="momentum",
                segment_id="ru_blue_chips",
                market_id="moex",
                symbol="SBER",
                side="BUY",
                qty=Decimal(100),
                price=Decimal(280),
                filled_at=buy_at,
            )
        )
        order_rows.append(
            _make_order(
                signal_id=sig_sell,
                strategy_name="momentum",  # same strategy as opener — simple attribution
                segment_id="ru_blue_chips",
                market_id="moex",
                symbol="SBER",
                side="SELL",
                qty=Decimal(100),
                price=Decimal(290),
                filled_at=sell_at,
            )
        )

    resp = _call_strategies_performance(sig_rows=[], order_rows=order_rows)
    assert resp.status_code == 200
    data = resp.json()["strategies"]
    row = next(
        (r for r in data if r["strategy"] == "momentum" and r["segment_id"] == "ru_blue_chips"),
        None,
    )
    assert row is not None, f"expected momentum/ru_blue_chips row, got {data}"
    assert row["trades_count"] == 5
    assert row["win_rate"] == 1.0
    assert row["profit_factor"] is None  # zero losses → undefined PF


def test_strategies_performance_segment_breakdown() -> None:
    """55-04-03: same strategy across two segments → two separate rows."""
    order_rows: list[Any] = []
    base = datetime.now(UTC) - timedelta(days=1)
    # segment A: ru_blue_chips, 5 pairs
    for i in range(5):
        buy_at = base + timedelta(hours=i * 2)
        sell_at = buy_at + timedelta(minutes=30)
        order_rows.append(
            _make_order(
                signal_id=uuid.uuid4(),
                strategy_name="momentum",
                segment_id="ru_blue_chips",
                market_id="moex",
                symbol="SBER",
                side="BUY",
                qty=Decimal(100),
                price=Decimal(280),
                filled_at=buy_at,
            )
        )
        order_rows.append(
            _make_order(
                signal_id=uuid.uuid4(),
                strategy_name="momentum",
                segment_id="ru_blue_chips",
                market_id="moex",
                symbol="SBER",
                side="SELL",
                qty=Decimal(100),
                price=Decimal(290),
                filled_at=sell_at,
            )
        )
    # segment B: ru_energy, 5 pairs on a different symbol
    for i in range(5):
        buy_at = base + timedelta(hours=i * 2)
        sell_at = buy_at + timedelta(minutes=30)
        order_rows.append(
            _make_order(
                signal_id=uuid.uuid4(),
                strategy_name="momentum",
                segment_id="ru_energy",
                market_id="moex",
                symbol="LKOH",
                side="BUY",
                qty=Decimal(10),
                price=Decimal(6000),
                filled_at=buy_at,
            )
        )
        order_rows.append(
            _make_order(
                signal_id=uuid.uuid4(),
                strategy_name="momentum",
                segment_id="ru_energy",
                market_id="moex",
                symbol="LKOH",
                side="SELL",
                qty=Decimal(10),
                price=Decimal(6200),
                filled_at=sell_at,
            )
        )

    resp = _call_strategies_performance(sig_rows=[], order_rows=order_rows)
    assert resp.status_code == 200
    data = resp.json()["strategies"]
    segs = {(r["strategy"], r["segment_id"]) for r in data if r["strategy"] == "momentum"}
    assert ("momentum", "ru_blue_chips") in segs
    assert ("momentum", "ru_energy") in segs
    # Both cells should carry trades_count==5 and a numeric win_rate
    for r in data:
        if r["strategy"] == "momentum" and r["segment_id"] in {"ru_blue_chips", "ru_energy"}:
            assert r["trades_count"] == 5
            assert r["win_rate"] == 1.0


def test_strategies_performance_orphan_orders_ignored() -> None:
    """Pitfall 5: closing order with signal_id=None does not crash and produces no row."""
    order_rows: list[Any] = []
    base = datetime.now(UTC) - timedelta(hours=2)
    # BUY with valid signal
    order_rows.append(
        _make_order(
            signal_id=uuid.uuid4(),
            strategy_name="momentum",
            segment_id="ru_blue_chips",
            market_id="moex",
            symbol="SBER",
            side="BUY",
            qty=Decimal(100),
            price=Decimal(280),
            filled_at=base,
        )
    )
    # SELL with signal_id=None (orphan) — fifo_pair will still pair, but attribution fails
    order_rows.append(
        _make_order(
            signal_id=None,
            strategy_name=None,
            segment_id="ru_blue_chips",
            market_id="moex",
            symbol="SBER",
            side="SELL",
            qty=Decimal(100),
            price=Decimal(290),
            filled_at=base + timedelta(minutes=30),
        )
    )

    resp = _call_strategies_performance(sig_rows=[], order_rows=order_rows)
    assert resp.status_code == 200
    data = resp.json()["strategies"]
    # No strategy attribution should produce any row — orphan skipped
    assert not any(r["trades_count"] > 0 for r in data), (
        f"expected no trades-count rows, got {data}"
    )


def test_strategies_performance_response_shape_has_new_fields() -> None:
    """Every returned row carries the new D-16 fields: segment_id, trades_count, signal_count."""
    sig_rows = [
        _make_signal_row(
            strategy_name="momentum",
            market_id="moex",
            segment_id="ru_blue_chips",
            sig_count=3,
            last_signal=datetime.now(UTC),
        ),
    ]

    resp = _call_strategies_performance(sig_rows=sig_rows, order_rows=[])
    assert resp.status_code == 200
    data = resp.json()["strategies"]
    assert len(data) == 1
    row = data[0]
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
        assert key in row, f"missing key {key} in {row}"
    assert row["strategy"] == "momentum"
    assert row["segment_id"] == "ru_blue_chips"
    assert row["signal_count"] == 3
    assert row["trades_count"] == 0
    assert row["win_rate"] is None  # no orders
    assert row["profit_factor"] is None
    # `trades_today` was removed from the schema
    assert "trades_today" not in row

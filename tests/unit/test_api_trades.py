from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from fastapi.testclient import TestClient

from finalayze.api.v1._fifo import PairedTrade, fifo_pair
from finalayze.api.v1._slippage import compute_slippage_bps
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


# ── Phase 55-02 Task 3: shared FIFO pairing helper at api/v1/_fifo.py ──


def _mk_order(  # type: ignore[no-untyped-def]
    side: str,
    qty: str,
    price: str,
    filled_at: datetime,
    signal_id: uuid.UUID | None = None,
    status: str = "filled",
    symbol: str = "SBER",
):
    """Build a duck-typed order row with OrderModel-shaped fields."""
    return SimpleNamespace(
        side=side,
        symbol=symbol,
        status=status,
        filled_quantity=Decimal(qty),
        filled_avg_price=Decimal(price),
        filled_at=filled_at,
        signal_id=signal_id,
    )


def test_fifo_pair_single_roundtrip() -> None:
    """One BUY + one SELL produces one PairedTrade; closing_signal_id is the SELL's."""
    sig = uuid.uuid4()
    orders = [
        _mk_order("BUY", "100", "280", datetime(2026, 4, 1, tzinfo=UTC)),
        _mk_order("SELL", "100", "290", datetime(2026, 4, 2, tzinfo=UTC), signal_id=sig),
    ]
    pairs = list(fifo_pair(orders))
    assert len(pairs) == 1
    assert isinstance(pairs[0], PairedTrade)
    assert pairs[0].entry_price == Decimal(280)
    assert pairs[0].exit_price == Decimal(290)
    assert pairs[0].quantity == Decimal(100)
    assert pairs[0].closing_signal_id == sig


def test_fifo_pair_partial_fills_split_by_quantity() -> None:
    """BUY 100 matched by two SELL 50s yields two PairedTrades (D-01 quantity split)."""
    orders = [
        _mk_order("BUY", "100", "280", datetime(2026, 4, 1, tzinfo=UTC)),
        _mk_order("SELL", "50", "290", datetime(2026, 4, 2, tzinfo=UTC)),
        _mk_order("SELL", "50", "295", datetime(2026, 4, 3, tzinfo=UTC)),
    ]
    pairs = list(fifo_pair(orders))
    assert len(pairs) == 2
    assert pairs[0].quantity == Decimal(50)
    assert pairs[1].quantity == Decimal(50)


def test_fifo_pair_sell_without_buy_is_skipped() -> None:
    """A SELL with no prior BUY yields nothing (Pitfall 3 — shorts out of scope)."""
    orders = [_mk_order("SELL", "100", "290", datetime(2026, 4, 2, tzinfo=UTC))]
    assert list(fifo_pair(orders)) == []


# ── Phase 55-03 Task 1: compute_slippage_bps helper + Settings commission fields ──


def test_compute_slippage_bps_buy_positive_when_filled_above_reference() -> None:
    """BUY filled ABOVE the reference price is adverse → positive bps (D-08)."""
    bps = compute_slippage_bps(Decimal(281), Decimal(280), "BUY")
    assert bps is not None
    assert 35.0 < bps < 36.5


def test_compute_slippage_bps_sell_positive_when_filled_below_reference() -> None:
    """SELL filled BELOW the reference price is adverse → positive bps (D-08)."""
    bps = compute_slippage_bps(Decimal(279), Decimal(280), "SELL")
    assert bps is not None
    assert 35.0 < bps < 36.5


def test_compute_slippage_bps_null_reference_returns_none() -> None:
    """No reference price → slippage is unknown (D-07 null fallback)."""
    assert compute_slippage_bps(Decimal(280), None, "BUY") is None


def test_compute_slippage_bps_zero_reference_returns_none() -> None:
    """Zero reference price → guard against div-by-zero (D-07 null fallback)."""
    assert compute_slippage_bps(Decimal(280), Decimal(0), "BUY") is None


def test_settings_exposes_default_commission_bps() -> None:
    """Settings exposes the three new analytics cost fields with the expected defaults."""
    from config.settings import Settings

    s = Settings()
    assert s.default_commission_bps_moex == 4.0  # noqa: PLR2004
    assert s.default_commission_bps_us == 1.0
    assert s.default_slippage_cost_bps == 5.0  # noqa: PLR2004

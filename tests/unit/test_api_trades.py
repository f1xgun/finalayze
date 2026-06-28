from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from finalayze.api.v1._fifo import PairedTrade, fifo_pair
from finalayze.api.v1._slippage import compute_slippage_bps
from finalayze.main import create_app

if TYPE_CHECKING:
    import pytest


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
    # D-13: default period is 30 (was 7 in pre-Phase-55 analytics stub).
    assert data["period_days"] == 30  # noqa: PLR2004


def test_trade_detail_returns_404_for_unknown_trade() -> None:
    # Audit 2026-06-28: the old test asserted 500 "without db", but the test env
    # HAS a reachable DB, so an unknown id is simply not-found -> 404 (the honest
    # behavior; the 404 branch was previously uncovered).
    resp = TestClient(create_app()).get(f"/api/v1/trades/{uuid.uuid4()}", headers=_auth())
    assert resp.status_code == 404


def test_trade_detail_returns_500_on_db_error(monkeypatch) -> None:
    # A genuine DB failure (not a missing row) must surface as 500, not 404.
    def _boom() -> None:
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("finalayze.core.db.get_async_session_factory", _boom)
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


# ── Phase 55-03 Task 2a: trade_analytics handler (FIFO + win_rate/PF/avg_win/avg_loss) ──


def _mk_row(  # type: ignore[no-untyped-def]
    side: str,
    qty: str,
    price: str,
    filled_at: datetime,
    *,
    symbol: str = "SBER",
    market_id: str = "moex",
    status: str = "filled",
    signal_price: Decimal | None = None,
    signal_id: uuid.UUID | None = None,
):
    """Build a duck-typed OrderModel-shaped row for analytics tests.

    When signal_price is provided, attaches a SimpleNamespace ``signal`` with
    that price so ``selectinload(OrderModel.signal)`` substitute returns a
    realistic eagerly-loaded attribute.
    """
    signal = (
        SimpleNamespace(signal_price=signal_price, id=signal_id or uuid.uuid4())
        if signal_price is not None
        else None
    )
    return SimpleNamespace(
        id=uuid.uuid4(),
        side=side,
        symbol=symbol,
        market_id=market_id,
        status=status,
        filled_quantity=Decimal(qty),
        filled_avg_price=Decimal(price),
        submitted_at=filled_at,
        filled_at=filled_at,
        signal_id=signal_id,
        signal=signal,
    )


def _patch_session_for_rows(monkeypatch: pytest.MonkeyPatch, rows: list[object]) -> None:
    """Patch finalayze.core.db.get_async_session_factory so handlers see ``rows``.

    Builds a MagicMock async-context-manager session whose ``execute`` returns
    a result that yields ``rows`` on ``.scalars().all()``. Second+ calls (e.g.
    the COUNT query in list_trades) also see the same rows/len. Works for both
    single-execute (trade_analytics) and multi-execute (list_trades) handlers.
    """
    scalars = MagicMock()
    scalars.all.return_value = list(rows)
    exec_result = MagicMock()
    exec_result.scalars.return_value = scalars
    exec_result.scalar.return_value = len(rows)
    exec_result.scalar_one_or_none.return_value = rows[0] if rows else None

    session = MagicMock()
    session.execute = AsyncMock(return_value=exec_result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=cm)

    monkeypatch.setattr("finalayze.core.db.get_async_session_factory", lambda: factory)


def test_trades_analytics_default_period_30(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-13: default period on /trades/analytics is 30 days, not 7."""
    _patch_session_for_rows(monkeypatch, [])
    resp = TestClient(create_app()).get("/api/v1/trades/analytics", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["period_days"] == 30  # noqa: PLR2004


def test_fifo_pair_orders_realized_pnl(monkeypatch: pytest.MonkeyPatch) -> None:
    """BUY 100@280 + SELL 100@290 → 1 trade, 100% win_rate, profit_factor None (no losses)."""
    now = datetime.now(UTC)
    rows = [
        _mk_row(
            "BUY",
            "100",
            "280",
            now - timedelta(hours=2),
            signal_price=Decimal(280),
        ),
        _mk_row(
            "SELL",
            "100",
            "290",
            now - timedelta(hours=1),
            signal_price=Decimal(290),
        ),
    ]
    _patch_session_for_rows(monkeypatch, rows)
    resp = TestClient(create_app()).get("/api/v1/trades/analytics", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["period_days"] == 30  # noqa: PLR2004
    assert data["total_trades"] == 1
    assert data["win_rate"] == 1.0
    assert data["avg_win"] is not None
    assert data["avg_win"] > 0
    # Pitfall 1: no losses → profit_factor must be None, not Infinity
    assert data["profit_factor"] is None
    # No losses → avg_loss is None too
    assert data["avg_loss"] is None


def test_analytics_win_rate_covers_commissions(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-03: pnl must exceed commission+slippage cost to count as a win.

    BUY 100@280 + SELL 100@280.02 → pnl=2.00. On MOEX 4 bps + 5 bps slippage
    against avg notional ≈ 28001 → cost ≈ 25.20. 2.00 < 25.20 → not a win.
    """
    now = datetime.now(UTC)
    rows = [
        _mk_row(
            "BUY",
            "100",
            "280",
            now - timedelta(hours=2),
            signal_price=Decimal(280),
        ),
        _mk_row(
            "SELL",
            "100",
            "280.02",
            now - timedelta(hours=1),
            signal_price=Decimal("280.02"),
        ),
    ]
    _patch_session_for_rows(monkeypatch, rows)
    resp = TestClient(create_app()).get("/api/v1/trades/analytics", headers=_auth())
    data = resp.json()
    assert data["total_trades"] == 1
    # pnl 2.00 < cost threshold ~25.20 → loss bucket
    assert data["win_rate"] == 0.0


def test_analytics_profit_factor_decimal_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decimal arithmetic end-to-end: profit_factor must match exact Decimal ratio.

    Builds a large-gain trade and a large-loss trade whose PF would diverge
    between float-summed and Decimal-summed totals. Both pnls exceed the D-03
    cost threshold (big moves), so the wins/losses buckets fire cleanly.
    """
    now = datetime.now(UTC)
    rows = [
        # Winner: BUY 100@280, SELL 100@320 → pnl 4000 >> cost threshold
        _mk_row("BUY", "100", "280", now - timedelta(days=3), signal_price=Decimal(280)),
        _mk_row(
            "SELL",
            "100",
            "320",
            now - timedelta(days=3) + timedelta(hours=1),
            signal_price=Decimal(320),
        ),
        # Loser: BUY 100@300, SELL 100@280 → pnl -2000 < 0
        _mk_row("BUY", "100", "300", now - timedelta(days=2), signal_price=Decimal(300)),
        _mk_row(
            "SELL",
            "100",
            "280",
            now - timedelta(days=2) + timedelta(hours=1),
            signal_price=Decimal(280),
        ),
    ]
    _patch_session_for_rows(monkeypatch, rows)
    resp = TestClient(create_app()).get("/api/v1/trades/analytics", headers=_auth())
    data = resp.json()
    assert data["total_trades"] == 2  # noqa: PLR2004
    # Decimal exact: gross_win=4000, gross_loss=2000 → PF=2.0 exactly
    assert data["profit_factor"] == 2.0
    assert data["win_rate"] == 0.5


def test_analytics_profit_factor_none_when_no_losses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pitfall 1: all wins → profit_factor is None (not raised, not infinity)."""
    now = datetime.now(UTC)
    rows = [
        _mk_row("BUY", "100", "280", now - timedelta(hours=2), signal_price=Decimal(280)),
        _mk_row(
            "SELL",
            "100",
            "320",
            now - timedelta(hours=1),
            signal_price=Decimal(320),
        ),
    ]
    _patch_session_for_rows(monkeypatch, rows)
    resp = TestClient(create_app()).get("/api/v1/trades/analytics", headers=_auth())
    data = resp.json()
    assert data["total_trades"] == 1
    assert data["win_rate"] == 1.0
    assert data["profit_factor"] is None  # no losses


# ── Phase 55-03 Task 2b: list_trades + get_trade populate slippage_bps ──


def test_list_trades_populates_slippage_bps_when_signal_price_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TRAD-01 read path: BUY filled above signal_price yields positive bps (D-08)."""
    now = datetime.now(UTC)
    rows = [
        _mk_row(
            "BUY",
            "100",
            "281",
            now - timedelta(hours=1),
            signal_price=Decimal(280),
        ),
    ]
    _patch_session_for_rows(monkeypatch, rows)
    resp = TestClient(create_app()).get("/api/v1/trades", headers=_auth())
    assert resp.status_code == 200
    trades = resp.json()["trades"]
    assert len(trades) == 1
    assert trades[0]["slippage_bps"] is not None
    assert trades[0]["slippage_bps"] > 0  # BUY filled above reference = adverse (D-08)


def test_list_trades_slippage_null_when_no_signal_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TRAD-01 read path: legacy signal with signal_price=None → slippage_bps null (D-07)."""
    now = datetime.now(UTC)
    rows = [
        # signal_price=None → SimpleNamespace(signal=None) per _mk_row helper
        _mk_row("BUY", "50", "190", now - timedelta(hours=1), signal_price=None),
    ]
    _patch_session_for_rows(monkeypatch, rows)
    resp = TestClient(create_app()).get("/api/v1/trades", headers=_auth())
    trades = resp.json()["trades"]
    assert len(trades) == 1
    assert trades[0]["slippage_bps"] is None


def test_get_trade_populates_slippage_bps(monkeypatch: pytest.MonkeyPatch) -> None:
    """TRAD-01 read path: single-trade fetch returns the same slippage as the list endpoint."""
    now = datetime.now(UTC)
    row = _mk_row(
        "BUY",
        "100",
        "281",
        now - timedelta(hours=1),
        signal_price=Decimal(280),
    )
    _patch_session_for_rows(monkeypatch, [row])
    resp = TestClient(create_app()).get(f"/api/v1/trades/{row.id}", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["slippage_bps"] is not None
    assert data["slippage_bps"] > 0

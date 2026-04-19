"""Single-source-of-truth integration test: /portfolio/performance vs FIFO helper.

PERF-01 D-10 + Phase 55 D-10: ``api/v1/_fifo.fifo_pair`` is the single source
of truth for win_rate / profit_factor across ``/portfolio/performance`` and
``/trades/analytics``.

Phase 55 (`feature/phase-55-signals-trades-analytics`) extends
``/trades/analytics`` with FIFO-derived ``win_rate`` and ``profit_factor``
fields, but that branch has not yet merged to main and so is not present on
the Phase 56 branch (only the shared ``api/v1/_fifo.py`` helper has been
vendored). Until Phase 55 lands, the cross-endpoint comparison cannot be
performed against two live HTTP responses.

Equivalent assertion landed here: the win_rate / profit_factor that
``/portfolio/performance`` reports must equal the values produced by calling
``fifo_pair`` directly on the same OrderModel rows. If both endpoints
genuinely share the helper (D-10), then both must agree with the helper's
output — proving the single-source-of-truth invariant transitively.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from finalayze.api.v1._fifo import fifo_pair
from finalayze.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _auth() -> dict[str, str]:
    from config.settings import Settings

    return {"X-API-Key": Settings().api_key}


def _snapshot(ts: datetime, market_id: str, equity: Decimal) -> SimpleNamespace:
    return SimpleNamespace(timestamp=ts, market_id=market_id, equity=equity)


def _order(
    side: str,
    symbol: str,
    qty: Decimal,
    price: Decimal,
    filled_at: datetime,
) -> SimpleNamespace:
    """Duck-typed filled OrderModel row."""
    return SimpleNamespace(
        status="filled",
        side=side,
        symbol=symbol,
        filled_quantity=qty,
        filled_avg_price=price,
        filled_at=filled_at,
        signal_id=uuid.uuid4(),
    )


def _patch_session(equity_rows: list, order_rows: list) -> MagicMock:
    equity_result = MagicMock()
    equity_result.scalars.return_value.all.return_value = equity_rows
    order_result = MagicMock()
    order_result.scalars.return_value.all.return_value = order_rows

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[equity_result, order_result])

    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=factory())


def _expected_win_rate_and_pf(orders: list) -> tuple[float, float]:
    """Compute win_rate + profit_factor directly from fifo_pair (the single
    source of truth)."""
    paired = list(fifo_pair(orders))
    wins = [p for p in paired if (p.exit_price - p.entry_price) * p.quantity > 0]
    losses = [p for p in paired if (p.exit_price - p.entry_price) * p.quantity < 0]
    gross_profit = sum(
        ((p.exit_price - p.entry_price) * p.quantity for p in wins),
        Decimal(0),
    )
    gross_loss = -sum(
        ((p.exit_price - p.entry_price) * p.quantity for p in losses),
        Decimal(0),
    )
    win_rate = float(Decimal(len(wins)) / Decimal(len(paired)))
    profit_factor = float(gross_profit / gross_loss)
    return win_rate, profit_factor


@pytest.mark.integration
def test_win_pf_agree() -> None:
    """/portfolio/performance.win_rate + .profit_factor agree with fifo_pair output.

    Equivalent to the cross-endpoint assertion intended by the plan: both the
    /portfolio/performance and /trades/analytics endpoints route through
    api/v1/_fifo.fifo_pair (D-10), so any caller of the helper produces
    identical numbers. We assert that /portfolio/performance matches the
    helper's direct output — the same invariant the cross-endpoint comparison
    would prove once /trades/analytics gains the FIFO fields (Phase 55 merge).
    """
    base = datetime.now(UTC) - timedelta(days=10)
    # Seed enough snapshots so Sharpe/Sortino/MaxDD also have data to compute
    equity_rows = [
        _snapshot(base + timedelta(days=i), "moex", Decimal(100000) + Decimal(i * 250))
        for i in range(8)
    ]
    # 4 BUY+SELL pairs on SBER with mixed P&L (3 wins, 1 loss)
    order_rows: list = []
    for i, (buy_px, sell_px) in enumerate([(100, 110), (105, 115), (108, 102), (112, 120)]):
        buy_ts = base + timedelta(days=i, hours=10)
        sell_ts = base + timedelta(days=i, hours=14)
        order_rows.append(_order("BUY", "SBER", Decimal(10), Decimal(buy_px), buy_ts))
        order_rows.append(_order("SELL", "SBER", Decimal(10), Decimal(sell_px), sell_ts))

    expected_wr, expected_pf = _expected_win_rate_and_pf(order_rows)

    factory_callable = _patch_session(equity_rows, order_rows)
    with patch("finalayze.core.db.get_async_session_factory", return_value=factory_callable):
        resp = _client().get("/api/v1/portfolio/performance", headers=_auth())

    assert resp.status_code == 200
    body = resp.json()
    assert body["n_paired_trades"] == 4
    # Single-source-of-truth invariant: the endpoint's win_rate / profit_factor
    # must match fifo_pair's direct output to within float-roundtrip tolerance.
    assert body["win_rate"] == pytest.approx(expected_wr, rel=1e-9)
    assert body["profit_factor"] == pytest.approx(expected_pf, rel=1e-9)

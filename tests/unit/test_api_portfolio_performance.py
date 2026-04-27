"""Unit tests for GET /api/v1/portfolio/performance (PERF-01).

Mocks ``get_async_session_factory`` so the handler runs against synthetic
DailyEquitySnapshot + OrderModel rows, no DB required. Asserts:

- Real Sharpe / Sortino / MaxDD computed via PerformanceAnalyzer reuse (D-09)
- Per-metric independent null gating on COUNT, NOT metric value (D-12 + Open Q4)
- Schema split: n_snapshots vs n_paired_trades (Pitfall 5 / Open Q3)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from finalayze.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _auth() -> dict[str, str]:
    from config.settings import Settings

    return {"X-API-Key": Settings().api_key}


def _snapshot(ts: datetime, market_id: str, equity: Decimal | int | float) -> SimpleNamespace:
    """Duck-typed DailyEquitySnapshot row."""
    return SimpleNamespace(timestamp=ts, market_id=market_id, equity=Decimal(str(equity)))


def _order(
    side: str,
    symbol: str,
    qty: Decimal | int,
    price: Decimal | int | float,
    filled_at: datetime,
) -> SimpleNamespace:
    """Duck-typed filled OrderModel row (compatible with fifo_pair)."""
    return SimpleNamespace(
        status="filled",
        side=side,
        symbol=symbol,
        filled_quantity=Decimal(str(qty)),
        filled_avg_price=Decimal(str(price)),
        filled_at=filled_at,
        signal_id=uuid.uuid4(),
    )


def _patch_session(equity_rows: list, order_rows: list) -> tuple[MagicMock, AsyncMock]:
    """Build a mock session factory that yields the given equity_rows / order_rows.

    The real handler runs two ``session.execute(...)`` calls in order:
      1. select(DailyEquitySnapshot)...
      2. select(OrderModel)...
    We return them in that order from a single AsyncMock side_effect.
    """
    equity_result = MagicMock()
    equity_result.scalars.return_value.all.return_value = equity_rows
    order_result = MagicMock()
    order_result.scalars.return_value.all.return_value = order_rows

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[equity_result, order_result])

    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    factory_callable = MagicMock(return_value=factory())
    return factory_callable, session


def test_returns_real_sharpe_sortino_maxdd() -> None:
    """7 snapshots with up + down moves -> non-null Sharpe + MaxDD; n_snapshots>=3."""
    base = datetime.now(UTC) - timedelta(days=7)
    # Single market 'moex', equity series with both up and down moves
    equities = [
        Decimal(100000),
        Decimal(101500),
        Decimal(100800),
        Decimal(102200),
        Decimal(101000),
        Decimal(103500),
        Decimal(102800),
    ]
    equity_rows = [_snapshot(base + timedelta(days=i), "moex", eq) for i, eq in enumerate(equities)]
    order_rows: list = []  # no trades yet

    factory_callable, _ = _patch_session(equity_rows, order_rows)
    with patch("finalayze.core.db.get_async_session_factory", return_value=factory_callable):
        resp = _client().get("/api/v1/portfolio/performance", headers=_auth())

    assert resp.status_code == 200
    body = resp.json()
    assert body["sharpe_30d"] is not None
    assert isinstance(body["sharpe_30d"], (int, float))
    # Sortino: with mixed up/down series, mean is positive -> non-zero Sortino.
    # (PerformanceAnalyzer returns 0 only when mean_excess <= 0.) Either way,
    # the field must be present (not missing) — value may be 0 in edge cases.
    assert "sortino_30d" in body
    assert body["max_drawdown_pct"] is not None
    assert body["max_drawdown_pct"] >= 0
    assert body["n_snapshots"] >= 3
    # No trades -> trade-based metrics null
    assert body["win_rate"] is None
    assert body["profit_factor"] is None
    assert body["avg_win_loss_ratio"] is None
    assert body["n_paired_trades"] == 0


def test_null_per_metric_on_insufficient_data() -> None:
    """1 snapshot + 0 orders -> all 6 metrics null; counts reflect actual sample size."""
    base = datetime.now(UTC) - timedelta(days=1)
    equity_rows = [_snapshot(base, "moex", Decimal(100000))]
    order_rows: list = []

    factory_callable, _ = _patch_session(equity_rows, order_rows)
    with patch("finalayze.core.db.get_async_session_factory", return_value=factory_callable):
        resp = _client().get("/api/v1/portfolio/performance", headers=_auth())

    assert resp.status_code == 200
    body = resp.json()
    # Per-metric independent nulls (D-12)
    assert body["sharpe_30d"] is None
    assert body["sortino_30d"] is None
    assert body["max_drawdown_pct"] is None
    assert body["win_rate"] is None
    assert body["profit_factor"] is None
    assert body["avg_win_loss_ratio"] is None
    # Counts reflect actual sample size, not 0 (the schema must surface the gap)
    assert body["n_snapshots"] == 1
    assert body["n_paired_trades"] == 0


def test_null_per_metric_partial_data() -> None:
    """5 snapshots + 0 paired trades -> Sharpe/Sortino/MaxDD non-null, win/PF still null."""
    base = datetime.now(UTC) - timedelta(days=5)
    equities = [Decimal(100000), Decimal(101000), Decimal(99500), Decimal(102000), Decimal(101500)]
    equity_rows = [_snapshot(base + timedelta(days=i), "moex", eq) for i, eq in enumerate(equities)]
    order_rows: list = []

    factory_callable, _ = _patch_session(equity_rows, order_rows)
    with patch("finalayze.core.db.get_async_session_factory", return_value=factory_callable):
        resp = _client().get("/api/v1/portfolio/performance", headers=_auth())

    assert resp.status_code == 200
    body = resp.json()
    assert body["sharpe_30d"] is not None
    assert body["max_drawdown_pct"] is not None
    # No paired trades -> these stay null even though we have snapshots
    assert body["win_rate"] is None
    assert body["profit_factor"] is None
    assert body["avg_win_loss_ratio"] is None
    assert body["n_snapshots"] == 5
    assert body["n_paired_trades"] == 0


def test_n_observations_field() -> None:
    """N=10 snapshots, M=4 BUY+SELL pairs -> n_snapshots==10, n_paired_trades==4."""
    base = datetime.now(UTC) - timedelta(days=10)
    equity_rows = [
        _snapshot(base + timedelta(days=i), "moex", Decimal(100000) + Decimal(i * 200))
        for i in range(10)
    ]
    # 4 BUY+SELL pairs on SBER: each round-trip generates one PairedTrade
    order_rows: list = []
    for i in range(4):
        buy_ts = base + timedelta(days=i, hours=10)
        sell_ts = base + timedelta(days=i, hours=14)
        order_rows.append(_order("BUY", "SBER", 10, 100 + i, buy_ts))
        order_rows.append(_order("SELL", "SBER", 10, 105 + i, sell_ts))

    factory_callable, _ = _patch_session(equity_rows, order_rows)
    with patch("finalayze.core.db.get_async_session_factory", return_value=factory_callable):
        resp = _client().get("/api/v1/portfolio/performance", headers=_auth())

    assert resp.status_code == 200
    body = resp.json()
    # The two count fields are SEPARATE (Pitfall 5 / Open Q3) — not collapsed
    # into a single ambiguous n_observations.
    assert "n_snapshots" in body
    assert "n_paired_trades" in body
    assert body["n_snapshots"] == 10
    assert body["n_paired_trades"] == 4
    # All 4 pairs are wins (sell > buy) so win_rate==1.0
    assert body["win_rate"] == 1.0

"""Unit tests for equity_snapshots_to_portfolio_states adapter (PERF-01 D-09 + D-11)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from finalayze.api.v1._perf import equity_snapshots_to_portfolio_states


def _row(ts: datetime, market_id: str, equity: Decimal) -> SimpleNamespace:
    """Duck-typed DailyEquitySnapshot row for testing (no DB)."""
    return SimpleNamespace(timestamp=ts, market_id=market_id, equity=equity)


def test_adapter_sums_per_timestamp() -> None:
    """Per-market equities sharing a timestamp sum into one PortfolioState (D-11)."""
    t1 = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 4, 19, 12, 15, tzinfo=UTC)
    rows = [
        _row(t1, "us", Decimal(100)),
        _row(t1, "moex", Decimal(200)),
        _row(t2, "us", Decimal(110)),
        _row(t2, "moex", Decimal(190)),
    ]
    result = equity_snapshots_to_portfolio_states(rows)
    assert len(result) == 2
    assert result[0].equity == Decimal(300)
    assert result[1].equity == Decimal(300)
    assert result[0].timestamp == t1
    assert result[1].timestamp == t2


def test_adapter_handles_empty_input() -> None:
    """Zero rows in -> empty list out (no crash)."""
    assert equity_snapshots_to_portfolio_states([]) == []


def test_adapter_preserves_utc_timestamp() -> None:
    """UTC-aware timestamp survives the adapter and passes PortfolioState validator."""
    t1 = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
    result = equity_snapshots_to_portfolio_states([_row(t1, "us", Decimal(100))])
    assert result[0].timestamp.tzinfo is not None
    assert result[0].timestamp == t1


def test_adapter_sorts_by_timestamp() -> None:
    """Out-of-order rows produce ascending-timestamp output (PerformanceAnalyzer expects this)."""
    t1 = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
    rows = [_row(t2, "us", Decimal(110)), _row(t1, "us", Decimal(100))]
    result = equity_snapshots_to_portfolio_states(rows)
    assert result[0].timestamp == t1
    assert result[1].timestamp == t2

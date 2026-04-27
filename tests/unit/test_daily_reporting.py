"""Unit tests for DailyReportingService.persist_cycle_snapshot (EQTY-01 D-01..D-03).

The per-cycle equity snapshot writer is invoked from
`TradingLoop._strategy_cycle_impl` after the per-market loop completes
(D-02 Route B). It mirrors `daily_reset` lines 102-138 minus the circuit
breaker reset and Telegram summary, and reuses
`TradingPersistence.persist_equity_snapshots` (which is fire-and-forget
under the PERSIST-05 envelope).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.orchestration.daily_reporting import DailyReportingService


def test_persist_cycle_snapshot_writes_per_market_plus_bonds() -> None:
    """persist_cycle_snapshot fires fire-and-forget per-market + moex_bonds (EQTY-01 D-01, D-03)."""
    reporter = MagicMock(spec=DailyReportingService)
    reporter._circuit_breakers = {"us": MagicMock(), "moex": MagicMock()}

    us_portfolio = MagicMock(equity=Decimal(50000))
    moex_portfolio = MagicMock(equity=Decimal(3000000))
    reporter._broker_router = MagicMock()
    reporter._broker_router.route.side_effect = lambda m: (
        MagicMock(get_portfolio=lambda: us_portfolio)
        if m == "us"
        else MagicMock(get_portfolio=lambda: moex_portfolio)
    )

    bond_ledger = MagicMock(current_equity=Decimal(1000000))
    reporter._bond_processor = MagicMock(_layer_ledgers={"core": bond_ledger})
    reporter._persistence = MagicMock()
    now = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)

    DailyReportingService.persist_cycle_snapshot(reporter, now)

    reporter._persistence.persist_equity_snapshots.assert_called_once()
    call_args = reporter._persistence.persist_equity_snapshots.call_args
    baselines, ts = call_args[0]
    assert baselines == {
        "us": Decimal(50000),
        "moex": Decimal(3000000),
        "moex_bonds": Decimal(1000000),
    }
    assert ts == now


def test_persist_cycle_snapshot_failure_swallowed() -> None:
    """Per-market broker errors do not abort the writer (PERSIST-05)."""
    reporter = MagicMock(spec=DailyReportingService)
    reporter._circuit_breakers = {"us": MagicMock(), "moex": MagicMock()}

    # us broker raises; moex returns valid equity
    moex_portfolio = MagicMock(equity=Decimal(3000000))
    reporter._broker_router = MagicMock()
    reporter._broker_router.route.side_effect = lambda m: (
        MagicMock(get_portfolio=MagicMock(side_effect=RuntimeError("us broker down")))
        if m == "us"
        else MagicMock(get_portfolio=lambda: moex_portfolio)
    )
    reporter._bond_processor = None
    reporter._persistence = MagicMock()
    now = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)

    # Should NOT raise -- us is logged-and-skipped
    DailyReportingService.persist_cycle_snapshot(reporter, now)

    # moex still made it into baselines
    baselines, _ = reporter._persistence.persist_equity_snapshots.call_args[0]
    assert "moex" in baselines
    assert "us" not in baselines  # broker call failed


def test_persist_cycle_snapshot_no_bond_processor() -> None:
    """When _bond_processor is None, no moex_bonds key is added (graceful skip)."""
    reporter = MagicMock(spec=DailyReportingService)
    reporter._circuit_breakers = {"moex": MagicMock()}

    moex_portfolio = MagicMock(equity=Decimal(3000000))
    reporter._broker_router = MagicMock()
    reporter._broker_router.route.return_value = MagicMock(get_portfolio=lambda: moex_portfolio)
    reporter._bond_processor = None
    reporter._persistence = MagicMock()
    now = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)

    DailyReportingService.persist_cycle_snapshot(reporter, now)

    baselines, _ = reporter._persistence.persist_equity_snapshots.call_args[0]
    assert "moex_bonds" not in baselines
    assert baselines == {"moex": Decimal(3000000)}


def test_persist_cycle_snapshot_empty_baselines_skips_persist() -> None:
    """When all markets fail and no bond processor, persist is NOT called (early-return)."""
    reporter = MagicMock(spec=DailyReportingService)
    reporter._circuit_breakers = {"us": MagicMock()}

    reporter._broker_router = MagicMock()
    reporter._broker_router.route.side_effect = RuntimeError("broker down")
    reporter._bond_processor = None
    reporter._persistence = MagicMock()
    now = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)

    DailyReportingService.persist_cycle_snapshot(reporter, now)

    reporter._persistence.persist_equity_snapshots.assert_not_called()

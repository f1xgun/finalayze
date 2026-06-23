"""Unit tests for the SAA persistence write path (Phase 77 P2-05).

Tests the deposit-tranche upsert statement without a live database. The reload
reconstruction (P2-06) is DEFERRED (see docs/research/phase77_saa_persistence_design.md
§Outcome): a replay-based reconstruction must match the live broker's TRADING-day accrual
cadence, not a calendar-day range -- a correct version (persist the year-scoped accumulators,
or replay over the real trading calendar) is a focused follow-up.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from decimal import Decimal
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.dialects import postgresql

from finalayze.core.schemas import DepositTranche
from finalayze.execution.deposit_broker import DepositSimulatedBroker
from finalayze.execution.deposit_loader import (
    restore_deposit_accumulators,
    serialize_deposit_accumulators,
)


class _AsyncCM:
    """Minimal async context manager yielding a mock session."""

    def __init__(self, session: AsyncMock) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncMock:
        return self._session

    async def __aexit__(self, *args: object) -> bool:
        return False


class TestDepositTrancheUpsert:
    """Per-bar accrual upsert (P2-05) — mock session, no DB."""

    @pytest.mark.asyncio
    async def test_upsert_issues_on_conflict_do_update(self) -> None:
        """The accrual upsert is an ON CONFLICT DO UPDATE on deposit_tranches that updates the
        MUTABLE columns only (accrued_*/broken/updated_at) -- in-place, never duplicating a row.
        """
        from finalayze.orchestration.db_persistence import TradingPersistence  # noqa: PLC0415

        persistence = TradingPersistence(db_url="postgresql://fake", async_loop=None)
        portfolio_id = uuid.uuid4()
        tranche = DepositTranche(
            principal=Decimal(100000),
            term_months=3,
            annual_rate=Decimal("0.1800"),
            open_date=date(2026, 1, 15),
            maturity_date=date(2026, 4, 15),
            accrued_net=Decimal(1000),
            accrued_gross=Decimal(1100),
        )
        mock_session = AsyncMock()
        with patch.object(
            persistence, "_get_bg_session_factory", return_value=lambda: _AsyncCM(mock_session)
        ):
            await persistence._upsert_deposit_tranches_async([tranche], portfolio_id)

        assert mock_session.execute.await_count == 1
        assert mock_session.commit.await_count == 1
        stmt = mock_session.execute.await_args.args[0]
        sql = str(stmt.compile(dialect=postgresql.dialect())).upper()
        assert "INSERT INTO DEPOSIT_TRANCHES" in sql
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql
        # The set_ clause updates the mutable mark/state columns, not the identity columns.
        assert "ACCRUED_NET" in sql
        assert "ACCRUED_GROSS" in sql

    @pytest.mark.asyncio
    async def test_upsert_wrapper_skips_empty_tranches(self) -> None:
        """An empty tranche list does not trigger a DB write."""
        from finalayze.orchestration.db_persistence import TradingPersistence  # noqa: PLC0415

        persistence = TradingPersistence(db_url="postgresql://fake", async_loop=None)
        portfolio_id = uuid.uuid4()
        with patch.object(persistence, "_persist_to_db_async") as mock_persist:
            await persistence.upsert_deposit_tranches_async([], portfolio_id)
        mock_persist.assert_not_called()


class TestDepositAccumulatorReload:
    """Persist -> restore reload (P2-06): DIRECT load, no replay, cadence-independent."""

    @staticmethod
    def _dates(start: date, n: int) -> list[date]:
        return [start + timedelta(days=i) for i in range(n)]

    _OPEN = date(2025, 12, 1)
    _N_BARS = 100  # crosses the Jan-1 YTD reset
    _TRANCHE: ClassVar[dict[str, object]] = {
        "principal": Decimal(100000),
        "term_months": 12,
        "annual_rate": Decimal("0.18"),
        "open_date": _OPEN,
        "maturity_date": date(2026, 12, 1),
    }

    def test_restore_then_next_accrue_matches_live(self) -> None:
        """After serialize -> restore, the NEXT accrue() resumes BIT-IDENTICALLY to a broker that
        never restarted -- the binding gate. No replay, so no calendar/trading-day cadence
        assumption (the CR-01 bug class is structurally impossible here).
        """
        live = DepositSimulatedBroker(
            initial_cash=Decimal(0), tranches=[DepositTranche(**self._TRANCHE)]
        )
        for d in self._dates(self._OPEN, self._N_BARS):
            live.accrue(d)

        # Persist: tranche marks (deposit_tranches) + accumulators (saa_portfolios JSONB).
        acc = serialize_deposit_accumulators(live)
        live_tr = live._tranches[0]  # noqa: SLF001
        # Reload: fresh broker carrying the persisted MARKS + restored accumulators (no replay).
        reloaded = DepositSimulatedBroker(
            initial_cash=Decimal(0),
            tranches=[
                DepositTranche(
                    **self._TRANCHE,
                    accrued_net=live_tr.accrued_net,
                    accrued_gross=live_tr.accrued_gross,
                )
            ],
        )
        restore_deposit_accumulators(reloaded, acc)

        # State restored exactly.
        assert reloaded._ytd_deposit_gross == live._ytd_deposit_gross  # noqa: SLF001
        assert reloaded._running_max_key_rate == live._running_max_key_rate  # noqa: SLF001
        assert reloaded._current_year == live._current_year  # noqa: SLF001
        assert reloaded._total_tax_paid == live._total_tax_paid  # noqa: SLF001
        assert reloaded._tranches[0].accrued_net > Decimal(0)  # noqa: SLF001  # non-trivial

        # The NEXT bar accrues bit-identically on both brokers (the real gate).
        next_d = self._OPEN + timedelta(days=self._N_BARS)
        assert live.accrue(next_d) == reloaded.accrue(next_d)
        assert reloaded._tranches[0].accrued_net == live._tranches[0].accrued_net  # noqa: SLF001
        assert reloaded._ytd_deposit_gross == live._ytd_deposit_gross  # noqa: SLF001

    def test_restore_round_trips_through_json(self) -> None:
        """serialize -> json round-trip -> restore preserves state (Decimals as strings)."""
        import json  # noqa: PLC0415

        live = DepositSimulatedBroker(
            initial_cash=Decimal(0), tranches=[DepositTranche(**self._TRANCHE)]
        )
        for d in self._dates(self._OPEN, self._N_BARS):
            live.accrue(d)
        acc = json.loads(json.dumps(serialize_deposit_accumulators(live)))  # JSONB round-trip
        reloaded = DepositSimulatedBroker(initial_cash=Decimal(0), tranches=[])
        restore_deposit_accumulators(reloaded, acc)
        assert reloaded._ytd_deposit_gross == live._ytd_deposit_gross  # noqa: SLF001
        assert reloaded._running_max_key_rate == live._running_max_key_rate  # noqa: SLF001
        assert reloaded._current_year == live._current_year  # noqa: SLF001

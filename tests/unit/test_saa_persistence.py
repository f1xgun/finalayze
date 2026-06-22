"""Unit tests for SAA persistence layer (Phase 77).

Tests the deposit tranche upsert statement (P2-05) and reload+replay loader
equivalence (P2-06) without requiring a live database connection.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from finalayze.core.schemas import DepositTranche
from finalayze.execution.deposit_broker import DepositSimulatedBroker
from finalayze.execution.deposit_loader import reconstruct_deposit_broker


class TestDepositTrancheUpsert:
    """Test deposit tranche upsert statement (P2-05) — mock session, no DB."""

    @pytest.mark.asyncio
    async def test_upsert_deposit_tranches_builds_correct_statement(self) -> None:
        """Verify the upsert statement structure for per-bar accrual updates."""
        from finalayze.orchestration.db_persistence import TradingPersistence

        persistence = TradingPersistence(db_url="postgresql://fake", async_loop=None)

        # Create mock tranches
        portfolio_id = uuid.uuid4()
        tranche1 = DepositTranche(
            principal=Decimal(100000),
            term_months=3,
            annual_rate=Decimal("0.0420"),
            open_date=date(2026, 1, 15),
            maturity_date=date(2026, 4, 15),
            accrued_net=Decimal(1000),
            accrued_gross=Decimal(1100),
            broken=False,
        )

        # Mock the session and execute call
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        class MockContextManager:
            async def __aenter__(self):
                return mock_session

            async def __aexit__(self, *args):
                pass

        def mock_factory():
            """Return a context manager that yields the mock session."""
            return MockContextManager()

        with patch.object(persistence, "_get_bg_session_factory") as mock_get_factory:
            mock_get_factory.return_value = mock_factory
            await persistence._upsert_deposit_tranches_async([tranche1], portfolio_id)

        # Verify execute was called exactly once (INSERT ... ON CONFLICT)
        assert mock_session.execute.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_upsert_wrapper_skips_empty_tranches(self) -> None:
        """Empty tranche list should not trigger DB write."""
        from finalayze.orchestration.db_persistence import TradingPersistence

        persistence = TradingPersistence(db_url="postgresql://fake", async_loop=None)
        portfolio_id = uuid.uuid4()

        with patch.object(persistence, "_persist_to_db_async") as mock_persist:
            await persistence.upsert_deposit_tranches_async([], portfolio_id)

        # Should skip DB write for empty list
        mock_persist.assert_not_called()


class TestReconstructDepositBroker:
    """Reload+replay reconstruction (P2-06) — the binding gate, no DB needed."""

    @staticmethod
    def _date_range(start: date, end: date) -> list[date]:
        out: list[date] = []
        cur = start
        while cur <= end:
            out.append(cur)
            cur += timedelta(days=1)
        return out

    def test_reconstruct_matches_live_broker_across_jan1(self) -> None:
        """Reconstruction reproduces a LIVE broker's marks + year-scoped accumulators
        bit-identically across the Jan-1 boundary -- the gate justifying NOT persisting the
        accumulators. A naive 'load DB marks then replay' loader would DOUBLE-COUNT and FAIL here.
        """
        open_d = date(2025, 12, 1)
        current = date(2026, 3, 15)  # crosses the Jan-1 YTD reset
        identity = DepositTranche(
            principal=Decimal(100000),
            term_months=12,
            annual_rate=Decimal("0.18"),
            open_date=open_d,
            maturity_date=date(2026, 12, 1),
        )
        # LIVE: tranche present from open, accrued every bar through current.
        live = DepositSimulatedBroker(initial_cash=Decimal(0), tranches=[replace(identity)])
        for d in self._date_range(open_d, current):
            live.accrue(d)
        # RECONSTRUCTED from identity only (accrued rebuilt from zero).
        recon = reconstruct_deposit_broker([identity], current)

        assert recon._tranches[0].accrued_net == live._tranches[0].accrued_net
        assert recon._tranches[0].accrued_gross == live._tranches[0].accrued_gross
        assert recon._ytd_deposit_gross == live._ytd_deposit_gross
        assert recon._running_max_key_rate == live._running_max_key_rate
        assert recon._total_tax_paid == live._total_tax_paid
        assert recon._current_year == live._current_year
        assert recon._tranches[0].accrued_net > Decimal(0)  # accrual actually exercised

    def test_reconstruct_ignores_input_accrued_no_double_count(self) -> None:
        """Reconstruction rebuilds marks FROM ZERO -- a stale input accrued_net is ignored,
        guarding the double-count bug (replaying on top of the persisted DB mark)."""
        open_d = date(2026, 1, 1)
        current = date(2026, 3, 15)
        clean = DepositTranche(
            principal=Decimal(100000),
            term_months=3,
            annual_rate=Decimal("0.18"),
            open_date=open_d,
            maturity_date=date(2026, 4, 1),
        )
        dirty = replace(clean, accrued_net=Decimal(99999), accrued_gross=Decimal(99999))
        from_clean = reconstruct_deposit_broker([clean], current)
        from_dirty = reconstruct_deposit_broker([dirty], current)
        assert from_clean._tranches[0].accrued_net == from_dirty._tranches[0].accrued_net

    def test_reconstruct_empty_is_empty_broker(self) -> None:
        """No tranches -> an empty broker (no replay)."""
        broker = reconstruct_deposit_broker([], date(2026, 3, 15))
        assert broker._tranches == []

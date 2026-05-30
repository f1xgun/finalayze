"""Unit tests for FundamentalSnapshot / ReportEvent schemas (Phase 59, FUND-01).

Covers the frozen-schema contract and the MoexMarketData.fundamentals seam.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from finalayze.core.schemas import (
    FundamentalSnapshot,
    MoexMarketData,
    ReportEvent,
)

_PE_RATIO = 5.2
_EPS_TTM = 120.0


class TestFundamentalSnapshot:
    """FundamentalSnapshot frozen-schema behaviour."""

    def test_constructs_and_is_frozen(self) -> None:
        snap = FundamentalSnapshot(
            symbol="SBER",
            as_of=datetime(2025, 1, 2, tzinfo=UTC),
            pe_ratio=_PE_RATIO,
            eps_ttm=_EPS_TTM,
        )
        assert snap.pe_ratio == _PE_RATIO
        assert snap.eps_ttm == _EPS_TTM
        with pytest.raises(ValidationError):
            snap.pe_ratio = 9.9  # type: ignore[misc]

    def test_unavailable_fields_default_to_none(self) -> None:
        snap = FundamentalSnapshot(
            symbol="SBER",
            as_of=datetime(2025, 1, 2, tzinfo=UTC),
        )
        # None = unavailable, never fabricated.
        assert snap.pe_ratio is None
        assert snap.ev_ebitda is None
        assert snap.revenue_ttm is None
        assert snap.net_margin is None
        assert snap.roe is None
        assert snap.eps_ttm is None
        assert snap.dividend_yield is None
        assert snap.market_cap is None
        assert snap.currency is None


class TestReportEvent:
    """ReportEvent frozen-schema behaviour."""

    def test_constructs(self) -> None:
        event = ReportEvent(
            symbol="SBER",
            report_date=datetime(2025, 7, 1, tzinfo=UTC),
            period_year=2025,
            period_num=2,
            period_type="QUARTER",
        )
        assert event.period_type == "QUARTER"
        assert event.period_year == 2025
        assert event.period_num == 2


class TestMoexMarketDataFundamentals:
    """The L2->L3 ambient-container seam."""

    def test_accepts_fundamentals_tuple(self) -> None:
        snap = FundamentalSnapshot(
            symbol="SBER",
            as_of=datetime(2025, 1, 2, tzinfo=UTC),
            pe_ratio=_PE_RATIO,
        )
        data = MoexMarketData(fundamentals=(snap,))
        assert data.fundamentals is not None
        assert data.fundamentals[0].pe_ratio == _PE_RATIO

    def test_defaults_to_none_and_is_frozen(self) -> None:
        data = MoexMarketData()
        assert data.fundamentals is None
        with pytest.raises((AttributeError, TypeError)):
            data.fundamentals = ()  # type: ignore[misc]

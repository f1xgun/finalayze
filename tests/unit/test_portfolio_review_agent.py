"""Tests for advisory portfolio review agent (Layer 3).

Covers schema safety (PFRA-03), field disjointness from Signal/OrderRequest,
frozen model enforcement, and code-grep for forbidden order-pipeline references.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from finalayze.analysis.portfolio_review_agent import (
    CatalystEvent,
    ConcentrationWarning,
    PortfolioReviewResult,
    PositionSummary,
)

# ── Schema Validation ──────────────────────────────────────────────────────


class TestPositionSummary:
    """PositionSummary uses ticker/market, NOT symbol/market_id."""

    def test_valid_construction(self) -> None:
        ps = PositionSummary(
            ticker="SBER",
            market="moex",
            quantity=Decimal(100),
            unrealized_pnl=Decimal("2340.50"),
            pct_of_portfolio=0.22,
        )
        assert ps.ticker == "SBER"
        assert ps.market == "moex"
        assert ps.quantity == Decimal(100)
        assert ps.unrealized_pnl == Decimal("2340.50")
        assert ps.pct_of_portfolio == pytest.approx(0.22)

    def test_uses_ticker_not_symbol(self) -> None:
        """Field is 'ticker', not 'symbol'."""
        assert "ticker" in PositionSummary.model_fields
        assert "symbol" not in PositionSummary.model_fields

    def test_uses_market_not_market_id(self) -> None:
        """Field is 'market', not 'market_id'."""
        assert "market" in PositionSummary.model_fields
        assert "market_id" not in PositionSummary.model_fields

    def test_frozen(self) -> None:
        ps = PositionSummary(
            ticker="SBER",
            market="moex",
            quantity=Decimal(100),
            unrealized_pnl=Decimal(0),
            pct_of_portfolio=0.1,
        )
        with pytest.raises(ValidationError):
            ps.ticker = "GAZP"  # type: ignore[misc]


class TestConcentrationWarning:
    """ConcentrationWarning validation."""

    def test_valid_construction(self) -> None:
        cw = ConcentrationWarning(
            ticker="SBER",
            market="moex",
            concentration_pct=0.25,
            warning_level="HIGH",
        )
        assert cw.ticker == "SBER"
        assert cw.concentration_pct == pytest.approx(0.25)
        assert cw.warning_level == "HIGH"

    def test_frozen(self) -> None:
        cw = ConcentrationWarning(
            ticker="SBER",
            market="moex",
            concentration_pct=0.25,
            warning_level="HIGH",
        )
        with pytest.raises(ValidationError):
            cw.warning_level = "LOW"  # type: ignore[misc]


class TestCatalystEvent:
    """CatalystEvent validation."""

    def test_valid_construction(self) -> None:
        ce = CatalystEvent(
            ticker="SBER",
            event_type="cbr_meeting",
            expected_date="2026-04-25",
        )
        assert ce.ticker == "SBER"
        assert ce.event_type == "cbr_meeting"
        assert ce.expected_date == "2026-04-25"

    def test_frozen(self) -> None:
        ce = CatalystEvent(
            ticker="SBER",
            event_type="earnings",
            expected_date="2026-05-01",
        )
        with pytest.raises(ValidationError):
            ce.event_type = "dividend"  # type: ignore[misc]


class TestPortfolioReviewResult:
    """PortfolioReviewResult schema safety and validation."""

    def test_valid_construction(self) -> None:
        result = PortfolioReviewResult(
            reviewed_at=datetime.now(tz=UTC),
            positions=[
                PositionSummary(
                    ticker="SBER",
                    market="moex",
                    quantity=Decimal(100),
                    unrealized_pnl=Decimal(2340),
                    pct_of_portfolio=0.22,
                ),
            ],
            concentration_warnings=[
                ConcentrationWarning(
                    ticker="SBER",
                    market="moex",
                    concentration_pct=0.22,
                    warning_level="HIGH",
                ),
            ],
            catalyst_events=[
                CatalystEvent(
                    ticker="SBER",
                    event_type="cbr_meeting",
                    expected_date="2026-04-25",
                ),
            ],
            overall_assessment="Portfolio moderately concentrated in financials.",
            risk_score=0.52,
        )
        assert len(result.positions) == 1
        assert len(result.concentration_warnings) == 1
        assert len(result.catalyst_events) == 1
        assert result.risk_score == pytest.approx(0.52)

    def test_frozen(self) -> None:
        result = PortfolioReviewResult(
            reviewed_at=datetime.now(tz=UTC),
            overall_assessment="OK",
            risk_score=0.3,
        )
        with pytest.raises(ValidationError):
            result.risk_score = 0.9  # type: ignore[misc]

    def test_default_factory_empty_lists(self) -> None:
        """PortfolioReviewResult validates with minimal required fields only."""
        result = PortfolioReviewResult(
            reviewed_at=datetime.now(tz=UTC),
            overall_assessment="No positions open.",
            risk_score=0.0,
        )
        assert result.positions == []
        assert result.concentration_warnings == []
        assert result.catalyst_events == []

    def test_no_direction_field(self) -> None:
        """PFRA-03: no 'direction' field (from Signal)."""
        assert "direction" not in PortfolioReviewResult.model_fields

    def test_no_confidence_field(self) -> None:
        """PFRA-03: no 'confidence' field (from Signal)."""
        assert "confidence" not in PortfolioReviewResult.model_fields

    def test_no_side_field(self) -> None:
        """PFRA-03: no 'side' field (from OrderRequest)."""
        assert "side" not in PortfolioReviewResult.model_fields


class TestModuleLevelForbiddenAssertion:
    """Module-level _FORBIDDEN_FIELDS assertion runs at import time."""

    def test_forbidden_fields_constant_exists(self) -> None:
        from finalayze.analysis.portfolio_review_agent import _FORBIDDEN_FIELDS

        assert {"direction", "confidence", "side"} == _FORBIDDEN_FIELDS

    def test_review_fields_disjoint_from_forbidden(self) -> None:
        from finalayze.analysis.portfolio_review_agent import _FORBIDDEN_FIELDS

        review_fields = set(PortfolioReviewResult.model_fields)
        overlap = review_fields & _FORBIDDEN_FIELDS
        assert not overlap, f"Forbidden fields found: {overlap}"


class TestCodeGrepSafety:
    """Code-grep: zero references to BrokerRouter, place_order, generate_signal."""

    _AGENT_FILE = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "finalayze"
        / "analysis"
        / "portfolio_review_agent.py"
    )

    @staticmethod
    def _read_agent_source() -> str:
        agent_file = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "finalayze"
            / "analysis"
            / "portfolio_review_agent.py"
        )
        return agent_file.read_text(encoding="utf-8")

    def test_no_broker_router_reference(self) -> None:
        source = self._read_agent_source()
        assert "BrokerRouter" not in source, "Found BrokerRouter in portfolio_review_agent.py"

    def test_no_place_order_reference(self) -> None:
        source = self._read_agent_source()
        assert "place_order" not in source, "Found place_order in portfolio_review_agent.py"

    def test_no_generate_signal_reference(self) -> None:
        source = self._read_agent_source()
        assert "generate_signal" not in source, "Found generate_signal in portfolio_review_agent.py"

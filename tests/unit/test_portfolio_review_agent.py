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


# ── Prompt Builder Tests ───────────────────────────────────────────────────


class TestBuildReviewPrompt:
    """build_review_prompt() constructs an LLM prompt from portfolio data dict."""

    def test_contains_position_tickers(self) -> None:
        from finalayze.analysis.portfolio_review_agent import build_review_prompt

        data: dict[str, object] = {
            "moex": {
                "equity": Decimal(500000),
                "cash": Decimal(50000),
                "positions": {"SBER": Decimal(100), "GAZP": Decimal(50)},
            },
        }
        prompt = build_review_prompt(data)
        assert "SBER" in prompt
        assert "GAZP" in prompt

    def test_contains_equity(self) -> None:
        from finalayze.analysis.portfolio_review_agent import build_review_prompt

        data: dict[str, object] = {
            "moex": {
                "equity": Decimal(500000),
                "cash": Decimal(50000),
                "positions": {"SBER": Decimal(100)},
            },
        }
        prompt = build_review_prompt(data)
        assert "500000" in prompt

    def test_empty_portfolio(self) -> None:
        from finalayze.analysis.portfolio_review_agent import build_review_prompt

        prompt = build_review_prompt({})
        assert isinstance(prompt, str)
        assert len(prompt) > 0  # Should still produce a valid prompt

    def test_multiple_markets(self) -> None:
        from finalayze.analysis.portfolio_review_agent import build_review_prompt

        data: dict[str, object] = {
            "moex": {
                "equity": Decimal(500000),
                "cash": Decimal(50000),
                "positions": {"SBER": Decimal(100)},
            },
            "us": {
                "equity": Decimal(10000),
                "cash": Decimal(2000),
                "positions": {"AAPL": Decimal(10)},
            },
        }
        prompt = build_review_prompt(data)
        assert "moex" in prompt
        assert "us" in prompt
        assert "SBER" in prompt
        assert "AAPL" in prompt


# ── Telegram Formatter Tests ──────────────────────────────────────────────


class TestFormatReviewTelegram:
    """format_review_telegram() produces structured Telegram messages."""

    def _make_result(
        self,
        *,
        positions: list[PositionSummary] | None = None,
        warnings: list[ConcentrationWarning] | None = None,
        catalysts: list[CatalystEvent] | None = None,
        assessment: str = "Portfolio looks healthy.",
        risk: float = 0.3,
    ) -> PortfolioReviewResult:
        return PortfolioReviewResult(
            reviewed_at=datetime(2026, 4, 14, 16, 0, tzinfo=UTC),
            positions=positions or [],
            concentration_warnings=warnings or [],
            catalyst_events=catalysts or [],
            overall_assessment=assessment,
            risk_score=risk,
        )

    def test_contains_header(self) -> None:
        from finalayze.analysis.portfolio_review_agent import format_review_telegram

        result = self._make_result()
        msg = format_review_telegram(result)
        assert "Portfolio Review" in msg

    def test_contains_position_tickers(self) -> None:
        from finalayze.analysis.portfolio_review_agent import format_review_telegram

        result = self._make_result(
            positions=[
                PositionSummary(
                    ticker="SBER",
                    market="moex",
                    quantity=Decimal(100),
                    unrealized_pnl=Decimal(2340),
                    pct_of_portfolio=0.22,
                ),
                PositionSummary(
                    ticker="GAZP",
                    market="moex",
                    quantity=Decimal(50),
                    unrealized_pnl=Decimal(-850),
                    pct_of_portfolio=0.18,
                ),
            ],
        )
        msg = format_review_telegram(result)
        assert "SBER" in msg
        assert "GAZP" in msg

    def test_contains_concentration_warnings(self) -> None:
        from finalayze.analysis.portfolio_review_agent import format_review_telegram

        result = self._make_result(
            warnings=[
                ConcentrationWarning(
                    ticker="SBER",
                    market="moex",
                    concentration_pct=0.22,
                    warning_level="HIGH",
                ),
            ],
        )
        msg = format_review_telegram(result)
        assert "Concentration" in msg
        assert "SBER" in msg
        assert "HIGH" in msg

    def test_contains_catalyst_events(self) -> None:
        from finalayze.analysis.portfolio_review_agent import format_review_telegram

        result = self._make_result(
            catalysts=[
                CatalystEvent(
                    ticker="SBER",
                    event_type="cbr_meeting",
                    expected_date="2026-04-25",
                ),
            ],
        )
        msg = format_review_telegram(result)
        assert "Catalyst" in msg
        assert "SBER" in msg
        assert "cbr_meeting" in msg

    def test_contains_assessment_and_risk(self) -> None:
        from finalayze.analysis.portfolio_review_agent import format_review_telegram

        result = self._make_result(
            assessment="Moderately concentrated in financials.",
            risk=0.52,
        )
        msg = format_review_telegram(result)
        assert "Moderately concentrated" in msg
        assert "0.52" in msg

    def test_empty_lists_graceful(self) -> None:
        from finalayze.analysis.portfolio_review_agent import format_review_telegram

        result = self._make_result()
        msg = format_review_telegram(result)
        assert isinstance(msg, str)
        assert len(msg) > 0
        # Should have some indication of no positions
        lower_msg = msg.lower()
        assert "no open positions" in lower_msg or "no positions" in lower_msg or "0" in msg

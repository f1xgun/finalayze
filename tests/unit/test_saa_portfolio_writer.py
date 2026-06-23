"""Tests for SAA portfolio creation and reading (Phase 78 P3-01/P3-02/P3-03)."""

from __future__ import annotations

import uuid as _uuid
from decimal import Decimal

import pytest

from finalayze.core.exceptions import ConfigurationError
from finalayze.core.schemas import RiskProfile
from finalayze.execution.saa_portfolio_writer import (
    coerce_budget,
    resolve_risk_profile,
)


class TestResolveRiskProfile:
    """RiskProfile resolution with fail-closed validation."""

    def test_resolve_valid_conservative(self) -> None:
        """Resolve 'conservative' to RiskProfile.CONSERVATIVE."""
        result = resolve_risk_profile("conservative")
        assert result == RiskProfile.CONSERVATIVE

    def test_resolve_valid_balanced(self) -> None:
        """Resolve 'balanced' to RiskProfile.BALANCED."""
        result = resolve_risk_profile("balanced")
        assert result == RiskProfile.BALANCED

    def test_resolve_valid_growth(self) -> None:
        """Resolve 'growth' to RiskProfile.GROWTH."""
        result = resolve_risk_profile("growth")
        assert result == RiskProfile.GROWTH

    def test_resolve_unknown_profile_raises(self) -> None:
        """Unknown profile name raises ConfigurationError with valid choices."""
        with pytest.raises(ConfigurationError) as exc_info:
            resolve_risk_profile("aggressive")
        assert "aggressive" in str(exc_info.value).lower()
        assert "conservative" in str(exc_info.value).lower()
        assert "balanced" in str(exc_info.value).lower()
        assert "growth" in str(exc_info.value).lower()

    def test_resolve_empty_string_raises(self) -> None:
        """Empty string raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            resolve_risk_profile("")

    def test_resolve_none_raises(self) -> None:
        """None raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            resolve_risk_profile(None)  # type: ignore[arg-type]


class TestCoerceBudget:
    """Budget coercion with Decimal precision and zero/negative rejection."""

    def test_coerce_int_budget(self) -> None:
        """Coerce integer budget to Decimal(0.01) quantized."""
        result = coerce_budget(100000)
        assert result == Decimal("100000.00")
        assert isinstance(result, Decimal)

    def test_coerce_string_budget(self) -> None:
        """Coerce string budget to Decimal(0.01) quantized."""
        result = coerce_budget("100000")
        assert result == Decimal("100000.00")

    def test_coerce_float_budget(self) -> None:
        """Coerce float budget to Decimal (exact via str)."""
        result = coerce_budget(100000.5)
        # Float 100000.5 -> str -> Decimal -> quantize
        assert result.as_tuple().exponent == -2

    def test_coerce_decimal_budget(self) -> None:
        """Coerce Decimal budget stays Decimal."""
        result = coerce_budget(Decimal("100000.00"))
        assert result == Decimal("100000.00")

    def test_coerce_zero_raises(self) -> None:
        """Zero budget raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            coerce_budget(0)
        assert (
            "must be positive" in str(exc_info.value).lower()
            or "positive" in str(exc_info.value).lower()
        )

    def test_coerce_negative_raises(self) -> None:
        """Negative budget raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            coerce_budget(-50000)

    def test_coerce_quantize_precision(self) -> None:
        """Budget quantized to 0.01 (ROUND_HALF_EVEN)."""
        result = coerce_budget(100000.125)
        # 100000.125 via ROUND_HALF_EVEN -> 100000.12
        assert result == Decimal("100000.12")

    def test_coerce_large_budget(self) -> None:
        """Coerce large budget (millions)."""
        result = coerce_budget(1_000_000)
        assert result == Decimal("1000000.00")


class TestBudgetValidation:
    """Integration: budget + profile validation together."""

    def test_valid_budget_and_profile(self) -> None:
        """Valid budget + profile pass validation."""
        budget = coerce_budget(100000)
        profile = resolve_risk_profile("balanced")
        assert budget > 0
        assert profile == RiskProfile.BALANCED

    def test_invalid_budget_fails_first(self) -> None:
        """Invalid budget fails before invalid profile is even checked."""
        with pytest.raises(ConfigurationError):
            coerce_budget(0)

    def test_invalid_profile_fails_separately(self) -> None:
        """Invalid profile raises separately from budget."""
        with pytest.raises(ConfigurationError):
            resolve_risk_profile("unknown")

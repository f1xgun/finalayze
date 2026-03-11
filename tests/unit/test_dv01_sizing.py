"""Unit tests for DV01-based bond position sizing."""

from __future__ import annotations

from decimal import Decimal

from finalayze.risk.dv01_sizing import DV01BudgetStep


class TestDV01BudgetStep:
    """Tests for DV01BudgetStep.compute_position_size."""

    def test_basic_sizing(self) -> None:
        """1.5M equity, 5% DD, 200bps -> max_dv01 = 375.

        With DV01 per unit = 7.5 and no existing portfolio DV01:
            max_by_dv01 = 375 / 7.5 = 50 bonds
            max_by_position = 1_500_000 * 0.30 / 1000 = 450 bonds
        Result = min(50, 450) = 50
        """
        step = DV01BudgetStep(
            max_dd_pct=Decimal("0.05"),
            expected_max_rate_move_bps=200,
            max_single_position_pct=Decimal("0.30"),
        )
        result = step.compute_position_size(
            layer_equity=Decimal(1500000),
            bond_dv01_per_unit=Decimal("7.5"),
            current_portfolio_dv01=Decimal(0),
            face_value=Decimal(1000),
        )
        assert result == 50

    def test_limited_by_dv01_budget(self) -> None:
        """When existing portfolio DV01 is high, remaining budget limits sizing.

        max_dv01 = 1_000_000 * 0.05 / 200 = 250
        remaining_dv01 = 250 - 200 = 50
        max_by_dv01 = 50 / 10 = 5 bonds
        max_by_position = 1_000_000 * 0.30 / 1000 = 300 bonds
        Result = min(5, 300) = 5
        """
        step = DV01BudgetStep(
            max_dd_pct=Decimal("0.05"),
            expected_max_rate_move_bps=200,
            max_single_position_pct=Decimal("0.30"),
        )
        result = step.compute_position_size(
            layer_equity=Decimal(1000000),
            bond_dv01_per_unit=Decimal(10),
            current_portfolio_dv01=Decimal(200),
            face_value=Decimal(1000),
        )
        assert result == 5

    def test_limited_by_single_position_cap(self) -> None:
        """When DV01 budget is large but position cap is tight.

        max_dv01 = 10_000_000 * 0.05 / 200 = 2500
        remaining_dv01 = 2500 - 0 = 2500
        max_by_dv01 = 2500 / 0.5 = 5000 bonds
        max_by_position = 10_000_000 * 0.10 / 1000 = 1000 bonds (10% cap)
        Result = min(5000, 1000) = 1000
        """
        step = DV01BudgetStep(
            max_dd_pct=Decimal("0.05"),
            expected_max_rate_move_bps=200,
            max_single_position_pct=Decimal("0.10"),
        )
        result = step.compute_position_size(
            layer_equity=Decimal(10000000),
            bond_dv01_per_unit=Decimal("0.5"),
            current_portfolio_dv01=Decimal(0),
            face_value=Decimal(1000),
        )
        assert result == 1000

    def test_budget_exhausted_returns_zero(self) -> None:
        """When portfolio DV01 already at or above max, return 0.

        max_dv01 = 500_000 * 0.05 / 200 = 125
        remaining_dv01 = 125 - 130 = -5 (exhausted)
        """
        step = DV01BudgetStep(
            max_dd_pct=Decimal("0.05"),
            expected_max_rate_move_bps=200,
        )
        result = step.compute_position_size(
            layer_equity=Decimal(500000),
            bond_dv01_per_unit=Decimal(8),
            current_portfolio_dv01=Decimal(130),
            face_value=Decimal(1000),
        )
        assert result == 0

    def test_zero_dv01_per_unit_returns_zero(self) -> None:
        """Bond with zero DV01 should return 0 (would cause division by zero)."""
        step = DV01BudgetStep()
        result = step.compute_position_size(
            layer_equity=Decimal(1000000),
            bond_dv01_per_unit=Decimal(0),
            current_portfolio_dv01=Decimal(0),
            face_value=Decimal(1000),
        )
        assert result == 0

    def test_negative_dv01_per_unit_returns_zero(self) -> None:
        """Negative DV01 per unit should return 0."""
        step = DV01BudgetStep()
        result = step.compute_position_size(
            layer_equity=Decimal(1000000),
            bond_dv01_per_unit=Decimal(-5),
            current_portfolio_dv01=Decimal(0),
            face_value=Decimal(1000),
        )
        assert result == 0

    def test_custom_face_value(self) -> None:
        """Bonds with non-standard face value (e.g. 10_000 RUB OFZ).

        max_dv01 = 2_000_000 * 0.05 / 200 = 500
        remaining_dv01 = 500 - 0 = 500
        max_by_dv01 = 500 / 50 = 10 bonds
        max_by_position = 2_000_000 * 0.30 / 10_000 = 60 bonds
        Result = min(10, 60) = 10
        """
        step = DV01BudgetStep()
        result = step.compute_position_size(
            layer_equity=Decimal(2000000),
            bond_dv01_per_unit=Decimal(50),
            current_portfolio_dv01=Decimal(0),
            face_value=Decimal(10000),
        )
        assert result == 10

    def test_default_parameters(self) -> None:
        """Default parameters: 5% DD, 200bps, 30% single position.

        max_dv01 = 1_000_000 * 0.05 / 200 = 250
        remaining_dv01 = 250
        max_by_dv01 = 250 / 5 = 50
        max_by_position = 1_000_000 * 0.30 / 1000 = 300
        Result = 50
        """
        step = DV01BudgetStep()
        result = step.compute_position_size(
            layer_equity=Decimal(1000000),
            bond_dv01_per_unit=Decimal(5),
            current_portfolio_dv01=Decimal(0),
        )
        assert result == 50

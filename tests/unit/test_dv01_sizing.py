"""Unit tests for bond position sizing (DV01 + equal-weight)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.core.schemas import BondPositionRecord
from finalayze.risk.dv01_sizing import DV01BudgetStep, EqualWeightBondSizer


# ── BondPositionRecord tests ─────────────────────────────────────────────

ENTRY_YTM = Decimal("12.50")
ENTRY_PRICE = Decimal("1030.00")
ENTRY_CLEAN_PCT = Decimal("98.50")
RECORD_QTY = Decimal(10)


class TestBondPositionRecord:
    """Tests for BondPositionRecord frozen dataclass."""

    def test_stores_all_fields(self) -> None:
        record = BondPositionRecord(
            symbol="SU26244RMFS2",
            quantity=RECORD_QTY,
            entry_ytm_pct=ENTRY_YTM,
            entry_date=date(2026, 3, 14),
            entry_price=ENTRY_PRICE,
            entry_clean_pct=ENTRY_CLEAN_PCT,
            layer_id="core",
        )
        assert record.symbol == "SU26244RMFS2"
        assert record.quantity == RECORD_QTY
        assert record.entry_ytm_pct == ENTRY_YTM
        assert record.entry_date == date(2026, 3, 14)
        assert record.entry_price == ENTRY_PRICE
        assert record.entry_clean_pct == ENTRY_CLEAN_PCT
        assert record.layer_id == "core"

    def test_frozen(self) -> None:
        record = BondPositionRecord(
            symbol="SU26244RMFS2",
            quantity=RECORD_QTY,
            entry_ytm_pct=ENTRY_YTM,
            entry_date=date(2026, 3, 14),
            entry_price=ENTRY_PRICE,
            entry_clean_pct=ENTRY_CLEAN_PCT,
            layer_id="core",
        )
        import pytest

        with pytest.raises(AttributeError):
            record.symbol = "OTHER"  # type: ignore[misc]


# ── DV01 dirty-price + transaction costs tests ──────────────────────────

DIRTY_PRICE = Decimal("1030")
FACE_VALUE_STD = Decimal("1000")
TRANSACTION_COST = Decimal("5")


class TestDV01UnitCostFix:
    """Tests for DV01BudgetStep with unit_cost (dirty price) parameter."""

    def test_unit_cost_reduces_bonds_vs_face_value(self) -> None:
        """Dirty price 1030 should produce fewer bonds than face value 1000.

        Use small DV01 per unit so DV01 budget is not the binding constraint
        and the position-size cap (which uses unit_cost) dominates.
        max_by_position (face) = 1_500_000 * 0.30 / 1000 = 450
        max_by_position (dirty) = 1_500_000 * 0.30 / 1030 = 436
        """
        step = DV01BudgetStep(
            max_dd_pct=Decimal("0.05"),
            expected_max_rate_move_bps=200,
            max_single_position_pct=Decimal("0.30"),
            max_dv01_per_bond_pct=Decimal("1.00"),  # disable per-bond DV01 cap
        )
        bonds_face = step.compute_position_size(
            layer_equity=Decimal(1500000),
            bond_dv01_per_unit=Decimal("0.5"),
            current_portfolio_dv01=Decimal(0),
            unit_cost=FACE_VALUE_STD,
        )
        bonds_dirty = step.compute_position_size(
            layer_equity=Decimal(1500000),
            bond_dv01_per_unit=Decimal("0.5"),
            current_portfolio_dv01=Decimal(0),
            unit_cost=DIRTY_PRICE,
        )
        assert bonds_dirty < bonds_face

    def test_default_unit_cost_backward_compat(self) -> None:
        """Default unit_cost=1000 preserves backward compatibility."""
        step = DV01BudgetStep()
        result = step.compute_position_size(
            layer_equity=Decimal(1000000),
            bond_dv01_per_unit=Decimal(5),
            current_portfolio_dv01=Decimal(0),
        )
        # Same as existing test: should be 8
        expected_bonds = 8
        assert result == expected_bonds

    def test_transaction_costs_reduce_bonds(self) -> None:
        """Adding transaction costs should reduce the number of bonds."""
        step = DV01BudgetStep(
            max_dd_pct=Decimal("0.05"),
            expected_max_rate_move_bps=200,
            max_single_position_pct=Decimal("0.30"),
        )
        bonds_no_cost = step.compute_position_size(
            layer_equity=Decimal(1500000),
            bond_dv01_per_unit=Decimal("7.5"),
            current_portfolio_dv01=Decimal(0),
            unit_cost=FACE_VALUE_STD,
            transaction_costs_per_unit=Decimal(0),
        )
        bonds_with_cost = step.compute_position_size(
            layer_equity=Decimal(1500000),
            bond_dv01_per_unit=Decimal("7.5"),
            current_portfolio_dv01=Decimal(0),
            unit_cost=FACE_VALUE_STD,
            transaction_costs_per_unit=TRANSACTION_COST,
        )
        # With costs, position cap shrinks so fewer bonds
        assert bonds_with_cost <= bonds_no_cost


class TestEqualWeightUnitCostFix:
    """Tests for EqualWeightBondSizer with unit_cost parameter."""

    def test_unit_cost_reduces_bonds(self) -> None:
        """Dirty price 1030 should produce fewer bonds than face 1000."""
        sizer = EqualWeightBondSizer(n_symbols=4)
        bonds_face = sizer.compute_position_size(
            layer_equity=Decimal(675000),
            bond_dv01_per_unit=Decimal(5),
            current_portfolio_dv01=Decimal(0),
            unit_cost=FACE_VALUE_STD,
        )
        bonds_dirty = sizer.compute_position_size(
            layer_equity=Decimal(675000),
            bond_dv01_per_unit=Decimal(5),
            current_portfolio_dv01=Decimal(0),
            unit_cost=DIRTY_PRICE,
        )
        assert bonds_dirty < bonds_face

    def test_default_unit_cost_backward_compat(self) -> None:
        """Default unit_cost=1000 preserves backward compatibility."""
        sizer = EqualWeightBondSizer(n_symbols=4)
        result = sizer.compute_position_size(
            layer_equity=Decimal(675000),
            bond_dv01_per_unit=Decimal(5),
            current_portfolio_dv01=Decimal(100),
        )
        expected_bonds = 168
        assert result == expected_bonds


class TestDV01BudgetStep:
    """Tests for DV01BudgetStep.compute_position_size."""

    def test_basic_sizing(self) -> None:
        """1.5M equity, 5% DD, 200bps -> max_dv01 = 375.

        With DV01 per unit = 7.5 and no existing portfolio DV01:
            max_by_dv01 = 375 / 7.5 = 50 bonds
            per_bond_cap = 375 * 0.40 = 150, max_by_per_bond = 150 / 7.5 = 20 bonds
            max_by_position = 1_500_000 * 0.30 / 1000 = 450 bonds
        Result = min(50, 20, 450) = 20
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
            unit_cost=Decimal(1000),
        )
        assert result == 20

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
            unit_cost=Decimal(1000),
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
            unit_cost=Decimal(1000),
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
            unit_cost=Decimal(1000),
        )
        assert result == 0

    def test_zero_dv01_per_unit_returns_zero(self) -> None:
        """Bond with zero DV01 should return 0 (would cause division by zero)."""
        step = DV01BudgetStep()
        result = step.compute_position_size(
            layer_equity=Decimal(1000000),
            bond_dv01_per_unit=Decimal(0),
            current_portfolio_dv01=Decimal(0),
            unit_cost=Decimal(1000),
        )
        assert result == 0

    def test_negative_dv01_per_unit_returns_zero(self) -> None:
        """Negative DV01 per unit should return 0."""
        step = DV01BudgetStep()
        result = step.compute_position_size(
            layer_equity=Decimal(1000000),
            bond_dv01_per_unit=Decimal(-5),
            current_portfolio_dv01=Decimal(0),
            unit_cost=Decimal(1000),
        )
        assert result == 0

    def test_custom_face_value(self) -> None:
        """Bonds with non-standard face value (e.g. 10_000 RUB OFZ).

        max_dv01 = 2_000_000 * 0.05 / 500 = 200
        remaining_dv01 = 200 - 0 = 200
        max_by_dv01 = 200 / 50 = 4 bonds
        per_bond_cap = 200 * 0.40 = 80, max_by_per_bond = 80 / 50 = 1 bond
        max_by_position = 2_000_000 * 0.30 / 10_000 = 60 bonds
        Result = min(4, 1, 60) = 1
        """
        step = DV01BudgetStep()
        result = step.compute_position_size(
            layer_equity=Decimal(2000000),
            bond_dv01_per_unit=Decimal(50),
            current_portfolio_dv01=Decimal(0),
            unit_cost=Decimal(10000),
        )
        assert result == 1

    def test_default_parameters(self) -> None:
        """Default parameters: 5% DD, 500bps, 30% single position, 40% per-bond cap.

        max_dv01 = 1_000_000 * 0.05 / 500 = 100
        remaining_dv01 = 100
        max_by_dv01 = 100 / 5 = 20
        per_bond_cap = 100 * 0.40 = 40, max_by_per_bond = 40 / 5 = 8
        max_by_position = 1_000_000 * 0.30 / 1000 = 300
        Result = min(20, 8, 300) = 8
        """
        step = DV01BudgetStep()
        result = step.compute_position_size(
            layer_equity=Decimal(1000000),
            bond_dv01_per_unit=Decimal(5),
            current_portfolio_dv01=Decimal(0),
        )
        assert result == 8


LAYER_EQUITY = Decimal(100000)
FACE_VALUE = Decimal(1000)


def test_dv01_per_bond_cap_limits_single_position() -> None:
    """Per-bond DV01 cap (40%) prevents single issue consuming entire budget."""
    sizer = DV01BudgetStep(max_dv01_per_bond_pct=Decimal("0.40"))
    # Large DV01 per unit — without cap would consume entire budget
    result = sizer.compute_position_size(
        layer_equity=LAYER_EQUITY,
        bond_dv01_per_unit=Decimal("1.0"),
        current_portfolio_dv01=Decimal(0),
        unit_cost=FACE_VALUE,
    )
    # Max DV01 budget = 100000 * 0.05 / 500 = 10.0
    # Per-bond cap = 10.0 * 0.40 = 4.0
    # Max by DV01 cap = 4 bonds
    # Max by position = 100000 * 0.30 / 1000 = 30 bonds
    assert result == 4


class TestEqualWeightBondSizer:
    """Tests for EqualWeightBondSizer."""

    def test_equal_weight_4_symbols(self) -> None:
        """675K equity / 4 symbols = 168.75K each → 168 bonds."""
        sizer = EqualWeightBondSizer(n_symbols=4)
        result = sizer.compute_position_size(
            layer_equity=Decimal(675000),
            bond_dv01_per_unit=Decimal(5),  # ignored
            current_portfolio_dv01=Decimal(100),  # ignored
        )
        assert result == 168

    def test_cap_limits_size(self) -> None:
        """With 10% cap: 1M * 0.10 = 100K → 100 bonds < 1M/4=250K."""
        sizer = EqualWeightBondSizer(n_symbols=4, max_single_position_pct=Decimal("0.10"))
        result = sizer.compute_position_size(
            layer_equity=Decimal(1000000),
            bond_dv01_per_unit=Decimal(5),
            current_portfolio_dv01=Decimal(0),
        )
        assert result == 100

    def test_single_symbol(self) -> None:
        """One symbol gets full allocation (capped at 25%)."""
        sizer = EqualWeightBondSizer(n_symbols=1)
        result = sizer.compute_position_size(
            layer_equity=Decimal(500000),
            bond_dv01_per_unit=Decimal(0),
            current_portfolio_dv01=Decimal(0),
        )
        # min(500K/1, 500K*0.25) = 125K → 125 bonds
        assert result == 125

    def test_ignores_dv01_arguments(self) -> None:
        """DV01 args should be ignored — same result regardless of DV01 values."""
        sizer = EqualWeightBondSizer(n_symbols=4)
        r1 = sizer.compute_position_size(
            layer_equity=Decimal(400000),
            bond_dv01_per_unit=Decimal(0),
            current_portfolio_dv01=Decimal(0),
        )
        r2 = sizer.compute_position_size(
            layer_equity=Decimal(400000),
            bond_dv01_per_unit=Decimal(999),
            current_portfolio_dv01=Decimal(9999),
        )
        assert r1 == r2 == 100

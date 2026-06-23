"""Unit tests for budget-driven curve rescaling (Phase 78 P3-05).

Tests the correctness of the rescale logic: multiplicative scaling
using each leg's own curve[0], with edge cases.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.orchestration.budget_driver import _rescale_curve


class TestRescaleCurve:
    """Curve rescaling logic: multiplicative to target notional."""

    def test_rescale_simple_curve(self) -> None:
        """Rescale a simple curve multiplicatively."""
        curve = [
            (date(2026, 1, 1), Decimal(1000)),
            (date(2026, 1, 2), Decimal(1050)),
            (date(2026, 1, 3), Decimal(1100)),
        ]
        target = Decimal(10000)  # 10x the base
        result = _rescale_curve(curve, target)

        assert len(result) == 3
        assert result[0] == (date(2026, 1, 1), Decimal(10000))
        assert result[1] == (date(2026, 1, 2), Decimal(10500))
        assert result[2] == (date(2026, 1, 3), Decimal(11000))

    def test_rescale_with_decimal_precision(self) -> None:
        """Rescale preserves Decimal precision (no float loss)."""
        curve = [
            (date(2026, 1, 1), Decimal("2525.75")),  # MCFTR real index level
            (date(2026, 1, 2), Decimal("2550.25")),
        ]
        target = Decimal("100000.00")
        result = _rescale_curve(curve, target)

        # Scale factor: 100000 / 2525.75 ≈ 39.592
        scale = target / Decimal("2525.75")
        expected_0 = Decimal("2525.75") * scale
        expected_1 = Decimal("2550.25") * scale

        assert result[0][1] == expected_0
        assert result[1][1] == expected_1
        assert isinstance(result[0][1], Decimal)

    def test_rescale_down_to_lower_notional(self) -> None:
        """Rescale down: target < base."""
        curve = [
            (date(2026, 1, 1), Decimal(10000)),
            (date(2026, 1, 2), Decimal(10500)),
        ]
        target = Decimal(5000)  # half the base
        result = _rescale_curve(curve, target)

        assert result[0][1] == Decimal(5000)
        assert result[1][1] == Decimal(5250)

    def test_rescale_empty_curve_raises(self) -> None:
        """An empty curve fails loud rather than silently returning unscaled (WR-03)."""
        with pytest.raises(ValueError, match="cannot rescale"):
            _rescale_curve([], Decimal(100000))

    def test_rescale_zero_base_raises(self) -> None:
        """A zero base fails loud -- a silent unscaled return would corrupt the notional (WR-03)."""
        curve = [(date(2026, 1, 1), Decimal(0)), (date(2026, 1, 2), Decimal(100))]
        with pytest.raises(ValueError, match="cannot rescale"):
            _rescale_curve(curve, Decimal(100000))

    def test_rescale_single_value_curve(self) -> None:
        """Single-point curve rescales correctly."""
        curve = [(date(2026, 1, 1), Decimal(1000))]
        target = Decimal(50000)
        result = _rescale_curve(curve, target)

        assert len(result) == 1
        assert result[0][0] == date(2026, 1, 1)
        assert result[0][1] == Decimal(50000)

    def test_rescale_identity_when_target_equals_base(self) -> None:
        """Rescale to the same base value returns curve unchanged."""
        curve = [
            (date(2026, 1, 1), Decimal("2525.75")),
            (date(2026, 1, 2), Decimal("2550.25")),
        ]
        result = _rescale_curve(curve, Decimal("2525.75"))

        assert result == curve

    def test_rescale_maintains_date_ordering(self) -> None:
        """Rescale preserves date order."""
        curve = [
            (date(2026, 1, 1), Decimal(1000)),
            (date(2026, 1, 5), Decimal(1100)),
            (date(2026, 1, 10), Decimal(1200)),
        ]
        result = _rescale_curve(curve, Decimal(100000))

        dates = [d for d, _ in result]
        assert dates == [date(2026, 1, 1), date(2026, 1, 5), date(2026, 1, 10)]

"""Unit tests for volatility-adjusted position sizing (6B.7)."""

from __future__ import annotations

from decimal import Decimal

from finalayze.risk.position_sizer import compute_vol_adjusted_position_size

BASE_POSITION = Decimal(10000)


class TestVolAdjustedPositionSize:
    def test_vol_scaling_reduces_size_for_high_vol(self) -> None:
        """Asset vol = 0.40, target vol = 0.15 -> scale down ~0.375x."""
        result = compute_vol_adjusted_position_size(
            base_position=BASE_POSITION,
            target_vol=Decimal("0.15"),
            asset_vol=Decimal("0.40"),
        )
        # scale = 0.15 / 0.40 = 0.375
        expected = BASE_POSITION * Decimal("0.375")
        assert result == expected

    def test_vol_scaling_increases_size_for_low_vol(self) -> None:
        """Asset vol = 0.08, target vol = 0.15 -> scale up ~1.875x."""
        result = compute_vol_adjusted_position_size(
            base_position=BASE_POSITION,
            target_vol=Decimal("0.15"),
            asset_vol=Decimal("0.08"),
        )
        # scale = 0.15 / 0.08 = 1.875
        expected = BASE_POSITION * Decimal("1.875")
        assert result == expected

    def test_vol_scaling_clamped_to_bounds(self) -> None:
        """Extreme vol ratios are clamped to min_scale/max_scale."""
        # Very high vol -> scale would be 0.01, but min_scale=0.25
        result_low = compute_vol_adjusted_position_size(
            base_position=BASE_POSITION,
            target_vol=Decimal("0.01"),
            asset_vol=Decimal("1.00"),
        )
        assert result_low == BASE_POSITION * Decimal("0.25")

        # Very low vol -> scale would be 15.0, but max_scale=2.0
        result_high = compute_vol_adjusted_position_size(
            base_position=BASE_POSITION,
            target_vol=Decimal("1.50"),
            asset_vol=Decimal("0.10"),
        )
        assert result_high == BASE_POSITION * Decimal("2.0")

    def test_vol_scaling_zero_vol_returns_base(self) -> None:
        """Asset vol = 0 -> return base position unchanged."""
        result = compute_vol_adjusted_position_size(
            base_position=BASE_POSITION,
            target_vol=Decimal("0.15"),
            asset_vol=Decimal(0),
        )
        assert result == BASE_POSITION

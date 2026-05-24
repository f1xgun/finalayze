"""Tests for RC2 (pipeline floor) and RC3 (strategy-specific stops)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.backtest.config import (
    DEFAULT_STRATEGY_STOP_ATR,
    resolve_stop_atr_multiplier,
)
from finalayze.risk.position_sizing_pipeline import (
    HardCapsStep,
    PositionSizingPipeline,
    RegimeStep,
    SizingContext,
    VolTargetStep,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EQUITY = Decimal(100000)
_MAX_PCT = Decimal("0.20")
_MIN_POS_USD = Decimal(500)
_MIN_POS_RUB = Decimal(5000)
_TARGET_VOL = Decimal("0.15")


def _make_context(
    base_position: Decimal = Decimal(10000),
    regime_scale: Decimal = Decimal("1.0"),
    asset_vol: Decimal = Decimal("0.20"),
    min_position_size: Decimal = _MIN_POS_USD,
) -> SizingContext:
    return SizingContext(
        equity=_EQUITY,
        base_position=base_position,
        max_position_pct=_MAX_PCT,
        min_position_size=min_position_size,
        asset_vol=asset_vol,
        target_vol=_TARGET_VOL,
        regime_scale=regime_scale,
        correlation_scale=Decimal("1.0"),
    )


class TestRegimeStepFloor:
    """RegimeStep floor at 15% replaces the old pipeline-level floor."""

    def test_regime_floor_at_15pct(self) -> None:
        """regime_scale=0.05 is clamped to 0.15 by RegimeStep."""
        step = RegimeStep()
        context = _make_context(regime_scale=Decimal("0.05"))
        base_size = Decimal(10000)
        result = step.adjust(base_size, context)
        # Floor clamps 0.05 -> 0.15, so output = 10000 * 0.15 = 1500
        expected = Decimal("1500.0000")
        assert result == expected

    def test_regime_normal_passthrough(self) -> None:
        """regime_scale=0.70 passes through without clamping."""
        step = RegimeStep()
        context = _make_context(regime_scale=Decimal("0.70"))
        base_size = Decimal(10000)
        result = step.adjust(base_size, context)
        # No clamping: output = 10000 * 0.70 = 7000
        expected = Decimal("7000.0000")
        assert result == expected

    def test_no_pipeline_floor(self) -> None:
        """Pipeline floor no longer exists; cascading reduction can eliminate positions.

        With VolTarget 0.25x and RegimeStep 0.15 floor:
        1500 * 0.25 * 0.15 = 56.25, well below half_min (250) -> eliminated.
        The old pipeline floor would have saved this as 1500*0.15=225; now it returns 0.
        """
        pipeline = PositionSizingPipeline()
        context = _make_context(
            base_position=Decimal(1500),
            regime_scale=Decimal("0.05"),
            asset_vol=Decimal("0.60"),  # 4x target -> VolTarget clamps at 0.25
        )
        result = pipeline.compute(context)
        # 1500 * 0.25(vol) * 0.15(regime floor) = 56.25 < 250(half_min) -> 0
        assert result == Decimal(0)

    def test_hard_cap_still_enforced(self) -> None:
        """Hard cap on position size is still enforced."""
        pipeline = PositionSizingPipeline()
        context = _make_context(base_position=Decimal(50000))
        result = pipeline.compute(context)
        max_cap = _EQUITY * _MAX_PCT  # 20000
        assert result <= max_cap

    def test_guarded_roundup_works(self) -> None:
        """When size falls between 0.5*min and min, round up if Kelly was positive."""
        pipeline = PositionSizingPipeline()
        # base=2000, regime=0.15 -> RegimeStep output = 2000*0.75*0.15 = 225
        # Actually let's pick values that land in the roundup zone [250, 500)
        # base=3000, vol=0.15/0.20=0.75, regime=0.30 -> 3000 * 0.75 * 0.30 = 675
        # That's above min. Let's try:
        # base=1500, vol=target so ratio=1.0, regime=0.25 -> 1500*1.0*0.25=375
        # 375 is in [250, 500) and base(1500) > min(500) -> rounds up to 500
        context = _make_context(
            base_position=Decimal(1500),
            regime_scale=Decimal("0.25"),
            asset_vol=Decimal("0.15"),  # same as target_vol -> ratio = 1.0
        )
        result = pipeline.compute(context)
        assert result == _MIN_POS_USD

    def test_negative_expectancy_not_rounded_up(self) -> None:
        """When Kelly base is below min_position_size, don't force trades."""
        pipeline = PositionSizingPipeline()
        context = _make_context(
            base_position=Decimal(100),  # Below min_pos -> negative expectancy
            regime_scale=Decimal("1.0"),
        )
        result = pipeline.compute(context)
        # 100 * 0.75(vol) * 1.0(regime) = 75 < 500 -> eliminated
        assert result == Decimal(0)


class TestStrategyStopATR:
    """RC3: Strategy-specific stop-loss multipliers."""

    def test_momentum_stop_multiplier(self) -> None:
        """Momentum uses 2.5 ATR."""
        expected = 2.5
        assert DEFAULT_STRATEGY_STOP_ATR["momentum"] == expected

    def test_mean_reversion_wider_stop(self) -> None:
        """Mean reversion uses 3.5 ATR (wider than momentum)."""
        expected = 3.5
        assert DEFAULT_STRATEGY_STOP_ATR["mean_reversion"] == expected

    def test_rsi2_connors_tight_stop(self) -> None:
        """RSI2 uses 2.5 ATR (tight stop + short max_hold)."""
        expected = 2.5
        assert DEFAULT_STRATEGY_STOP_ATR["rsi2_connors"] == expected

    def test_resolve_us_segment(self) -> None:
        """US segments use base multiplier without uplift."""
        result = resolve_stop_atr_multiplier("momentum", segment_id="us_tech")
        assert result == Decimal("2.5")

    def test_resolve_ru_segment_uplift(self) -> None:
        """MOEX segments get 1.2x uplift due to higher volatility."""
        result = resolve_stop_atr_multiplier("momentum", segment_id="ru_blue_chips")
        assert result == Decimal("3.0")  # 2.5 * 1.2

    def test_resolve_unknown_strategy_uses_fallback(self) -> None:
        """Unknown strategy name uses 3.0 fallback."""
        result = resolve_stop_atr_multiplier("unknown_strategy")
        assert result == Decimal("3.0")

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            ("momentum", 2.5),
            ("dual_momentum", 3.0),
            ("mean_reversion", 3.5),
            ("ou_mean_reversion", 3.5),
            ("rsi2_connors", 2.5),
            ("pairs", 3.0),
            ("dividend_gap", 3.0),
        ],
    )
    def test_all_strategies_have_stop_atr(self, strategy: str, expected: float) -> None:
        """All active strategies have defined stop multipliers."""
        assert DEFAULT_STRATEGY_STOP_ATR[strategy] == expected

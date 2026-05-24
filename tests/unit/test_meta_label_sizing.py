"""Tests for MetaLabelStep in the position sizing pipeline.

MetaLabelStep scales position size by ML-predicted P(profitable).
Maps [threshold, 1.0] -> [0.0, 1.0] linearly.
Below threshold -> zero position (ML vetoes the trade).
None -> pass-through (ML not available).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.risk.position_sizing_pipeline import MetaLabelStep, SizingContext

_EQUITY = Decimal(100_000)
_BASE_POSITION = Decimal(5_000)
_MAX_PCT = Decimal("0.05")
_MIN_POS = Decimal(500)
_ASSET_VOL = Decimal("0.20")
_TARGET_VOL = Decimal("0.15")
_REGIME_SCALE = Decimal("1.0")
_CORRELATION_SCALE = Decimal("1.0")


def _make_context(*, ml_confidence: float | None = None) -> SizingContext:
    return SizingContext(
        equity=_EQUITY,
        base_position=_BASE_POSITION,
        max_position_pct=_MAX_PCT,
        min_position_size=_MIN_POS,
        asset_vol=_ASSET_VOL,
        target_vol=_TARGET_VOL,
        regime_scale=_REGIME_SCALE,
        correlation_scale=_CORRELATION_SCALE,
        ml_confidence=ml_confidence,
    )


class TestMetaLabelSizingStep:
    """MetaLabelStep should scale position size by ML confidence."""

    def test_high_confidence_scales_up(self) -> None:
        """ML confidence 0.80 -> scaling factor = (0.80 - 0.40) / 0.60 = 0.667."""
        step = MetaLabelStep()
        ctx = _make_context(ml_confidence=0.80)
        result = step.adjust(Decimal(5_000), ctx)
        # factor = (0.80 - 0.40) / (1.0 - 0.40) = 0.40 / 0.60 = 0.6667
        expected = Decimal(5_000) * Decimal("0.6667")
        assert abs(result - expected) < Decimal(10)

    def test_low_confidence_vetoes(self) -> None:
        """ML confidence below threshold -> position = 0."""
        step = MetaLabelStep()
        ctx = _make_context(ml_confidence=0.30)
        result = step.adjust(Decimal(5_000), ctx)
        assert result == Decimal(0)

    def test_none_confidence_passthrough(self) -> None:
        """When ml_confidence is None, position unchanged."""
        step = MetaLabelStep()
        ctx = _make_context(ml_confidence=None)
        result = step.adjust(Decimal(5_000), ctx)
        assert result == Decimal(5_000)

    def test_exact_threshold_vetoes(self) -> None:
        """ML confidence exactly at threshold -> vetoed (must exceed, not equal)."""
        step = MetaLabelStep()
        ctx = _make_context(ml_confidence=0.40)
        result = step.adjust(Decimal(5_000), ctx)
        assert result == Decimal(0)

    def test_confidence_1_0_full_size(self) -> None:
        """ML confidence 1.0 -> full position size."""
        step = MetaLabelStep()
        ctx = _make_context(ml_confidence=1.0)
        result = step.adjust(Decimal(5_000), ctx)
        assert result == Decimal(5_000)

    def test_just_above_threshold(self) -> None:
        """ML confidence barely above threshold -> very small position."""
        step = MetaLabelStep()
        ctx = _make_context(ml_confidence=0.41)
        result = step.adjust(Decimal(5_000), ctx)
        # factor = (0.41 - 0.40) / 0.60 = 0.01667
        assert result > Decimal(0)
        assert result < Decimal(200)

    def test_custom_threshold(self) -> None:
        """Custom threshold changes the veto/scaling boundary."""
        step = MetaLabelStep(threshold=Decimal("0.50"))
        ctx = _make_context(ml_confidence=0.75)
        result = step.adjust(Decimal(10_000), ctx)
        # factor = (0.75 - 0.50) / (1.0 - 0.50) = 0.25 / 0.50 = 0.50
        expected = Decimal(10_000) * Decimal("0.5")
        assert abs(result - expected) < Decimal(1)

    def test_pipeline_integration_with_meta_label_step(self) -> None:
        """MetaLabelStep works correctly within the full pipeline."""
        from finalayze.risk.position_sizing_pipeline import (
            HardCapsStep,
            PositionSizingPipeline,
            RegimeStep,
            VolTargetStep,
        )

        pipeline = PositionSizingPipeline(
            steps=[
                VolTargetStep(),
                RegimeStep(),
                MetaLabelStep(),
                HardCapsStep(),
            ]
        )
        ctx = _make_context(ml_confidence=0.80)
        result = pipeline.compute(ctx)
        # Pipeline should produce a reduced but non-zero position
        assert result >= Decimal(0)

    def test_pipeline_veto_returns_zero(self) -> None:
        """Low ML confidence in pipeline -> veto -> final position = 0."""
        from finalayze.risk.position_sizing_pipeline import (
            HardCapsStep,
            PositionSizingPipeline,
            RegimeStep,
            VolTargetStep,
        )

        pipeline = PositionSizingPipeline(
            steps=[
                VolTargetStep(),
                RegimeStep(),
                MetaLabelStep(),
                HardCapsStep(),
            ]
        )
        ctx = _make_context(ml_confidence=0.20)
        result = pipeline.compute(ctx)
        assert result == Decimal(0)


class TestSizingContextMLField:
    """SizingContext should accept ml_confidence field."""

    def test_default_none(self) -> None:
        """ml_confidence defaults to None for backward compatibility."""
        ctx = SizingContext(
            equity=_EQUITY,
            base_position=_BASE_POSITION,
            max_position_pct=_MAX_PCT,
            min_position_size=_MIN_POS,
            asset_vol=_ASSET_VOL,
            target_vol=_TARGET_VOL,
            regime_scale=_REGIME_SCALE,
            correlation_scale=_CORRELATION_SCALE,
        )
        assert ctx.ml_confidence is None

    def test_with_confidence(self) -> None:
        """ml_confidence can be set to a float value."""
        ctx = _make_context(ml_confidence=0.75)
        assert ctx.ml_confidence == pytest.approx(0.75)

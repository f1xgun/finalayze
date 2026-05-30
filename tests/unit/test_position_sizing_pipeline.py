"""Position-sizing pipeline unit tests for the Phase-60 CpiRiskOffStep (INTG-03).

Mirrors the existing CBRRegimeStep step tests (tests/unit/test_moex_sizing.py):
ru_-gated tier scale-down, non-ru_ passthrough, missing-data passthrough, and
Decimal-quantized output. Named so ``-k "cpi or sizing"`` discovers them.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from finalayze.risk.position_sizing_pipeline import (
    _FOUR_DP,
    CpiRiskOffStep,
    SizingContext,
)

_TEST_SIZE = Decimal("100000.0000")
_HIGH_INFLATION_SCALE = Decimal("0.6")
_NEUTRAL_SCALE = Decimal("1.0")
_HIGH_CPI = 0.10
_LOW_CPI = 0.05
_AT_CUT_CPI = 0.09  # exactly at the cut -> high-inflation tier


def _sizing_context() -> SizingContext:
    return SizingContext(
        equity=Decimal(1_000_000),
        base_position=_TEST_SIZE,
        max_position_pct=Decimal("0.20"),
        min_position_size=Decimal(500),
        asset_vol=Decimal("0.30"),
        target_vol=Decimal("0.20"),
        regime_scale=Decimal("1.0"),
        correlation_scale=Decimal("1.0"),
    )


class TestCpiRiskOffStepSizing:
    """CpiRiskOffStep scales ru_* positions down under high inflation."""

    def test_cpi_sizing_high_inflation_scales_down(self) -> None:
        step = CpiRiskOffStep(cpi_yoy_fraction=_HIGH_CPI, segment_id="ru_energy")
        result = step.adjust(_TEST_SIZE, _sizing_context())
        assert result == (_TEST_SIZE * _HIGH_INFLATION_SCALE).quantize(
            _FOUR_DP, rounding=ROUND_HALF_UP
        )

    def test_cpi_sizing_at_cut_scales_down(self) -> None:
        """CPI exactly at the 9% cut counts as high inflation (>= cut)."""
        step = CpiRiskOffStep(cpi_yoy_fraction=_AT_CUT_CPI, segment_id="ru_blue_chips")
        result = step.adjust(_TEST_SIZE, _sizing_context())
        assert result == (_TEST_SIZE * _HIGH_INFLATION_SCALE).quantize(
            _FOUR_DP, rounding=ROUND_HALF_UP
        )

    def test_cpi_sizing_low_inflation_neutral(self) -> None:
        step = CpiRiskOffStep(cpi_yoy_fraction=_LOW_CPI, segment_id="ru_energy")
        result = step.adjust(_TEST_SIZE, _sizing_context())
        assert result == (_TEST_SIZE * _NEUTRAL_SCALE).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)

    def test_cpi_sizing_non_ru_passthrough(self) -> None:
        step = CpiRiskOffStep(cpi_yoy_fraction=_HIGH_CPI, segment_id="us_tech")
        result = step.adjust(_TEST_SIZE, _sizing_context())
        assert result == _TEST_SIZE

    def test_cpi_sizing_missing_data_passthrough(self) -> None:
        step = CpiRiskOffStep(cpi_yoy_fraction=0.0, segment_id="ru_energy")
        result = step.adjust(_TEST_SIZE, _sizing_context())
        assert result == _TEST_SIZE

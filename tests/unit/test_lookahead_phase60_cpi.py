"""Phase-60 CPI risk-off look-ahead test suite (INTG-03, T-60-04/05).

Proves the rule-based ``CpiRiskOffStep`` (position-sizing pipeline) is point-in-time
correct: under a high-inflation regime it scales ``ru_*`` positions DOWN, it passes
through for non-``ru_`` segments and for missing CPI data, and — critically — it
resolves the CPI value PER-BAR via ``get_latest_published_cpi_month(context.bar_date)``
so a CPI month published AFTER the bar date can never influence that bar's sizing
(mirrors ``test_lookahead_phase59.py::TestLookaheadCpi`` publication-lag guard).

Every test is named ``test_lookahead_cpi_*`` / ``test_cpi_*`` so ``-k "lookahead or cpi"``
collects the whole suite. No live data / token is required.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from finalayze.risk.position_sizing_pipeline import (
    _FOUR_DP,
    CpiRiskOffStep,
    RegimeStep,
    SizingContext,
)

# ── Shared constants (ruff PLR2004: no magic numbers) ───────────────────────
_TEST_SIZE = Decimal("100000.0000")

# CpiRiskOffStep tier (documented in the step): cpi_yoy >= 0.09 -> 0.6x risk-off.
_HIGH_INFLATION_SCALE = Decimal("0.6")
_NEUTRAL_SCALE = Decimal("1.0")
_HIGH_CPI_FRACTION = 0.10  # 10% YoY -> above the 9% high-inflation cut
_LOW_CPI_FRACTION = 0.05  # 5% YoY -> below the cut, neutral
_MISSING_CPI = 0.0

# Per-bar look-ahead anchors. From cbr.CPI_PUBLICATION_DATES / _CPI_DATA:
#   2024-12 (YoY 9.5%, high) is published 2025-01-15.
#   2025-06 (YoY 9.1%, high) is published 2025-07-11.
# An early bar BEFORE a publication date must not see that month.
_EARLY_BAR = date(2025, 1, 10)  # before 2024-12 publication (2025-01-15)
_AFTER_DEC24_PUB = date(2025, 1, 20)  # after 2024-12 publication
_RU_SEG = "ru_energy"
_US_SEG = "us_tech"


def _make_context(bar_date: date | None = None) -> SizingContext:
    return SizingContext(
        equity=Decimal(1_000_000),
        base_position=_TEST_SIZE,
        max_position_pct=Decimal("0.20"),
        min_position_size=Decimal(500),
        asset_vol=Decimal("0.30"),
        target_vol=Decimal("0.20"),
        regime_scale=Decimal("1.0"),
        correlation_scale=Decimal("1.0"),
        bar_date=bar_date,
    )


class TestCpiRiskOffStep:
    """Fixed-value CpiRiskOffStep: scale / passthrough / missing-data behavior."""

    def test_cpi_high_inflation_scales_down(self) -> None:
        """High CPI (>= 9% cut) + ru_ segment -> size scaled DOWN to 0.6x."""
        step = CpiRiskOffStep(cpi_yoy_fraction=_HIGH_CPI_FRACTION, segment_id=_RU_SEG)
        result = step.adjust(_TEST_SIZE, _make_context())
        expected = (_TEST_SIZE * _HIGH_INFLATION_SCALE).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
        assert result == expected
        assert result < _TEST_SIZE

    def test_cpi_low_inflation_neutral(self) -> None:
        """Low CPI (< cut) + ru_ segment -> size unchanged (1.0x)."""
        step = CpiRiskOffStep(cpi_yoy_fraction=_LOW_CPI_FRACTION, segment_id=_RU_SEG)
        result = step.adjust(_TEST_SIZE, _make_context())
        expected = (_TEST_SIZE * _NEUTRAL_SCALE).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
        assert result == expected

    def test_cpi_non_ru_passthrough(self) -> None:
        """Non-ru_ segment -> size unchanged regardless of CPI level."""
        step = CpiRiskOffStep(cpi_yoy_fraction=_HIGH_CPI_FRACTION, segment_id=_US_SEG)
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == _TEST_SIZE

    def test_cpi_missing_data_passthrough(self) -> None:
        """cpi <= 0.0 (missing data) -> size unchanged (graceful degradation)."""
        step = CpiRiskOffStep(cpi_yoy_fraction=_MISSING_CPI, segment_id=_RU_SEG)
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == _TEST_SIZE

    def test_cpi_adds_no_floor_regime_floor_dominates(self) -> None:
        """RegimeStep's 0.15 floor stays the lower bound; CpiRiskOffStep only scales.

        Compose RegimeStep (regime_scale below floor -> floored at 0.15) then
        CpiRiskOffStep at high inflation: the CPI step scales the already-floored
        size by 0.6 and adds NO floor of its own, so the result is
        floor(0.15) * 0.6 < 0.15 -- i.e. the only floor in play is RegimeStep's.
        """
        ctx = SizingContext(
            equity=Decimal(1_000_000),
            base_position=_TEST_SIZE,
            max_position_pct=Decimal("0.20"),
            min_position_size=Decimal(500),
            asset_vol=Decimal("0.30"),
            target_vol=Decimal("0.20"),
            regime_scale=Decimal("0.01"),  # below 0.15 floor -> RegimeStep floors it
            correlation_scale=Decimal("1.0"),
        )
        after_regime = RegimeStep().adjust(_TEST_SIZE, ctx)
        floored = (_TEST_SIZE * Decimal("0.15")).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
        assert after_regime == floored
        step = CpiRiskOffStep(cpi_yoy_fraction=_HIGH_CPI_FRACTION, segment_id=_RU_SEG)
        after_cpi = step.adjust(after_regime, ctx)
        # CPI step scales below the regime floor (no second floor added).
        assert after_cpi < floored
        assert after_cpi == (after_regime * _HIGH_INFLATION_SCALE).quantize(
            _FOUR_DP, rounding=ROUND_HALF_UP
        )


class TestLookaheadCpiPerBar:
    """Per-bar CPI resolution via context.bar_date -- no future-month leak."""

    def test_lookahead_cpi_per_bar_unpublished_month_not_visible(self) -> None:
        """An early bar must resolve to the LATEST PUBLISHED month, never a future one.

        The step is built WITHOUT a fixed value (the backtest path), so it resolves
        CPI per-bar from context.bar_date. For a bar of 2025-01-10 the most recent
        published CPI month is 2024-11 (published 2024-12-13); 2024-12 (YoY 9.5%)
        is NOT yet published (publishes 2025-01-15) so it cannot drive the scale.
        """
        step = CpiRiskOffStep(segment_id=_RU_SEG)  # per-bar mode, no fixed value
        # 2024-11 YoY is 8.9% -> below the 9% cut -> neutral (1.0x).
        result = step.adjust(_TEST_SIZE, _make_context(bar_date=_EARLY_BAR))
        expected = (_TEST_SIZE * _NEUTRAL_SCALE).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
        assert result == expected

    def test_lookahead_cpi_per_bar_published_month_becomes_visible(self) -> None:
        """Once the bar date passes the publication date, the month drives the scale.

        For a bar of 2025-01-20 the latest published month is 2024-12 (YoY 9.5%,
        published 2025-01-15) -> above the 9% cut -> 0.6x risk-off. Same step,
        only the bar_date moved forward past the publication boundary.
        """
        step = CpiRiskOffStep(segment_id=_RU_SEG)  # per-bar mode
        result = step.adjust(_TEST_SIZE, _make_context(bar_date=_AFTER_DEC24_PUB))
        expected = (_TEST_SIZE * _HIGH_INFLATION_SCALE).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
        assert result == expected
        assert result < _TEST_SIZE

    def test_lookahead_cpi_per_bar_early_vs_late_differ_only_by_publication(self) -> None:
        """Early bar (neutral) and late bar (risk-off) differ purely by what was
        published on/before each bar_date -- proving per-bar resolution, no leak."""
        step = CpiRiskOffStep(segment_id=_RU_SEG)
        early = step.adjust(_TEST_SIZE, _make_context(bar_date=_EARLY_BAR))
        late = step.adjust(_TEST_SIZE, _make_context(bar_date=_AFTER_DEC24_PUB))
        assert early > late  # late bar saw the high-inflation 2024-12 print

    def test_lookahead_cpi_per_bar_no_bar_date_passthrough(self) -> None:
        """Per-bar mode with bar_date=None and no fixed value -> passthrough."""
        step = CpiRiskOffStep(segment_id=_RU_SEG)
        result = step.adjust(_TEST_SIZE, _make_context(bar_date=None))
        assert result == _TEST_SIZE

    def test_lookahead_cpi_per_bar_non_ru_passthrough(self) -> None:
        """Per-bar mode for a non-ru_ segment passes through regardless of bar_date."""
        step = CpiRiskOffStep(segment_id=_US_SEG)
        result = step.adjust(_TEST_SIZE, _make_context(bar_date=_AFTER_DEC24_PUB))
        assert result == _TEST_SIZE

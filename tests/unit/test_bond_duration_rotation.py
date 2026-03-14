"""Unit tests for bond_duration_rotation.classify_regime() — decision-based regime classifier."""

from __future__ import annotations

from decimal import Decimal

from finalayze.strategies.bond_duration_rotation import CBRRegime, classify_regime

# ── 1. last_cbr_decision == "hike" → HAWKISH regardless of gap ──────────────


class TestHikeDecision:
    """When the CBR hiked rates, regime must be HAWKISH."""

    def test_hike_with_negative_gap(self) -> None:
        """Hike decision dominates even when gap is deeply negative (proxy data)."""
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),  # gap = -0.50 (proxy scenario)
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hike",
        )
        assert result == CBRRegime.HAWKISH

    def test_hike_with_zero_gap(self) -> None:
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("16.00"),  # gap = 0
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hike",
        )
        assert result == CBRRegime.HAWKISH

    def test_hike_with_positive_gap(self) -> None:
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("17.00"),  # gap = +1.00
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hike",
        )
        assert result == CBRRegime.HAWKISH


# ── 2. last_cbr_decision == "cut" → DOVISH (unless CPI override) ────────────


class TestCutDecision:
    """When the CBR cut rates, regime must be DOVISH (absent CPI override)."""

    def test_cut_with_negative_gap(self) -> None:
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),  # gap = -0.50
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.DOVISH

    def test_cut_with_zero_gap(self) -> None:
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("16.00"),
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.DOVISH

    def test_cut_with_positive_gap(self) -> None:
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("17.00"),
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.DOVISH


# ── 3. last_cbr_decision == "hold" → RUONIA gap tiebreaker ──────────────────


class TestHoldDecisionTiebreaker:
    """When the CBR held at non-restrictive rates, use RUONIA gap as tiebreaker.

    Uses key_rate=10.00 (below 15% restrictive threshold) to isolate gap logic.
    """

    def test_hold_gap_below_negative_threshold_is_dovish(self) -> None:
        """gap < -0.75 → markets pricing in cut → DOVISH."""
        result = classify_regime(
            key_rate=Decimal("10.00"),
            ruonia_7d_avg=Decimal("9.20"),  # gap = -0.80
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.DOVISH

    def test_hold_gap_above_positive_threshold_is_hawkish(self) -> None:
        """gap > +0.75 → markets pricing in hike → HAWKISH."""
        result = classify_regime(
            key_rate=Decimal("10.00"),
            ruonia_7d_avg=Decimal("10.80"),  # gap = +0.80
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.HAWKISH

    def test_hold_gap_in_neutral_band_is_neutral(self) -> None:
        """gap in [-0.75, +0.75] → NEUTRAL."""
        result = classify_regime(
            key_rate=Decimal("10.00"),
            ruonia_7d_avg=Decimal("10.10"),  # gap = +0.10
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.NEUTRAL

    def test_hold_gap_exactly_at_negative_boundary_is_neutral(self) -> None:
        """gap == -0.75 → inside neutral band (not less than)."""
        result = classify_regime(
            key_rate=Decimal("10.00"),
            ruonia_7d_avg=Decimal("9.25"),  # gap = -0.75
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.NEUTRAL

    def test_hold_gap_exactly_at_positive_boundary_is_neutral(self) -> None:
        """gap == +0.75 → inside neutral band (not greater than)."""
        result = classify_regime(
            key_rate=Decimal("10.00"),
            ruonia_7d_avg=Decimal("10.75"),  # gap = +0.75
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.NEUTRAL

    def test_hold_gap_zero_is_neutral(self) -> None:
        result = classify_regime(
            key_rate=Decimal("10.00"),
            ruonia_7d_avg=Decimal("10.00"),  # gap = 0
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.NEUTRAL


# ── 4. CPI stagflation override → forces at least NEUTRAL ───────────────────


class TestCPIStagflationOverride:
    """CPI > 8% forces regime to at least NEUTRAL (never DOVISH)."""

    def test_cut_decision_with_high_cpi_forced_to_neutral(self) -> None:
        """cut → DOVISH, but CPI > 8% forces up to NEUTRAL."""
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy_latest_published=Decimal("9.0"),
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.NEUTRAL

    def test_hold_dovish_gap_with_high_cpi_forced_to_neutral(self) -> None:
        """hold + dovish gap → DOVISH, but CPI > 8% forces up to NEUTRAL.

        Uses key_rate=10.00 to avoid key-rate restrictive override.
        """
        result = classify_regime(
            key_rate=Decimal("10.00"),
            ruonia_7d_avg=Decimal("9.60"),  # gap = -0.40
            cpi_yoy_latest_published=Decimal("8.5"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.NEUTRAL

    def test_hike_with_high_cpi_stays_hawkish(self) -> None:
        """hike → HAWKISH; CPI override cannot lower it, HAWKISH > NEUTRAL."""
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy_latest_published=Decimal("10.0"),
            last_cbr_decision="hike",
        )
        assert result == CBRRegime.HAWKISH

    def test_hold_neutral_with_high_cpi_stays_neutral(self) -> None:
        """hold + neutral gap → NEUTRAL; CPI override is no-op (already NEUTRAL).

        Uses key_rate=10.00 to avoid key-rate restrictive override.
        """
        result = classify_regime(
            key_rate=Decimal("10.00"),
            ruonia_7d_avg=Decimal("10.00"),
            cpi_yoy_latest_published=Decimal("9.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.NEUTRAL

    def test_cpi_exactly_8_no_override(self) -> None:
        """CPI == 8.0 does not trigger override (threshold is strictly >8)."""
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy_latest_published=Decimal("8.0"),
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.DOVISH

    def test_cpi_below_8_no_override(self) -> None:
        """CPI < 8% → no override, DOVISH stays DOVISH."""
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy_latest_published=Decimal("7.0"),
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.DOVISH


# ── 5. Key-rate restrictive override → forces HAWKISH at high rates ──────────


class TestKeyRateRestrictiveOverride:
    """When key_rate >= 15% and CBR is not cutting, force HAWKISH."""

    def test_hold_at_16pct_forced_hawkish(self) -> None:
        """hold at 16% key rate → HAWKISH (not NEUTRAL)."""
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("16.00"),
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.HAWKISH

    def test_hold_at_21pct_forced_hawkish(self) -> None:
        """hold at 21% key rate → HAWKISH."""
        result = classify_regime(
            key_rate=Decimal("21.00"),
            ruonia_7d_avg=Decimal("20.50"),
            cpi_yoy_latest_published=Decimal("9.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.HAWKISH

    def test_hike_at_high_rate_stays_hawkish(self) -> None:
        """hike at 18% → already HAWKISH, override is no-op."""
        result = classify_regime(
            key_rate=Decimal("18.00"),
            ruonia_7d_avg=Decimal("17.50"),
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hike",
        )
        assert result == CBRRegime.HAWKISH

    def test_cut_at_high_rate_stays_dovish(self) -> None:
        """cut at 16% → DOVISH; key-rate override does NOT apply to cuts."""
        result = classify_regime(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.DOVISH

    def test_hold_at_14pct_no_override(self) -> None:
        """hold at 14% → below threshold, NEUTRAL stays NEUTRAL."""
        result = classify_regime(
            key_rate=Decimal("14.00"),
            ruonia_7d_avg=Decimal("14.00"),
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.NEUTRAL

    def test_hold_at_exactly_15pct_triggers(self) -> None:
        """hold at exactly 15% → at boundary, override triggers."""
        result = classify_regime(
            key_rate=Decimal("15.00"),
            ruonia_7d_avg=Decimal("15.00"),
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.HAWKISH

    def test_cut_at_15pct_exempt(self) -> None:
        """cut at exactly 15% → exempt from override, stays DOVISH."""
        result = classify_regime(
            key_rate=Decimal("15.00"),
            ruonia_7d_avg=Decimal("14.50"),
            cpi_yoy_latest_published=Decimal("5.0"),
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.DOVISH

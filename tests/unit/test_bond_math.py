"""Unit tests for bond math utilities (Layer 0).

Tests cover NKD, dirty price, YTM (Newton-Raphson), modified duration,
convexity, DV01, and price change estimation. Includes a real-world
validation test using OFZ 26244 parameters.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.core.bond_math import (
    convexity,
    dirty_price,
    dv01,
    modified_duration,
    nkd,
    price_change_estimate,
    ytm,
)

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────

# NKD test constants
COUPON_AMOUNT = Decimal("35.65")  # Semiannual coupon for 7.13% on 1000 face
DAYS_SINCE_LAST_COUPON = 91
COUPON_PERIOD_DAYS = 182
EXPECTED_NKD_MID_PERIOD = Decimal("17.83")  # 35.65 * 91 / 182 = 17.825 → 17.83

# Face value
FACE_VALUE = Decimal(1000)

# Clean price constants
CLEAN_PRICE_AT_PAR = Decimal("100.00")
CLEAN_PRICE_PREMIUM = Decimal("105.00")
CLEAN_PRICE_DISCOUNT = Decimal("85.50")
CLEAN_PRICE_DEEP_DISCOUNT = Decimal("70.00")

# Coupon rate constants
COUPON_RATE_STANDARD = Decimal("7.13")  # 7.13%
COUPON_RATE_HIGH = Decimal("11.25")  # 11.25% (OFZ 26244)
COUPON_RATE_LOW = Decimal("5.00")  # 5.00%

# Frequency
SEMIANNUAL = 2
ANNUAL = 1

# Dates
SETTLEMENT_2024 = date(2024, 6, 15)
MATURITY_5Y = date(2029, 6, 15)
MATURITY_10Y = date(2034, 6, 15)
MATURITY_NEAR = date(2024, 12, 15)  # ~6 months

# OFZ 26244 real-world constants
OFZ_26244_COUPON_RATE = Decimal("11.25")
OFZ_26244_MATURITY = date(2034, 3, 15)
OFZ_26244_CLEAN_PRICE = Decimal("85.80")
OFZ_26244_SETTLEMENT = date(2024, 6, 15)

# Duration / DV01 constants
KNOWN_DURATION = Decimal("5.00")
KNOWN_DIRTY_PRICE = Decimal("950.00")
EXPECTED_DV01 = Decimal("0.4750")  # 5.0 * 950 * 0.0001

# Price change constants
YIELD_CHANGE_100BPS = Decimal(100)
YIELD_CHANGE_NEG_50BPS = Decimal(-50)

# Precision tolerances
YTM_TOLERANCE = Decimal("0.50")  # Within 0.50% for approximate checks
DURATION_TOLERANCE = Decimal("0.50")  # Within 0.50 years
CONVEXITY_MIN = Decimal(0)


# ── NKD ──────────────────────────────────────────────────────────────────


class TestNkd:
    """Tests for accrued interest (NKD) calculation."""

    def test_mid_period_nkd(self) -> None:
        """NKD at mid-period should be proportional to days elapsed."""
        result = nkd(COUPON_AMOUNT, DAYS_SINCE_LAST_COUPON, COUPON_PERIOD_DAYS)
        assert result == EXPECTED_NKD_MID_PERIOD

    def test_nkd_at_period_start(self) -> None:
        """NKD should be zero at coupon payment date."""
        result = nkd(COUPON_AMOUNT, 0, COUPON_PERIOD_DAYS)
        assert result == Decimal("0.00")

    def test_nkd_at_period_end(self) -> None:
        """NKD at end of period should approximately equal coupon amount."""
        result = nkd(COUPON_AMOUNT, COUPON_PERIOD_DAYS, COUPON_PERIOD_DAYS)
        assert result == COUPON_AMOUNT

    def test_nkd_rounded_to_2dp(self) -> None:
        """NKD should be rounded to 2 decimal places."""
        result = nkd(Decimal("33.33"), 91, 182)
        # 33.33 * 91 / 182 = 16.665 -> rounds to 16.67
        assert result == Decimal("16.67")


# ── Dirty Price ──────────────────────────────────────────────────────────


class TestDirtyPrice:
    """Tests for dirty price (clean + NKD) calculation."""

    def test_at_par_with_nkd(self) -> None:
        """Dirty price at par plus NKD."""
        nkd_val = Decimal("17.82")
        result = dirty_price(CLEAN_PRICE_AT_PAR, nkd_val, FACE_VALUE)
        expected = Decimal("1017.82")  # 100/100 * 1000 + 17.82
        assert result == expected

    def test_deep_discount(self) -> None:
        """Dirty price for deep discount bond."""
        nkd_val = Decimal("10.00")
        result = dirty_price(CLEAN_PRICE_DEEP_DISCOUNT, nkd_val, FACE_VALUE)
        expected = Decimal("710.00")  # 70/100 * 1000 + 10
        assert result == expected

    def test_zero_nkd(self) -> None:
        """Dirty price with zero NKD equals clean price in RUB."""
        result = dirty_price(CLEAN_PRICE_DISCOUNT, Decimal(0), FACE_VALUE)
        expected = Decimal("855.00")  # 85.5/100 * 1000
        assert result == expected


# ── YTM ──────────────────────────────────────────────────────────────────


class TestYtm:
    """Tests for yield-to-maturity via Newton-Raphson."""

    def test_at_par_bond_ytm_equals_coupon_rate(self) -> None:
        """When bond is priced at par, YTM should approximately equal coupon rate."""
        result = ytm(
            clean_price_pct=CLEAN_PRICE_AT_PAR,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        diff = abs(result - COUPON_RATE_STANDARD)
        assert diff < YTM_TOLERANCE, f"YTM {result} too far from coupon rate {COUPON_RATE_STANDARD}"

    def test_premium_bond_ytm_less_than_coupon(self) -> None:
        """Premium bond should have YTM less than coupon rate."""
        result = ytm(
            clean_price_pct=CLEAN_PRICE_PREMIUM,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        assert result < COUPON_RATE_STANDARD, (
            f"Premium bond YTM {result} should be < coupon {COUPON_RATE_STANDARD}"
        )

    def test_discount_bond_ytm_greater_than_coupon(self) -> None:
        """Discount bond should have YTM greater than coupon rate."""
        result = ytm(
            clean_price_pct=CLEAN_PRICE_DISCOUNT,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        assert result > COUPON_RATE_STANDARD, (
            f"Discount bond YTM {result} should be > coupon {COUPON_RATE_STANDARD}"
        )

    def test_near_maturity_one_coupon(self) -> None:
        """Bond with ~1 coupon remaining should still converge."""
        result = ytm(
            clean_price_pct=Decimal("99.50"),
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_NEAR,
        )
        # Should produce a positive yield
        assert result > Decimal(0), f"Near-maturity YTM {result} should be positive"

    def test_convergence_failure_raises_value_error(self) -> None:
        """Newton-Raphson with max_iterations=0 should fail to converge."""
        with pytest.raises(ValueError, match="converge"):
            ytm(
                clean_price_pct=CLEAN_PRICE_DISCOUNT,
                coupon_rate=COUPON_RATE_STANDARD,
                face_value=FACE_VALUE,
                coupon_frequency=SEMIANNUAL,
                settlement_date=SETTLEMENT_2024,
                maturity_date=MATURITY_5Y,
                max_iterations=0,
            )

    def test_ytm_rounded_to_4dp(self) -> None:
        """YTM should be rounded to 4 decimal places."""
        result = ytm(
            clean_price_pct=CLEAN_PRICE_DISCOUNT,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        # Check it has at most 4 decimal places
        as_tuple = result.as_tuple()
        assert as_tuple.exponent >= -4, f"YTM {result} has more than 4 dp"  # type: ignore[operator]


# ── Modified Duration ────────────────────────────────────────────────────


class TestModifiedDuration:
    """Tests for modified duration calculation."""

    def test_higher_coupon_shorter_duration(self) -> None:
        """Higher coupon rate should produce shorter duration."""
        dur_low = modified_duration(
            ytm_pct=COUPON_RATE_LOW,
            coupon_rate=COUPON_RATE_LOW,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        dur_high = modified_duration(
            ytm_pct=COUPON_RATE_HIGH,
            coupon_rate=COUPON_RATE_HIGH,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        assert dur_low > dur_high, (
            f"Low-coupon duration {dur_low} should be > high-coupon {dur_high}"
        )

    def test_longer_maturity_longer_duration(self) -> None:
        """Longer maturity should produce longer duration."""
        dur_5y = modified_duration(
            ytm_pct=COUPON_RATE_STANDARD,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        dur_10y = modified_duration(
            ytm_pct=COUPON_RATE_STANDARD,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_10Y,
        )
        assert dur_10y > dur_5y, f"10Y duration {dur_10y} should be > 5Y duration {dur_5y}"

    def test_duration_less_than_maturity(self) -> None:
        """Modified duration should be less than time to maturity."""
        dur = modified_duration(
            ytm_pct=COUPON_RATE_STANDARD,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        years_to_maturity = Decimal(str((MATURITY_5Y - SETTLEMENT_2024).days / 365))
        assert dur < years_to_maturity, f"Duration {dur} should be < maturity {years_to_maturity}"

    def test_duration_rounded_to_2dp(self) -> None:
        """Duration should be rounded to 2 decimal places."""
        dur = modified_duration(
            ytm_pct=COUPON_RATE_STANDARD,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        as_tuple = dur.as_tuple()
        assert as_tuple.exponent >= -2, f"Duration {dur} has more than 2 dp"  # type: ignore[operator]


# ── DV01 ─────────────────────────────────────────────────────────────────


class TestDv01:
    """Tests for dollar value of one basis point."""

    def test_known_values(self) -> None:
        """DV01 for known duration and price."""
        result = dv01(KNOWN_DURATION, KNOWN_DIRTY_PRICE)
        assert result == EXPECTED_DV01

    def test_higher_duration_higher_dv01(self) -> None:
        """Higher duration should produce higher DV01."""
        dv01_short = dv01(Decimal("3.00"), KNOWN_DIRTY_PRICE)
        dv01_long = dv01(Decimal("7.00"), KNOWN_DIRTY_PRICE)
        assert dv01_long > dv01_short

    def test_dv01_rounded_to_4dp(self) -> None:
        """DV01 should be rounded to 4 decimal places."""
        result = dv01(KNOWN_DURATION, KNOWN_DIRTY_PRICE)
        as_tuple = result.as_tuple()
        assert as_tuple.exponent >= -4, f"DV01 {result} has more than 4 dp"  # type: ignore[operator]


# ── Convexity ────────────────────────────────────────────────────────────


class TestConvexity:
    """Tests for convexity calculation."""

    def test_positive_convexity(self) -> None:
        """All standard bonds should have positive convexity."""
        cx = convexity(
            ytm_pct=COUPON_RATE_STANDARD,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        assert cx > CONVEXITY_MIN, f"Convexity {cx} should be positive"

    def test_longer_bond_higher_convexity(self) -> None:
        """Longer bonds should have higher convexity."""
        cx_5y = convexity(
            ytm_pct=COUPON_RATE_STANDARD,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        cx_10y = convexity(
            ytm_pct=COUPON_RATE_STANDARD,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_10Y,
        )
        assert cx_10y > cx_5y, f"10Y convexity {cx_10y} should be > 5Y {cx_5y}"

    def test_convexity_rounded_to_2dp(self) -> None:
        """Convexity should be rounded to 2 decimal places."""
        cx = convexity(
            ytm_pct=COUPON_RATE_STANDARD,
            coupon_rate=COUPON_RATE_STANDARD,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=SETTLEMENT_2024,
            maturity_date=MATURITY_5Y,
        )
        as_tuple = cx.as_tuple()
        assert as_tuple.exponent >= -2, f"Convexity {cx} has more than 2 dp"  # type: ignore[operator]


# ── Price Change Estimate ────────────────────────────────────────────────


class TestPriceChangeEstimate:
    """Tests for estimated price change from yield shift."""

    def test_positive_yield_change_negative_price(self) -> None:
        """Rising yield should decrease bond price (duration effect dominates)."""
        cx_val = Decimal("30.00")  # Typical 5Y convexity
        result = price_change_estimate(
            KNOWN_DURATION,
            cx_val,
            KNOWN_DIRTY_PRICE,
            YIELD_CHANGE_100BPS,
        )
        assert result < Decimal(0), f"Price change {result} should be negative for +100bps"

    def test_negative_yield_change_positive_price(self) -> None:
        """Falling yield should increase bond price."""
        cx_val = Decimal("30.00")
        result = price_change_estimate(
            KNOWN_DURATION,
            cx_val,
            KNOWN_DIRTY_PRICE,
            YIELD_CHANGE_NEG_50BPS,
        )
        assert result > Decimal(0), f"Price change {result} should be positive for -50bps"

    def test_convexity_dampens_loss(self) -> None:
        """With convexity, actual loss should be less than linear duration estimate."""
        cx_val = Decimal("30.00")
        duration_only_loss = -KNOWN_DURATION * Decimal("0.01") * KNOWN_DIRTY_PRICE
        actual_change = price_change_estimate(
            KNOWN_DURATION,
            cx_val,
            KNOWN_DIRTY_PRICE,
            YIELD_CHANGE_100BPS,
        )
        # Convexity adds a positive term, so actual loss is less negative
        assert actual_change > duration_only_loss, (
            f"With convexity, loss {actual_change} should be less than linear {duration_only_loss}"
        )

    def test_100bps_on_5y_duration_approx_5pct(self) -> None:
        """100bps on 5Y duration ≈ -5% price change (before convexity)."""
        cx_val = Decimal("30.00")
        result = price_change_estimate(
            KNOWN_DURATION,
            cx_val,
            KNOWN_DIRTY_PRICE,
            YIELD_CHANGE_100BPS,
        )
        pct_change = result / KNOWN_DIRTY_PRICE * Decimal(100)
        # Should be around -5% (slightly less negative due to convexity)
        assert Decimal(-6) < pct_change < Decimal(-4), (
            f"Price change % {pct_change} should be around -5%"
        )


# ── OFZ 26244 Real-World Validation ─────────────────────────────────────


class TestOfz26244Validation:
    """Approximate validation against OFZ 26244 bond parameters.

    Coupon 11.25%, maturity 2034-03-15, face 1000 RUB, clean ~85.80%.
    Expected: YTM ~14-16%, modified duration ~4-5Y.
    """

    OFZ_YTM_LOW = Decimal("13.50")
    OFZ_YTM_HIGH = Decimal("16.50")
    OFZ_DURATION_LOW = Decimal("3.50")
    OFZ_DURATION_HIGH = Decimal("6.00")

    def test_ofz_ytm_in_expected_range(self) -> None:
        """OFZ 26244 YTM should be approximately 14-16%."""
        result = ytm(
            clean_price_pct=OFZ_26244_CLEAN_PRICE,
            coupon_rate=OFZ_26244_COUPON_RATE,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=OFZ_26244_SETTLEMENT,
            maturity_date=OFZ_26244_MATURITY,
        )
        assert self.OFZ_YTM_LOW < result < self.OFZ_YTM_HIGH, (
            f"OFZ 26244 YTM {result}% outside expected range "
            f"[{self.OFZ_YTM_LOW}, {self.OFZ_YTM_HIGH}]"
        )

    def test_ofz_duration_in_expected_range(self) -> None:
        """OFZ 26244 modified duration should be approximately 4-5Y."""
        ytm_val = ytm(
            clean_price_pct=OFZ_26244_CLEAN_PRICE,
            coupon_rate=OFZ_26244_COUPON_RATE,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=OFZ_26244_SETTLEMENT,
            maturity_date=OFZ_26244_MATURITY,
        )
        dur = modified_duration(
            ytm_pct=ytm_val,
            coupon_rate=OFZ_26244_COUPON_RATE,
            face_value=FACE_VALUE,
            coupon_frequency=SEMIANNUAL,
            settlement_date=OFZ_26244_SETTLEMENT,
            maturity_date=OFZ_26244_MATURITY,
        )
        assert self.OFZ_DURATION_LOW < dur < self.OFZ_DURATION_HIGH, (
            f"OFZ 26244 duration {dur}Y outside expected range "
            f"[{self.OFZ_DURATION_LOW}, {self.OFZ_DURATION_HIGH}]"
        )

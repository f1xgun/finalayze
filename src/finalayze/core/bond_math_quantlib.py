"""QuantLib wrapper for bond math (Layer 0 -- pure computation, no I/O).

Provides pricing for bond types that bond_math.py cannot handle:
- OFZ-PK floating-rate bonds (via QuantLib FloatingRateBond + RUONIA curve)
- OFZ-AD amortizing bonds (via custom cashflow construction)
- Effective duration via +/-25bps rate shock (works for any bond type)

Also provides QuantLib-based fixed-bond pricing for cross-validation
against the Newton-Raphson implementation in bond_math.py.

Day-count convention: Actual/365 Fixed (OFZ / Russian market standard).
Calendar: ql.Russia() for schedule generation.

Price convention: MOEX quotes bond prices as percentage of face value
(e.g. 85.50% means the bond trades at 855 RUB for 1000 RUB face).
All clean_price_pct parameters follow this convention.
"""

from __future__ import annotations

import contextlib
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import QuantLib as ql  # noqa: N813

_QUANT_2DP = Decimal("0.01")
_QUANT_4DP = Decimal("0.0001")


# ---------------------------------------------------------------------------
# Date conversion helpers
# ---------------------------------------------------------------------------


def to_ql_date(d: date) -> ql.Date:
    """Convert Python date to QuantLib Date.

    QuantLib Date(day, month, year) uses integer month (1-12).
    """
    return ql.Date(d.day, d.month, d.year)


def from_ql_date(d: ql.Date) -> date:
    """Convert QuantLib Date to Python date."""
    return date(d.year(), d.month(), d.dayOfMonth())


# ---------------------------------------------------------------------------
# Yield curve construction
# ---------------------------------------------------------------------------


def build_ruonia_curve(
    settlement_date: date,
    ruonia_rate: float,
) -> ql.YieldTermStructureHandle:
    """Build a flat RUONIA forward curve (MVP simplification).

    Args:
        settlement_date: Evaluation date.
        ruonia_rate: Current RUONIA rate as decimal (e.g. 0.21 for 21%).

    Returns:
        ql.YieldTermStructureHandle wrapping a FlatForward curve.
    """
    ql_settlement = to_ql_date(settlement_date)
    day_count = ql.Actual365Fixed()
    curve = ql.FlatForward(ql_settlement, ruonia_rate, day_count)
    return ql.YieldTermStructureHandle(curve)


# ---------------------------------------------------------------------------
# Fixed-rate bond (QuantLib, for cross-validation)
# ---------------------------------------------------------------------------


def price_fixed_bond_ql(
    settlement_date: date,
    maturity_date: date,
    face_value: Decimal,
    coupon_rate: Decimal,
    coupon_frequency: int,
    clean_price_pct: Decimal,
) -> tuple[Decimal, Decimal]:
    """Price a fixed-rate bond using QuantLib.

    Uses ql.FixedRateBond + bondYield for YTM, BondFunctions for duration.
    For cross-validation against bond_math.py Newton-Raphson.

    Args:
        settlement_date: Trade settlement date.
        maturity_date: Bond maturity date.
        face_value: Face value per bond (e.g. 1000 RUB).
        coupon_rate: Annual coupon rate as % (e.g. 7.10 for 7.10%).
        coupon_frequency: Coupons per year (2 for semiannual).
        clean_price_pct: Clean price as % of face (e.g. 85.50).

    Returns:
        (ytm_pct, modified_duration) where ytm_pct is annual YTM as %
        (e.g. 15.50 for 15.50%) and modified_duration is in years.
    """
    ql_settlement = to_ql_date(settlement_date)
    ql_maturity = to_ql_date(maturity_date)
    ql.Settings.instance().evaluationDate = ql_settlement

    calendar = ql.Russia()
    day_count = ql.Actual365Fixed()
    fv = float(face_value)
    cr = float(coupon_rate) / 100.0

    schedule = ql.Schedule(
        ql_settlement,
        ql_maturity,
        ql.Period(int(12 / coupon_frequency), ql.Months),
        calendar,
        ql.ModifiedFollowing,
        ql.ModifiedFollowing,
        ql.DateGeneration.Backward,
        False,
    )

    bond = ql.FixedRateBond(
        settlementDays=1,
        faceAmount=fv,
        schedule=schedule,
        coupons=[cr],
        paymentDayCounter=day_count,
    )

    # Compute YTM from clean price
    # QuantLib uses clean price as % of face (same convention as MOEX)
    bond_price = ql.BondPrice(float(clean_price_pct), ql.BondPrice.Clean)
    ytm_decimal = bond.bondYield(
        bond_price,
        day_count,
        ql.Compounded,
        ql.Semiannual,
    )
    ytm_pct = Decimal(str(ytm_decimal * 100.0)).quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)

    # Compute modified duration
    mod_dur = ql.BondFunctions.duration(
        bond,
        ytm_decimal,
        day_count,
        ql.Compounded,
        ql.Semiannual,
        ql.Duration.Modified,
    )
    mod_dur_dec = Decimal(str(mod_dur)).quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)

    return ytm_pct, mod_dur_dec


# ---------------------------------------------------------------------------
# Floating-rate bond (OFZ-PK)
# ---------------------------------------------------------------------------


def price_floating_rate_bond(
    settlement_date: date,
    maturity_date: date,
    face_value: Decimal,
    spread: Decimal,
    ruonia_rate: float,
    coupon_frequency: int = 2,
) -> tuple[float, float]:
    """Price an OFZ-PK floating-rate bond using QuantLib.

    Uses a flat RUONIA forward curve (MVP simplification).

    Args:
        settlement_date: Trade settlement date.
        maturity_date: Bond maturity date.
        face_value: Face value per bond (e.g. 1000 RUB).
        spread: Spread over RUONIA as decimal (e.g. 0.013 for 130bps).
        ruonia_rate: Current RUONIA rate as decimal (e.g. 0.21 for 21%).
        coupon_frequency: Coupons per year (2 for semiannual).

    Returns:
        (clean_price, ytm) where clean_price is in RUB and ytm is decimal
        (e.g. 0.223 for 22.3%).
    """
    ql_settlement = to_ql_date(settlement_date)
    ql_maturity = to_ql_date(maturity_date)
    ql.Settings.instance().evaluationDate = ql_settlement

    calendar = ql.Russia()
    day_count = ql.Actual365Fixed()
    fv = float(face_value)
    spread_float = float(spread)

    # Build flat RUONIA forward curve
    ruonia_curve = ql.FlatForward(ql_settlement, ruonia_rate, day_count)
    ruonia_handle = ql.YieldTermStructureHandle(ruonia_curve)

    # Create RUONIA overnight index
    ruonia_index = ql.OvernightIndex(
        "RUONIA", 1, ql.RUBCurrency(), calendar, day_count, ruonia_handle
    )

    # Add past fixings so QuantLib can price the current coupon period.
    # We use the flat rate as a proxy for all historical fixings (MVP).
    # Walk backward from settlement to provide enough fixings.
    fixing_date = ql_settlement - ql.Period(1, ql.Years)
    end_date = ql_settlement
    while fixing_date <= end_date:
        if calendar.isBusinessDay(fixing_date):
            with contextlib.suppress(RuntimeError):
                ruonia_index.addFixing(fixing_date, ruonia_rate)
        fixing_date = fixing_date + ql.Period(1, ql.Days)

    # Bond schedule
    schedule = ql.Schedule(
        ql_settlement,
        ql_maturity,
        ql.Period(int(12 / coupon_frequency), ql.Months),
        calendar,
        ql.ModifiedFollowing,
        ql.ModifiedFollowing,
        ql.DateGeneration.Backward,
        False,
    )

    bond = ql.FloatingRateBond(
        settlementDays=1,
        faceAmount=fv,
        schedule=schedule,
        index=ruonia_index,
        paymentDayCounter=day_count,
        spreads=[spread_float],
    )

    # Pricing engine
    bond.setPricingEngine(ql.DiscountingBondEngine(ruonia_handle))

    clean_price = bond.cleanPrice()
    ytm_val = bond.bondYield(day_count, ql.Compounded, ql.Semiannual)

    return clean_price, ytm_val


# ---------------------------------------------------------------------------
# Amortizing fixed-rate bond (OFZ-AD)
# ---------------------------------------------------------------------------


def price_amortizing_bond(
    settlement_date: date,
    maturity_date: date,
    face_value: Decimal,
    coupon_rate: Decimal,
    coupon_frequency: int,
    clean_price_pct: Decimal,
    amortization_schedule: list[tuple[date, Decimal]],
) -> tuple[Decimal, Decimal]:
    """Price an amortizing fixed-rate bond.

    Builds a custom cashflow schedule that reflects decreasing nominal
    at each amortization event.

    Args:
        settlement_date: Trade settlement date.
        maturity_date: Bond maturity date.
        face_value: Initial face value per bond (e.g. 1000 RUB).
        coupon_rate: Annual coupon rate as % (e.g. 8.00 for 8.00%).
        coupon_frequency: Coupons per year (2 for semiannual).
        clean_price_pct: Clean price as % of current face (e.g. 95.00).
        amortization_schedule: List of (date, remaining_nominal_pct) pairs.
            remaining_nominal_pct is % of original face (e.g. 80.00 for 80%).

    Returns:
        (ytm_pct, modified_duration) where ytm_pct is annual YTM as %
        and modified_duration is effective duration in years.
    """
    ql_settlement = to_ql_date(settlement_date)
    ql_maturity = to_ql_date(maturity_date)
    ql.Settings.instance().evaluationDate = ql_settlement

    calendar = ql.Russia()
    day_count = ql.Actual365Fixed()
    fv = float(face_value)
    cr = float(coupon_rate) / 100.0

    # Build amortization notional schedule
    # Sort amortization events by date
    amort_sorted = sorted(amortization_schedule, key=lambda x: x[0])

    # Generate coupon schedule
    schedule = ql.Schedule(
        ql_settlement,
        ql_maturity,
        ql.Period(int(12 / coupon_frequency), ql.Months),
        calendar,
        ql.ModifiedFollowing,
        ql.ModifiedFollowing,
        ql.DateGeneration.Backward,
        False,
    )

    # Build notionals array: one per coupon period
    # Start with full face value, reduce at amortization dates
    notionals = []
    current_nominal = fv
    amort_idx = 0

    for i in range(len(schedule) - 1):
        period_start = schedule[i]
        # Apply any amortization events that fall on or before this period start
        while amort_idx < len(amort_sorted):
            amort_date = to_ql_date(amort_sorted[amort_idx][0])
            if amort_date <= period_start:
                remaining_pct = float(amort_sorted[amort_idx][1]) / 100.0
                current_nominal = fv * remaining_pct
                amort_idx += 1
            else:
                break
        notionals.append(current_nominal)

    if not notionals:
        notionals = [fv]

    # Create amortizing bond
    bond = ql.AmortizingFixedRateBond(
        settlementDays=1,
        notionals=notionals,
        schedule=schedule,
        coupons=[cr],
        accrualDayCounter=day_count,
    )

    # Compute YTM from clean price
    # QuantLib uses clean price as % of face (same convention as MOEX)
    bond_price = ql.BondPrice(float(clean_price_pct), ql.BondPrice.Clean)
    ytm_decimal = bond.bondYield(
        bond_price,
        day_count,
        ql.Compounded,
        ql.Semiannual,
    )
    ytm_pct = Decimal(str(ytm_decimal * 100.0)).quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)

    # Effective duration via rate shock (more appropriate for amortizing bonds)
    curve = ql.FlatForward(ql_settlement, ytm_decimal, day_count)
    curve_handle = ql.YieldTermStructureHandle(curve)
    bond.setPricingEngine(ql.DiscountingBondEngine(curve_handle))

    eff_dur = effective_duration_rate_shock(bond, curve_handle)
    eff_dur_dec = Decimal(str(eff_dur)).quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)

    return ytm_pct, eff_dur_dec


# ---------------------------------------------------------------------------
# Effective duration via rate shock
# ---------------------------------------------------------------------------


def effective_duration_rate_shock(
    bond: ql.Bond,
    yield_curve_handle: ql.YieldTermStructureHandle,
    shock_bps: int = 25,
) -> float:
    """Compute effective duration via +/-N bps parallel rate shock.

    Reprices the bond with shifted yield curves and computes numerical
    duration: (P_down - P_up) / (2 * P_base * dy).

    Works for any bond type (fixed, floating, amortizing).

    Args:
        bond: QuantLib bond instrument (already has pricing engine set).
        yield_curve_handle: The yield curve used for pricing.
        shock_bps: Shock size in basis points (default 25).

    Returns:
        Effective duration in years.
    """
    # Get the underlying curve
    base_curve = yield_curve_handle.currentLink()
    ref_date = base_curve.referenceDate()
    day_count = ql.Actual365Fixed()

    # Base price
    base_price = bond.cleanPrice()

    shock_decimal = shock_bps / 10000.0

    # Get base rate from curve at a reference point
    base_rate = base_curve.zeroRate(
        ref_date + ql.Period(1, ql.Years), day_count, ql.Continuous
    ).rate()

    # Shock up
    up_curve = ql.FlatForward(ref_date, base_rate + shock_decimal, day_count)
    up_handle = ql.YieldTermStructureHandle(up_curve)
    bond.setPricingEngine(ql.DiscountingBondEngine(up_handle))
    price_up = bond.cleanPrice()

    # Shock down
    dn_curve = ql.FlatForward(ref_date, base_rate - shock_decimal, day_count)
    dn_handle = ql.YieldTermStructureHandle(dn_curve)
    bond.setPricingEngine(ql.DiscountingBondEngine(dn_handle))
    price_dn = bond.cleanPrice()

    # Restore original engine
    bond.setPricingEngine(ql.DiscountingBondEngine(yield_curve_handle))

    # Numerical duration
    dy = 2 * shock_decimal
    if base_price == 0.0:
        return 0.0

    return float((price_dn - price_up) / (base_price * dy))

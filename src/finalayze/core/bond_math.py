"""Bond math utilities (Layer 0 -- pure computation, no I/O).

YTM via Newton-Raphson, modified duration, convexity, DV01, NKD.
Precision: YTM to 4 dp, duration to 2 dp, DV01 to 4 dp.

Day-count convention: actual/365 (OFZ / Russian market standard).
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HUNDRED = Decimal(100)
_BASIS_POINT = Decimal("0.0001")
_DAYS_PER_YEAR = 365.0
_DERIV_EPSILON = 1e-15

_QUANT_2DP = Decimal("0.01")
_QUANT_4DP = Decimal("0.0001")


# ---------------------------------------------------------------------------
# Coupon schedule helper
# ---------------------------------------------------------------------------


def _remaining_coupon_dates(
    settlement_date: date,
    maturity_date: date,
    coupon_frequency: int,
) -> list[tuple[date, float]]:
    """Generate remaining coupon dates and time-to-coupon in years.

    Works backward from *maturity_date* by coupon_frequency intervals to find
    all future coupon dates strictly after *settlement_date*.

    Returns:
        List of ``(coupon_date, years_from_settlement)`` ordered chronologically.
    """
    months_per_coupon = 12 // coupon_frequency
    dates: list[tuple[date, float]] = []

    # Walk backward from maturity
    coupon_date = maturity_date
    while coupon_date > settlement_date:
        days_to_coupon = (coupon_date - settlement_date).days
        years = days_to_coupon / _DAYS_PER_YEAR
        dates.append((coupon_date, years))

        # Step back one coupon period
        month = coupon_date.month - months_per_coupon
        year = coupon_date.year
        while month <= 0:
            month += 12
            year -= 1
        coupon_date = date(year, month, coupon_date.day)

    dates.sort(key=lambda x: x[0])
    return dates


# ---------------------------------------------------------------------------
# NKD (accrued interest)
# ---------------------------------------------------------------------------


def nkd(
    coupon_amount: Decimal,
    days_since_last_coupon: int,
    coupon_period_days: int,
    *,
    day_count: str = "actual/365",
) -> Decimal:
    """Compute accrued interest (NKD) for a bond.

    ``NKD = coupon_amount * adjusted_days / adjusted_period``

    Args:
        coupon_amount: Coupon payment per bond (RUB).
        days_since_last_coupon: Calendar days since last coupon payment.
        coupon_period_days: Total days in the coupon period.
        day_count: Day-count convention. Supported: "actual/365" (default), "30/360".

    Returns:
        NKD in RUB per bond, rounded to 2 decimal places.
    """
    if day_count == "30/360":
        # Under 30/360 convention, approximate months from actual days
        # and use 30-day months / 360-day year
        months_elapsed = round(days_since_last_coupon / 30.4375)
        months_in_period = round(coupon_period_days / 30.4375)
        adj_days = months_elapsed * 30
        adj_period = months_in_period * 30
        if adj_period == 0:
            return Decimal("0.00")
        result = coupon_amount * Decimal(adj_days) / Decimal(adj_period)
    else:
        # actual/365 (default, existing behavior)
        result = coupon_amount * Decimal(days_since_last_coupon) / Decimal(coupon_period_days)
    return result.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Dirty price
# ---------------------------------------------------------------------------


def dirty_price(
    clean_price_pct: Decimal,
    nkd_per_bond: Decimal,
    face_value: Decimal,
) -> Decimal:
    """Compute dirty price (clean + NKD) in RUB.

    ``dirty = (clean_price_pct / 100) * face_value + nkd_per_bond``
    """
    return (clean_price_pct / _HUNDRED) * face_value + nkd_per_bond


# ---------------------------------------------------------------------------
# YTM (Newton-Raphson)
# ---------------------------------------------------------------------------


def ytm(
    clean_price_pct: Decimal,
    coupon_rate: Decimal,
    face_value: Decimal,
    coupon_frequency: int,
    settlement_date: date,
    maturity_date: date,
    max_iterations: int = 100,
    tolerance: Decimal = Decimal("0.00001"),
) -> Decimal:
    """Compute yield-to-maturity using Newton-Raphson.

    Uses float arithmetic internally for performance, then converts back to
    :class:`~decimal.Decimal` at the end.

    Args:
        clean_price_pct: Clean price as % of face (e.g. ``85.50``).
        coupon_rate: Annual coupon rate as % (e.g. ``7.10`` for 7.10%).
        face_value: Face value per bond (1000 for OFZ).
        coupon_frequency: Coupons per year (2 for semiannual).
        settlement_date: Trade settlement date (T+1 for MOEX).
        maturity_date: Bond maturity date.
        max_iterations: Newton-Raphson iteration limit.
        tolerance: Convergence tolerance.

    Returns:
        Annual YTM as % (e.g. ``15.50`` for 15.50%), rounded to 4 dp.

    Raises:
        ValueError: If Newton-Raphson fails to converge.
    """
    schedule = _remaining_coupon_dates(settlement_date, maturity_date, coupon_frequency)
    if not schedule:
        msg = "No remaining coupon dates; bond may have already matured"
        raise ValueError(msg)

    # Convert to float for fast iteration
    fv = float(face_value)
    market_price = float(clean_price_pct) / 100.0 * fv
    coupon_payment = float(coupon_rate) / 100.0 * fv / coupon_frequency
    tol = float(tolerance)

    # Time fractions (in years) for each cash flow
    times = [t for _, t in schedule]

    # Initial guess: current yield (fallback 10% if market price is non-positive)
    y = float(coupon_rate) / 100.0 if market_price > 0 else 0.10

    for _ in range(max_iterations):
        price_val = 0.0
        deriv_val = 0.0

        for i, t in enumerate(times):
            is_maturity = i == len(times) - 1
            cf = coupon_payment + (fv if is_maturity else 0.0)

            discount = (1.0 + y / coupon_frequency) ** (t * coupon_frequency)
            if discount == 0.0:
                continue

            pv = cf / discount
            price_val += pv
            # Derivative of PV with respect to y:
            # d/dy [CF / (1+y/f)^(t*f)] = -t * CF / (1+y/f)^(t*f+1)
            deriv_val += -t * cf / ((1.0 + y / coupon_frequency) ** (t * coupon_frequency + 1.0))

        diff = price_val - market_price

        if abs(diff) < tol:
            ytm_pct = Decimal(str(y * 100.0))
            return ytm_pct.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)

        if abs(deriv_val) < _DERIV_EPSILON:
            break

        y = y - diff / deriv_val

    msg = f"Newton-Raphson failed to converge after {max_iterations} iterations"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Modified duration
# ---------------------------------------------------------------------------


def modified_duration(
    ytm_pct: Decimal,
    coupon_rate: Decimal,
    face_value: Decimal,
    coupon_frequency: int,
    settlement_date: date,
    maturity_date: date,
) -> Decimal:
    """Compute Macaulay duration then convert to modified duration.

    ``Modified duration = Macaulay duration / (1 + ytm / frequency)``

    Returns:
        Modified duration in years, rounded to 2 dp.
    """
    schedule = _remaining_coupon_dates(settlement_date, maturity_date, coupon_frequency)
    if not schedule:
        return Decimal("0.00")

    y = float(ytm_pct) / 100.0
    fv = float(face_value)
    coupon_payment = float(coupon_rate) / 100.0 * fv / coupon_frequency
    times = [t for _, t in schedule]

    total_pv = 0.0
    weighted_pv = 0.0

    for i, t in enumerate(times):
        is_maturity = i == len(times) - 1
        cf = coupon_payment + (fv if is_maturity else 0.0)

        discount = (1.0 + y / coupon_frequency) ** (t * coupon_frequency)
        if discount == 0.0:
            continue

        pv = cf / discount
        total_pv += pv
        weighted_pv += t * pv

    if total_pv == 0.0:
        return Decimal("0.00")

    macaulay = weighted_pv / total_pv
    mod_dur = macaulay / (1.0 + y / coupon_frequency)

    return Decimal(str(mod_dur)).quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Convexity
# ---------------------------------------------------------------------------


def convexity(
    ytm_pct: Decimal,
    coupon_rate: Decimal,
    face_value: Decimal,
    coupon_frequency: int,
    settlement_date: date,
    maturity_date: date,
) -> Decimal:
    """Compute convexity for price change approximation.

    ``Convexity = (1/P) * sum(t*(t+1/f) * CF_t / (1+y/f)^(t*f+2)) / f^2``

    where *t* is in coupon periods, but we work in years and convert.

    Returns:
        Convexity, rounded to 2 dp.
    """
    schedule = _remaining_coupon_dates(settlement_date, maturity_date, coupon_frequency)
    if not schedule:
        return Decimal("0.00")

    y = float(ytm_pct) / 100.0
    fv = float(face_value)
    coupon_payment = float(coupon_rate) / 100.0 * fv / coupon_frequency
    f = coupon_frequency
    times = [t for _, t in schedule]

    total_pv = 0.0
    cx_sum = 0.0

    for i, t in enumerate(times):
        is_maturity = i == len(times) - 1
        cf = coupon_payment + (fv if is_maturity else 0.0)

        # Number of periods (may be fractional)
        n_periods = t * f
        discount = (1.0 + y / f) ** n_periods
        if discount == 0.0:
            continue

        pv = cf / discount
        total_pv += pv

        # Convexity contribution: n*(n+1) * CF / (1+y/f)^(n+2) / f^2
        cx_sum += n_periods * (n_periods + 1) * cf / ((1.0 + y / f) ** (n_periods + 2))

    if total_pv == 0.0:
        return Decimal("0.00")

    cx = cx_sum / (total_pv * f * f)

    return Decimal(str(cx)).quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# DV01
# ---------------------------------------------------------------------------


def dv01(
    modified_duration_years: Decimal,
    dirty_price_rub: Decimal,
) -> Decimal:
    """Compute dollar value of one basis point (DV01).

    ``DV01 = modified_duration * dirty_price * 0.0001``

    Returns:
        DV01 in RUB per bond, rounded to 4 dp.
    """
    result = modified_duration_years * dirty_price_rub * _BASIS_POINT
    return result.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Price change estimate
# ---------------------------------------------------------------------------


def price_change_estimate(
    modified_duration_years: Decimal,
    convexity_val: Decimal,
    dirty_price_rub: Decimal,
    yield_change_bps: Decimal,
) -> Decimal:
    """Estimate price change for a given yield change.

    ``delta_P ~ -duration * delta_y * P + 0.5 * convexity * (delta_y)^2 * P``

    Args:
        modified_duration_years: Modified duration in years.
        convexity_val: Convexity value.
        dirty_price_rub: Dirty price in RUB.
        yield_change_bps: Yield change in basis points (e.g. 100 for +1%).

    Returns:
        Estimated price change in RUB per bond.
    """
    dy = yield_change_bps * _BASIS_POINT  # convert bps to decimal
    duration_effect = -modified_duration_years * dy * dirty_price_rub
    convexity_effect = Decimal("0.5") * convexity_val * dy * dy * dirty_price_rub
    return duration_effect + convexity_effect

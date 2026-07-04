"""LDV (3-year long-term holding relief, art. 219.1) eligibility + headroom.

Layer L2 (pure Decimal + stdlib date). Computes:

- ``ldv_eligible(lot, today)``: True only if the lot is held >= 3 FULL years,
  acquired on/after 2014-01-01, from a Russian issuer, NOT on an IIS, and its
  cost basis is known (not INPUT_SECURITIES). A conservative T+2-style boundary
  buffer rejects lots whose 3rd anniversary is only a couple of days away.
- ``ldv_headroom(items)``: the exempt headroom using the Kcb coefficient for
  mixed holding periods -- ``Kcb = sum(Vi*i) / sum(Vi)`` where ``Vi`` is the
  positive finrez of securities held ``i >= 3`` full years; the cap is
  ``Kcb * 3_000_000``.

LDV exempts only the capital result of disposal/redemption -- NEVER coupons or
dividends (design section 1.1). Foreign / EAEU lots are FLAGGED not credited
(2025 change, FZ 58-FZ): callers surface those as needs-operator flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from finalayze.tax.lots import TaxLot

# --- statutory named constants (art. 219.1) ---
LDV_MIN_FULL_YEARS = 3
LDV_EARLIEST_ACQUIRE = date(2014, 1, 1)
LDV_ANNUAL_LIMIT = Decimal(3_000_000)  # RUB per full year of holding

# Conservative boundary buffer: require a few extra days past the 3rd
# anniversary before crediting relief (T+2 settlement + "acquire date" vs
# "credit date" ambiguity). Under-count relief, never over-count.
LDV_BOUNDARY_BUFFER_DAYS = 3


@dataclass(frozen=True)
class LdvHoldingItem:
    """One disposed/hypothetical lot's contribution to the Kcb coefficient.

    ``positive_finrez`` (Vi) is the POSITIVE capital result on that lot; only
    positive results with ``full_years >= 3`` feed the coefficient (art. 219.1).
    """

    positive_finrez: Decimal
    full_years: int


@dataclass(frozen=True)
class LdvHeadroom:
    """Result of an LDV exempt-headroom computation."""

    kcb: Decimal
    cap: Decimal
    exempt_amount: Decimal


def full_years_held(acquire: date, today: date) -> int:
    """Full calendar years from ``acquire`` to ``today``, minus the safety buffer.

    "3 full years" expire on the same day/month of the third year (art. 6.1 para
    3). We compute anniversary-based full years, then apply a conservative buffer:
    a lot must be held at least ``anniversary + LDV_BOUNDARY_BUFFER_DAYS`` for the
    year to count, so borderline lots are rejected (under-count, never over-count).
    """
    if today < acquire:
        return 0
    years = today.year - acquire.year
    # step down while the buffered anniversary has not yet been reached
    while years > 0:
        anniversary = _add_years(acquire, years)
        if today >= anniversary + timedelta(days=LDV_BOUNDARY_BUFFER_DAYS):
            break
        years -= 1
    return years


def _add_years(base: date, years: int) -> date:
    """``base`` plus ``years`` calendar years, clamping Feb-29 to Feb-28."""
    try:
        return base.replace(year=base.year + years)
    except ValueError:
        # Feb 29 -> Feb 28 in a non-leap target year
        return base.replace(year=base.year + years, day=28)


def ldv_eligible(lot: TaxLot, today: date) -> bool:
    """True iff ``lot`` qualifies for the 3-year LDV relief as of ``today``.

    Requires: Russian issuer, acquired >= 2014-01-01, held >= 3 full years
    (buffered), NOT on an IIS, and a known cost basis (not INPUT_SECURITIES).
    Any failure returns False -- the caller flags the reason; the engine never
    fabricates a headroom number for an ineligible lot.
    """
    if not lot.cost_basis_known:
        return False
    if not lot.russian_issuer:
        return False
    if lot.on_iis:
        return False
    if lot.acquire_date < LDV_EARLIEST_ACQUIRE:
        return False
    return full_years_held(lot.acquire_date, today) >= LDV_MIN_FULL_YEARS


def kcb_coefficient(items: Iterable[LdvHoldingItem]) -> Decimal:
    """Kcb = sum(Vi * i) / sum(Vi) over items with full_years >= 3 and Vi > 0.

    Returns 0 when there is no qualifying positive finrez.
    """
    numerator = Decimal(0)
    denominator = Decimal(0)
    for it in items:
        if it.full_years >= LDV_MIN_FULL_YEARS and it.positive_finrez > 0:
            numerator += it.positive_finrez * Decimal(it.full_years)
            denominator += it.positive_finrez
    if denominator == 0:
        return Decimal(0)
    return numerator / denominator


def ldv_headroom(items: Iterable[LdvHoldingItem]) -> LdvHeadroom:
    """Exempt headroom for a set of LDV-qualifying lots via the Kcb coefficient.

    cap = Kcb * 3_000_000 ; exempt_amount = min(total positive finrez, cap).
    Only lots with full_years >= 3 and positive finrez contribute.
    """
    qualifying = [
        it for it in items if it.full_years >= LDV_MIN_FULL_YEARS and it.positive_finrez > 0
    ]
    kcb = kcb_coefficient(qualifying)
    cap = kcb * LDV_ANNUAL_LIMIT
    total_finrez = sum((it.positive_finrez for it in qualifying), Decimal(0))
    exempt = min(total_finrez, cap)
    return LdvHeadroom(kcb=kcb, cap=cap, exempt_amount=exempt)

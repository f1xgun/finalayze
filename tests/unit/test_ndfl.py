"""RED scaffold: progressive-band + deposit-floor NDFL helpers (ACCT-03 / R-2 / R-3).

Pins the L0 ``finalayze.core.ndfl`` contract before it exists:
- ``ndfl_marginal(delta, ytd_before)`` applies the marginal 13/15% band across
  the 2.4M RUB progressive threshold (R-3), look-ahead-free (integrates only
  realized-so-far income);
- ``ndfl_on_deposit_interest(gross, ytd_deposit_gross_before, running_floor)``
  taxes only deposit interest above the running-max non-taxable floor (R-2),
  and a higher (later-observed) floor never increases tax retroactively.

RED now: ``finalayze.core.ndfl`` / ``finalayze.core.constants`` do not exist yet.
"""

from __future__ import annotations

from decimal import Decimal

from finalayze.core.constants import (
    NDFL_PROGRESSIVE_THRESHOLD,
    NDFL_RATE,
    NDFL_RATE_HIGH,
)
from finalayze.core.ndfl import YtdTaxAccumulator, ndfl_marginal, ndfl_on_deposit_interest

# ── Constants (named -- no magic numbers, ruff PLR2004) ─────────────────────

_ZERO = Decimal(0)

# Progressive-band scenarios (R-3).
_BAND_YTD_BEFORE = Decimal(2_300_000)  # 100k RUB headroom below the 2.4M threshold
_BAND_DELTA = Decimal(200_000)  # credited delta that straddles the threshold
_BAND_BELOW_PORTION = Decimal(100_000)  # taxed at 13%
_BAND_ABOVE_PORTION = Decimal(100_000)  # taxed at 15%
_BAND_YTD_AFTER = Decimal(2_500_000)

_WHOLLY_BELOW_DELTA = Decimal(500_000)  # ytd_before = 0 -> all 13%
_WHOLLY_ABOVE_YTD_BEFORE = Decimal(3_000_000)  # already past threshold
_WHOLLY_ABOVE_DELTA = Decimal(100_000)  # all 15%

# Deposit-floor scenarios (R-2). floor = 1M x running-max key rate (fraction).
_DEPOSIT_FLOOR = Decimal(210_000)  # 1M x 0.21
_BELOW_FLOOR_YTD_BEFORE = Decimal(100_000)
_BELOW_FLOOR_GROSS = Decimal(50_000)  # 100k + 50k = 150k <= 210k floor -> 0 tax
_CROSS_FLOOR_YTD_BEFORE = Decimal(200_000)
_CROSS_FLOOR_GROSS = Decimal(50_000)  # 200k + 50k = 250k -> 40k over floor
_CROSS_FLOOR_TAXABLE = Decimal(40_000)  # 250k minus 210k
_HIGHER_FLOOR = Decimal(260_000)  # a later, higher monthly rate observed


def test_progressive_band_marginal() -> None:
    """Marginal 13/15% band across the 2.4M RUB threshold (R-3)."""
    tax, ytd_after = ndfl_marginal(_BAND_DELTA, _BAND_YTD_BEFORE)
    expected_tax = _BAND_BELOW_PORTION * NDFL_RATE + _BAND_ABOVE_PORTION * NDFL_RATE_HIGH
    assert tax == expected_tax
    assert ytd_after == _BAND_YTD_AFTER

    # Wholly below the threshold -> all at the base rate.
    below_tax, below_after = ndfl_marginal(_WHOLLY_BELOW_DELTA, _ZERO)
    assert below_tax == _WHOLLY_BELOW_DELTA * NDFL_RATE
    assert below_after == _WHOLLY_BELOW_DELTA

    # Wholly above the threshold -> all at the high rate.
    above_tax, above_after = ndfl_marginal(_WHOLLY_ABOVE_DELTA, _WHOLLY_ABOVE_YTD_BEFORE)
    assert above_tax == _WHOLLY_ABOVE_DELTA * NDFL_RATE_HIGH
    assert above_after == _WHOLLY_ABOVE_YTD_BEFORE + _WHOLLY_ABOVE_DELTA
    # Sanity: the threshold constant is what splits the band.
    assert NDFL_PROGRESSIVE_THRESHOLD == _BAND_YTD_BEFORE + _BAND_BELOW_PORTION


def test_deposit_floor_running_max() -> None:
    """Deposit interest below the running-max floor is untaxed; excess is taxed (R-2)."""
    # Entirely below the floor -> zero tax.
    below = ndfl_on_deposit_interest(_BELOW_FLOOR_GROSS, _BELOW_FLOOR_YTD_BEFORE, _DEPOSIT_FLOOR)
    assert below == _ZERO

    # Crossing the floor -> only the excess over the floor is taxed at 13%.
    crossing = ndfl_on_deposit_interest(_CROSS_FLOOR_GROSS, _CROSS_FLOOR_YTD_BEFORE, _DEPOSIT_FLOOR)
    assert crossing == _CROSS_FLOOR_TAXABLE * NDFL_RATE


def test_deposit_floor_monotone_no_future_month() -> None:
    """A higher (later-observed) running-max floor never increases tax (R-2 / D-17).

    The running-max floor is look-ahead-free: observing a higher monthly key
    rate later in the year can only raise the floor, which can only *reduce*
    the taxable excess for the same gross interest -- never increase it.
    """
    low_floor_tax = ndfl_on_deposit_interest(
        _CROSS_FLOOR_GROSS, _CROSS_FLOOR_YTD_BEFORE, _DEPOSIT_FLOOR
    )
    high_floor_tax = ndfl_on_deposit_interest(
        _CROSS_FLOOR_GROSS, _CROSS_FLOOR_YTD_BEFORE, _HIGHER_FLOOR
    )
    assert high_floor_tax <= low_floor_tax


# ── WR-01: YtdTaxAccumulator wires the progressive band against a running YTD ──

_YEAR = 2023
_NEXT_YEAR = 2024
_BELOW_CREDIT = Decimal(1_000_000)  # << 2.4M -> all 13%
_CROSS_CREDIT = Decimal(2_000_000)  # second 2M credit crosses the 2.4M threshold


def test_ytd_accumulator_below_threshold_is_flat_13() -> None:
    """A credit wholly below the 2.4M YTD threshold is taxed flat 13% (WR-01)."""
    acc = YtdTaxAccumulator()
    tax = acc.tax(_BELOW_CREDIT, _YEAR)
    assert tax == _BELOW_CREDIT * NDFL_RATE
    assert acc.ytd_taxable == _BELOW_CREDIT


def test_ytd_accumulator_15_fires_above_threshold() -> None:
    """Once cumulative YTD taxable income exceeds 2.4M, the excess is taxed 15% (WR-01).

    First credit (1M) is wholly below 2.4M -> 13%. The second credit (2M) takes
    cumulative YTD from 1M to 3M, straddling the 2.4M threshold: 1.4M of it at 13%
    (up to the threshold) and 0.6M at 15%.
    """
    acc = YtdTaxAccumulator()
    acc.tax(_BELOW_CREDIT, _YEAR)  # YTD now 1M

    second_tax = acc.tax(_CROSS_CREDIT, _YEAR)  # YTD 1M -> 3M
    below_portion = NDFL_PROGRESSIVE_THRESHOLD - _BELOW_CREDIT  # 1.4M at 13%
    above_portion = _CROSS_CREDIT - below_portion  # 0.6M at 15%
    expected = below_portion * NDFL_RATE + above_portion * NDFL_RATE_HIGH
    assert second_tax == expected
    assert acc.ytd_taxable == _BELOW_CREDIT + _CROSS_CREDIT


def test_ytd_accumulator_resets_on_new_tax_year() -> None:
    """The accumulator resets on a Jan-1 tax-year boundary -> the band restarts (WR-01)."""
    acc = YtdTaxAccumulator()
    acc.tax(_CROSS_CREDIT, _YEAR)
    acc.tax(_CROSS_CREDIT, _YEAR)  # well above the threshold this year
    assert acc.ytd_taxable == _CROSS_CREDIT + _CROSS_CREDIT

    # New year: YTD resets, so a sub-threshold credit is flat 13% again.
    next_year_tax = acc.tax(_BELOW_CREDIT, _NEXT_YEAR)
    assert next_year_tax == _BELOW_CREDIT * NDFL_RATE
    assert acc.ytd_taxable == _BELOW_CREDIT

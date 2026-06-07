"""Single NDFL band/floor helper (Layer 0).

Pure-Decimal arithmetic consumed by all three sleeves (equity dividends, bond
coupons, deposit interest). The function holds NO state -- the YTD accumulators
are owned by the caller (engine/broker) and passed in, so the helper stays a
true L0 leaf. Imports only the L0 constants module (downward-legal) + stdlib.

- ``ndfl_marginal`` applies the progressive 13/15% band marginally across the
  2.4M RUB threshold (R-3), look-ahead-free (integrates only realized-so-far
  income). ``YtdTaxAccumulator`` is the stateful per-run wiring (WR-01): the
  engine owns one per ``run_portfolio`` call and threads it into the dividend
  credit so the band is applied against a single cross-sleeve running YTD that
  resets each tax year.
- ``ndfl_on_deposit_interest`` taxes only the portion of YTD deposit interest
  above the running-max non-taxable floor (R-2); a higher (later-observed)
  floor never increases tax retroactively.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal

from finalayze.core.constants import (
    NDFL_PROGRESSIVE_THRESHOLD,
    NDFL_RATE,
    NDFL_RATE_HIGH,
)


def ndfl_marginal(delta: Decimal, ytd_before: Decimal) -> tuple[Decimal, Decimal]:
    """Marginal NDFL on a delta of taxable income credited at bar T.

    ``ytd_before`` is the cumulative taxable income (all three sleeves) credited
    on/before T. Returns ``(tax, ytd_after)``. Look-ahead-free: uses only
    realized-so-far income, so re-ordering future bars cannot change a past bar.
    """
    below = max(Decimal(0), NDFL_PROGRESSIVE_THRESHOLD - ytd_before)
    at_13 = min(delta, below)
    at_15 = delta - at_13
    tax = at_13 * NDFL_RATE + at_15 * NDFL_RATE_HIGH
    return tax, ytd_before + delta


class YtdTaxAccumulator:
    """Per-run cumulative-YTD taxable-income accumulator for the progressive band (WR-01 / R-3).

    Owned by the engine for one ``run_portfolio`` call and shared across the income
    credit paths so the cumulative 13/15% band (``ndfl_marginal``) is applied once
    against a single cross-sleeve YTD total. The accumulator resets on a Jan-1 tax-year
    boundary (``year`` change), so the band restarts each year. Look-ahead-free: it
    integrates only income credited on/before the current bar, so re-ordering future
    bars never changes a past bar's tax.

    Stateful by design (it accumulates across bars), but it is a leaf object holding
    only Decimals + an int year -- it imports no project modules beyond the L0
    constants its ``ndfl_marginal`` already uses, so it stays a true L0 helper.
    """

    def __init__(self) -> None:
        self._ytd_taxable = Decimal(0)
        self._current_year: int | None = None

    @property
    def ytd_taxable(self) -> Decimal:
        """Cumulative taxable income credited so far in the current tax year."""
        return self._ytd_taxable

    def tax(self, gross: Decimal, year: int) -> Decimal:
        """Marginal NDFL on ``gross`` credited in ``year``; advances the YTD total.

        Resets the YTD accumulator on a new tax year (Jan-1 boundary) before applying
        the marginal band, so cumulative income above the 2.4M RUB threshold within
        the SAME year is taxed at 15% (R-3). Returns only the tax for THIS credit.
        """
        if year != self._current_year:
            self._ytd_taxable = Decimal(0)
            self._current_year = year
        tax, self._ytd_taxable = ndfl_marginal(gross, self._ytd_taxable)
        return tax


def ndfl_on_deposit_interest(
    gross: Decimal, ytd_deposit_gross_before: Decimal, running_floor: Decimal
) -> Decimal:
    """Tax only the portion of YTD deposit interest above the running-max floor.

    ``running_floor = DEPOSIT_FLOOR_BASE x running_max_key_rate_so_far``
    (a fraction). Only key-rate observations on/before the current bar feed the
    floor -- it monotonically rises and never uses a future month, so the result
    is look-ahead-safe (R-2). A higher (later) floor can only reduce the taxable
    excess for the same gross, never increase it.
    """
    ytd_after = ytd_deposit_gross_before + gross
    taxable_before = max(Decimal(0), ytd_deposit_gross_before - running_floor)
    taxable_after = max(Decimal(0), ytd_after - running_floor)
    return (taxable_after - taxable_before) * NDFL_RATE

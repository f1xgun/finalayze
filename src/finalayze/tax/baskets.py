"""Per-base YTD tax accumulators (Layer L2, wraps core.ndfl.ndfl_marginal).

Each Russian personal-income tax base tracked by the first slice gets its OWN
YTD accumulator with its OWN 2.4M RUB progressive threshold (design sections
1.4, 1.6, 2.1, 2.6):

- ``TaxBase.SECURITIES`` (base A): realized capital results of exchange-traded
  securities PLUS coupons, netted together within the year.
- ``TaxBase.DIVIDENDS`` (base D): a SEPARATE base. NEVER netted with base A,
  NEVER harvested. Its own 2.4M threshold. A base-A loss must NOT reduce
  dividend tax (INVARIANT 1).

This does NOT modify ``core/ndfl.py`` in place (its single cross-sleeve
``YtdTaxAccumulator`` is a documented live defect, out of scope). Instead each
base owns a private running-YTD total and applies ``ndfl_marginal`` marginally.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from finalayze.core.ndfl import ndfl_marginal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from finalayze.tax.lots import Operation, RealizedResult

from finalayze.tax.lots import OperationType


class TaxBase(StrEnum):
    """First-slice tax bases, each with an independent 2.4M threshold."""

    SECURITIES = "SECURITIES"  # base A: realized gains/losses + coupons, netted
    DIVIDENDS = "DIVIDENDS"  # base D: separate, never netted, never harvested


class BaseAccumulators:
    """Independent per-base YTD accumulators over ``ndfl_marginal``.

    Each base keeps its own cumulative-YTD taxable total and its own tax-year, so
    the progressive 13/15% band is applied independently per base against its own
    2.4M threshold. Crediting one base never affects another base's running YTD.
    """

    def __init__(self) -> None:
        self._ytd: dict[TaxBase, Decimal] = dict.fromkeys(TaxBase, Decimal(0))
        self._year: dict[TaxBase, int | None] = dict.fromkeys(TaxBase, None)

    def ytd(self, base: TaxBase) -> Decimal:
        """Cumulative taxable income credited so far this year for ``base``."""
        return self._ytd[base]

    def credit(self, base: TaxBase, delta: Decimal, year: int) -> Decimal:
        """Apply the marginal band to ``delta`` credited to ``base`` in ``year``.

        Resets that base's YTD on a new tax year (Jan-1 boundary). Returns the tax
        for THIS credit only. A negative ``delta`` (a base-A loss) is accumulated
        into the base's own running YTD (reducing its positive result) but never
        leaks into another base.
        """
        if year != self._year[base]:
            self._ytd[base] = Decimal(0)
            self._year[base] = year
        tax, self._ytd[base] = ndfl_marginal(delta, self._ytd[base])
        return tax


def realized_ytd_base_a(
    realized: Iterable[RealizedResult],
    coupons: Iterable[Operation],
) -> Decimal:
    """Realized YTD base-A total: netted realized results + coupons (NO dividends).

    Sums realized capital results (gains net of losses) and adds COUPON payments.
    DIVIDEND operations are excluded by construction (only ``OperationType.COUPON``
    contributes) -- dividends live in a separate base and are never netted here
    (INVARIANT 1).
    """
    total = Decimal(0)
    for r in realized:
        total += r.realized
    for op in coupons:
        if op.op_type is OperationType.COUPON:
            total += op.payment
    return total

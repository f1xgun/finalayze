"""Tax-loss harvesting vs realized YTD base-A result (Layer L2, pure Decimal).

Computes the savings from realizing an unrealized loss to offset the positive
YTD base-A result (realized gains + coupons). Design section 2.4:

    offset_used     = min(total_harvestable_loss, positive_YTD_base_A)
    savings_estimate= tax(positive_YTD) - tax(positive_YTD - offset_used)

The offset comes off the TOP of the YTD base-A income, so the effective relief
rate follows the marginal 13/15% band (relief at 15% while above the 2.4M
threshold, 13% below). A harvested base-A loss offsets coupons (base A) but
NEVER dividends (base D) -- ``harvestable`` takes no dividend input at all.

Honest warning: a harvest SELL closes the OLDEST lot first (strict FIFO). If
that resets another lot's LDV clock, we emit a warning (no wash-sale rule in RU,
so an immediate re-buy is allowed, but the LDV clock DOES restart).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.ndfl import ndfl_marginal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from finalayze.tax.lots import TaxLot


@dataclass(frozen=True)
class HarvestCandidate:
    """An open lot carrying an unrealized loss that could be harvested.

    ``unrealized_loss`` is the (negative or magnitude) loss on the lot; the engine
    uses its absolute value. ``breaks_ldv_clock`` flags that harvesting this lot's
    SELL would (via FIFO) reset another lot's LDV holding clock.
    """

    lot: TaxLot
    unrealized_loss: Decimal
    breaks_ldv_clock: bool = False


@dataclass(frozen=True)
class HarvestResult:
    """Result of a harvest-vs-YTD-base-A computation."""

    offset_used: Decimal
    savings_estimate: Decimal
    warnings: list[str] = field(default_factory=list)


def harvestable(
    realized_ytd_base_a: Decimal,
    candidates: Iterable[HarvestCandidate],
) -> HarvestResult:
    """Estimate harvest savings against the positive YTD base-A result.

    Only a POSITIVE YTD base-A result can be offset (there is nothing to relieve
    otherwise). The savings estimate uses the marginal band so relief above the
    2.4M threshold is valued at 15% and below at 13%.
    """
    candidate_list = list(candidates)
    total_loss = sum(
        (abs(c.unrealized_loss) for c in candidate_list),
        Decimal(0),
    )
    positive_ytd = max(Decimal(0), realized_ytd_base_a)
    offset_used = min(total_loss, positive_ytd)

    # marginal relief: tax on the full positive YTD minus tax on the reduced base
    tax_before, _ = ndfl_marginal(positive_ytd, Decimal(0))
    tax_after, _ = ndfl_marginal(positive_ytd - offset_used, Decimal(0))
    savings = tax_before - tax_after

    warnings: list[str] = [
        (
            f"harvest SELL of {c.lot.ticker} (FIFO) would RESET the LDV holding "
            f"clock on the affected lot (acquired {c.lot.acquire_date}); "
            f"3-year relief would restart"
        )
        for c in candidate_list
        if c.breaks_ldv_clock
    ]

    return HarvestResult(
        offset_used=offset_used,
        savings_estimate=savings,
        warnings=warnings,
    )

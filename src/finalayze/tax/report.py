"""Decision-support report assembly (Layer L2, pure Decimal).

Assembles the tax-optimization decision-support report from the lower modules
(lots, baskets, ldv, harvest). Design sections 2.3-2.5, 4.3 steps 8-10.

HARD SCOPE: decision-support ONLY. Every action item is advisory -- the engine
NEVER places an order or trades. Real money = HARD STOP. A data gap always
yields an honest DEGRADATION FLAG (with a reason), never a silent number.

Honest-degradation flags emitted (design section 3.2):
- COST_BASIS_UNKNOWN  (INPUT_SECURITIES transferred-in lot)
- FX_NOT_COMPUTED     (foreign-currency lot -- FX revaluation not computed)
- HISTORY_TRUNCATED   (operations history hit the sidecar page cap)
- IIS_EXCLUDED        (IIS lot excluded from LDV/harvest)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from finalayze.core.constants import NDFL_RATE, NDFL_RATE_HIGH
from finalayze.core.ndfl import ndfl_marginal
from finalayze.tax.baskets import TaxBase, realized_ytd_base_a
from finalayze.tax.harvest import harvestable
from finalayze.tax.ldv import LdvHeadroom, ldv_eligible, ldv_headroom

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date

    from finalayze.tax.harvest import HarvestCandidate
    from finalayze.tax.ldv import LdvHoldingItem
    from finalayze.tax.lots import Operation, RealizedResult, TaxLot

_RUB = "RUB"

Confidence = str  # "confirmed" | "doubtful" | "needs-operator"

SCOPE_BANNER = (
    "DECISION-SUPPORT ONLY -- NOT tax advice. This report outputs action items to "
    "CONSIDER and RUB savings ESTIMATES. It NEVER places an order and NEVER trades. "
    "Real money = HARD STOP: any real-account action requires explicit operator "
    "confirmation. All figures are estimates reconstructed from operations history and "
    "may differ from the broker/FNS calculation -- reconcile with the 2-NDFL broker "
    "statement before filing 3-NDFL. Before ANY number is trusted the operator MUST "
    "confirm personal facts: RU tax residency, total annual investment+labour income, "
    "IIS presence/type/open-year, number of brokers, carried-forward prior-year losses."
)


class FlagReason(StrEnum):
    """Honest-degradation flag reasons (a gap yields a flag, never a silent number)."""

    COST_BASIS_UNKNOWN = "COST_BASIS_UNKNOWN"
    FX_NOT_COMPUTED = "FX_NOT_COMPUTED"
    HISTORY_TRUNCATED = "HISTORY_TRUNCATED"
    IIS_EXCLUDED = "IIS_EXCLUDED"


@dataclass(frozen=True)
class DegradationFlag:
    """A degraded input surfaced honestly instead of a fabricated number."""

    reason: FlagReason
    detail: str


@dataclass(frozen=True)
class ActionItem:
    """One advisory action item. ``places_order`` is ALWAYS False by construction."""

    category: str  # LDV | HARVEST | TAX_DRAG | DIVIDEND | DEFECT | IIS
    description: str
    savings_estimate: Decimal
    confidence: Confidence
    places_order: bool = False


@dataclass(frozen=True)
class ForwardIncome:
    """A forward income event (coupon/dividend) projected for tax drag."""

    kind: TaxBase
    amount: Decimal
    currency: str = _RUB


@dataclass(frozen=True)
class TaxDragProjection:
    """Projected tax on forward income for the remainder of the year."""

    projected_tax: Decimal
    flags: list[DegradationFlag] = field(default_factory=list)


@dataclass(frozen=True)
class TaxReport:
    """The assembled decision-support report."""

    scope_banner: str
    action_items: list[ActionItem]
    flags: list[DegradationFlag]
    realized_ytd_base_a: Decimal
    dividend_ytd: Decimal


def _effective_rate(ytd_before: Decimal, amount: Decimal) -> Decimal:
    """Marginal effective rate on ``amount`` credited above ``ytd_before``."""
    if amount == 0:
        return NDFL_RATE
    tax, _ = ndfl_marginal(amount, ytd_before)
    return tax / amount


def project_tax_drag(
    forward_income: Iterable[ForwardIncome],
    year: int,  # noqa: ARG001 - year documents the projection window; band is per-event here
    *,
    div_ytd_before: Decimal = Decimal(0),
    sec_ytd_before: Decimal = Decimal(0),
) -> TaxDragProjection:
    """Project tax on forward coupons/dividends (design section 2.5).

    Dividends are taxed at the full 13/15% band (never harvested). Base-A coupons
    at 13/15%. A foreign-currency event yields an FX_NOT_COMPUTED flag AND is not
    silently taxed (the FX revaluation is not computed in the first slice).

    ``div_ytd_before`` / ``sec_ytd_before`` seed each base's marginal band from the
    already-realized YTD position (WR-01) so forward income above the 2.4M
    threshold is taxed at 15%, not silently re-started at the bottom of the 13%
    band (which would UNDER-state the drag -- a cost -- the non-conservative
    direction, design section 0 / 2.5 "по маржинальной ставке базы").
    """
    projected = Decimal(0)
    flags: list[DegradationFlag] = []
    div_ytd = div_ytd_before
    sec_ytd = sec_ytd_before
    for fwd in forward_income:
        if fwd.currency != _RUB:
            flags.append(
                DegradationFlag(
                    reason=FlagReason.FX_NOT_COMPUTED,
                    detail=(
                        f"forward {fwd.kind.value} in {fwd.currency}: FX revaluation not "
                        f"computed -- taxable amount not estimated"
                    ),
                )
            )
            continue
        if fwd.kind is TaxBase.DIVIDENDS:
            tax, div_ytd = ndfl_marginal(fwd.amount, div_ytd)
        else:
            tax, sec_ytd = ndfl_marginal(fwd.amount, sec_ytd)
        projected += tax
    return TaxDragProjection(projected_tax=projected, flags=flags)


def _lot_flags(lots: Iterable[TaxLot], *, history_truncated: bool) -> list[DegradationFlag]:
    flags: list[DegradationFlag] = []
    for lot in lots:
        if not lot.cost_basis_known:
            flags.append(
                DegradationFlag(
                    reason=FlagReason.COST_BASIS_UNKNOWN,
                    detail=(
                        f"{lot.ticker}: cost basis unknown (transferred-in / "
                        f"INPUT_SECURITIES) -- LDV clock + cost UNKNOWN, not estimated"
                    ),
                )
            )
        if lot.currency != _RUB:
            flags.append(
                DegradationFlag(
                    reason=FlagReason.FX_NOT_COMPUTED,
                    detail=(
                        f"{lot.ticker}: {lot.currency} lot -- FX revaluation not computed "
                        f"(taxable by default unless MinFin sovereign eurobond)"
                    ),
                )
            )
        if lot.on_iis:
            flags.append(
                DegradationFlag(
                    reason=FlagReason.IIS_EXCLUDED,
                    detail=(
                        f"{lot.ticker}: on IIS -- excluded from LDV/harvest; confirm "
                        f"IIS type/open-year with the broker"
                    ),
                )
            )
    if history_truncated:
        flags.append(
            DegradationFlag(
                reason=FlagReason.HISTORY_TRUNCATED,
                detail=(
                    "operations history hit the sidecar page cap -- lot reconstruction "
                    "is INCOMPLETE; window by --days and re-run"
                ),
            )
        )
    return flags


def _ldv_action_items(
    open_lots: Iterable[TaxLot],
    today: date,
    ldv_hypothetical: dict[str, list[LdvHoldingItem]] | None,
    positive_ytd_base_a: Decimal,
) -> list[ActionItem]:
    """One aggregated LDV action item PER figi (WR-02 dedupe / WR-03 Kcb blend).

    ``ldv_hypothetical`` maps a figi to ALL its LDV-qualifying disposal lots. A
    figi's headroom is computed ONCE over the full list, so the Kcb coefficient
    blends across mixed holding periods (design 2.3) and a figi with several open
    lots emits a single item -- never a per-lot double count off the same figi.
    """
    if not ldv_hypothetical:
        return []
    items: list[ActionItem] = []
    seen_figis: set[str] = set()
    for lot in open_lots:
        if lot.figi in seen_figis:
            continue
        if not ldv_eligible(lot, today):
            continue
        hyps = ldv_hypothetical.get(lot.figi)
        if not hyps:
            continue
        seen_figis.add(lot.figi)
        hr: LdvHeadroom = ldv_headroom(hyps)
        total_finrez = sum((h.positive_finrez for h in hyps), Decimal(0))
        # exempt result relieves tax at the marginal rate ABOVE the current base-A YTD
        rate = _effective_rate(positive_ytd_base_a, hr.exempt_amount)
        savings = hr.exempt_amount * rate
        items.append(
            ActionItem(
                category="LDV",
                description=(
                    f"{lot.ticker}: {len(hyps)} qualifying lot(s); a disposal would "
                    f"realize ~{total_finrez} RUB finrez, of which ~{hr.exempt_amount} "
                    f"RUB is LDV-exempt (Kcb={hr.kcb}, cap={hr.cap}) -> tax saved "
                    f"~{savings} RUB. Consider holding to preserve / using the relief on sale."
                ),
                savings_estimate=savings,
                confidence="confirmed",
            )
        )
    return items


def _harvest_action_items(
    positive_ytd_base_a: Decimal,
    harvest_candidates: Iterable[HarvestCandidate],
) -> list[ActionItem]:
    candidates = list(harvest_candidates)
    if not candidates:
        return []
    res = harvestable(positive_ytd_base_a, candidates)
    if res.offset_used <= 0:
        return []
    warn = " ".join(res.warnings)
    description = (
        f"Harvest ~{res.offset_used} RUB of unrealized base-A losses against the positive "
        f"YTD base-A result -> tax saved ~{res.savings_estimate} RUB. The loss offsets "
        f"gains AND coupons but NEVER dividends (separate base)."
    )
    if warn:
        description = f"{description} WARNING: {warn}"
    return [
        ActionItem(
            category="HARVEST",
            description=description,
            savings_estimate=res.savings_estimate,
            confidence="confirmed",
        )
    ]


def _dividend_action_item(dividend_ytd: Decimal) -> list[ActionItem]:
    if dividend_ytd <= 0:
        return []
    tax, _ = ndfl_marginal(dividend_ytd, Decimal(0))
    return [
        ActionItem(
            category="DIVIDEND",
            description=(
                f"Dividend base (separate, own 2.4M threshold): ~{dividend_ytd} RUB YTD -> "
                f"~{tax} RUB NDFL. Dividends are NEVER netted or harvested and are not "
                f"LDV/IIS-B exempt. Figure is an ESTIMATE (payment is net of withholding)."
            ),
            savings_estimate=Decimal(0),
            confidence="confirmed",
        )
    ]


def _standing_notes() -> list[ActionItem]:
    """Documented doubtful / needs-operator notes carried in every report."""
    return [
        ActionItem(
            category="DEFECT",
            description=(
                "Documented live defect (out of scope for this slice): the allocation "
                "gate's shared YtdTaxAccumulator mixes dividends (a separate base with its "
                "own 2.4M threshold) with coupons/finrez, pushing into 15% too early. This "
                "engine builds per-base accumulators instead; fixing ndfl.py is a separate "
                "action-item."
            ),
            savings_estimate=Decimal(0),
            confidence="needs-operator",
        ),
        ActionItem(
            category="IIS",
            description=(
                "Clarify the IIS-3 type-B exemption limit (30M 'per year across accounts "
                "closed this year' vs 'whole term') and IIS type/open-year with the "
                "broker/FNS before sizing any IIS headroom -- not computed automatically."
            ),
            savings_estimate=Decimal(0),
            confidence="doubtful",
        ),
    ]


def build_report(
    *,
    today: date,
    year: int,
    open_lots: list[TaxLot],
    realized_ytd: list[RealizedResult],
    coupons_ytd: list[Operation],
    dividends_ytd: list[Operation],
    forward_income: list[ForwardIncome],
    harvest_candidates: list[HarvestCandidate],
    history_truncated: bool,
    ldv_hypothetical: dict[str, list[LdvHoldingItem]] | None = None,
) -> TaxReport:
    """Assemble the decision-support report.

    Never emits an order. Every data gap becomes a DegradationFlag. LDV items are
    produced ONLY for eligible lots with a supplied hypothetical finrez; an
    INPUT_SECURITIES lot is ineligible and yields a flag, not a number.
    """
    base_a_ytd = realized_ytd_base_a(realized_ytd, coupons_ytd)
    positive_ytd_base_a = max(Decimal(0), base_a_ytd)
    dividend_ytd = sum(
        (op.payment for op in dividends_ytd),
        Decimal(0),
    )

    flags = _lot_flags(open_lots, history_truncated=history_truncated)

    # WR-01: seed each base's marginal band from the real realized YTD so forward
    # income above 2.4M is taxed at 15%, not re-started at the 13% band bottom.
    drag = project_tax_drag(
        forward_income,
        year,
        div_ytd_before=max(Decimal(0), dividend_ytd),
        sec_ytd_before=positive_ytd_base_a,
    )
    flags.extend(drag.flags)

    action_items: list[ActionItem] = []
    action_items.extend(_ldv_action_items(open_lots, today, ldv_hypothetical, positive_ytd_base_a))
    action_items.extend(_harvest_action_items(positive_ytd_base_a, harvest_candidates))
    if drag.projected_tax > 0:
        action_items.append(
            ActionItem(
                category="TAX_DRAG",
                description=(
                    f"Projected NDFL drag on forward coupons/dividends: ~{drag.projected_tax} "
                    f"RUB. FX-bond legs are flagged not computed."
                ),
                savings_estimate=Decimal(0),
                confidence="confirmed",
            )
        )
    action_items.extend(_dividend_action_item(dividend_ytd))
    action_items.extend(_standing_notes())

    return TaxReport(
        scope_banner=SCOPE_BANNER,
        action_items=action_items,
        flags=flags,
        realized_ytd_base_a=base_a_ytd,
        dividend_ytd=dividend_ytd,
    )


# re-export for convenience so callers can build the marginal rates without a
# second import path (both are used in item descriptions / effective-rate math)
__all__ = [
    "NDFL_RATE",
    "NDFL_RATE_HIGH",
    "ActionItem",
    "DegradationFlag",
    "FlagReason",
    "ForwardIncome",
    "TaxDragProjection",
    "TaxReport",
    "build_report",
    "project_tax_drag",
]

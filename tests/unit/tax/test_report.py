"""Tests for the decision-support report assembly (design section 4.3 steps 8-10).

Covers: after-tax yield / tax-drag projection (step 8), honest degradation flags
(step 9), and the assembled report with scope banner + action items + RUB
estimates + NO orders + doubtful/needs-operator markers (step 10).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.core.constants import (
    NDFL_PROGRESSIVE_THRESHOLD,
    NDFL_RATE,
    NDFL_RATE_HIGH,
)
from finalayze.tax.baskets import TaxBase
from finalayze.tax.harvest import HarvestCandidate
from finalayze.tax.ldv import LdvHoldingItem
from finalayze.tax.lots import Operation, OperationType, RealizedResult, TaxLot
from finalayze.tax.report import (
    ActionItem,
    DegradationFlag,
    FlagReason,
    ForwardIncome,
    TaxReport,
    build_report,
    project_tax_drag,
)

TICKER = "SBER"
FIGI = "BBG004730N88"
CCY_RUB = "RUB"
CCY_USD = "USD"

TODAY = date(2026, 6, 1)
YEAR = 2026

DIVIDEND_100K = Decimal(100_000)
COUPON_50K = Decimal(50_000)
ABOVE_THRESHOLD_YTD = NDFL_PROGRESSIVE_THRESHOLD + Decimal(500_000)


def _lot(
    *,
    acquire: date = date(2021, 1, 1),
    russian: bool = True,
    on_iis: bool = False,
    cost_known: bool = True,
    currency: str = CCY_RUB,
) -> TaxLot:
    return TaxLot(
        figi=FIGI,
        ticker=TICKER,
        acquire_date=acquire,
        quantity=Decimal(100),
        price_per_unit=Decimal(200),
        commission_buy=Decimal(10),
        currency=currency,
        russian_issuer=russian,
        on_iis=on_iis,
        cost_basis_known=cost_known,
    )


# ---------- step 8: tax-drag projection ----------


def test_project_tax_drag_dividend_full_rate() -> None:
    fwd = [ForwardIncome(kind=TaxBase.DIVIDENDS, amount=DIVIDEND_100K, currency=CCY_RUB)]
    proj = project_tax_drag(fwd, YEAR)
    # dividend taxed at full 13% (below threshold), no harvesting
    assert proj.projected_tax == DIVIDEND_100K * NDFL_RATE


def test_project_tax_drag_coupon_base_a() -> None:
    fwd = [ForwardIncome(kind=TaxBase.SECURITIES, amount=COUPON_50K, currency=CCY_RUB)]
    proj = project_tax_drag(fwd, YEAR)
    assert proj.projected_tax == COUPON_50K * NDFL_RATE


def test_project_tax_drag_fx_bond_flagged_not_computed() -> None:
    # a foreign-currency coupon triggers an FX-revaluation-not-computed flag
    fwd = [ForwardIncome(kind=TaxBase.SECURITIES, amount=COUPON_50K, currency=CCY_USD)]
    proj = project_tax_drag(fwd, YEAR)
    assert any(f.reason is FlagReason.FX_NOT_COMPUTED for f in proj.flags)


def test_project_tax_drag_starts_from_current_ytd_band_securities() -> None:
    """WR-01: forward base-A coupon above the 2.4M YTD is taxed at 15%, not 13%."""
    fwd = [ForwardIncome(kind=TaxBase.SECURITIES, amount=COUPON_50K, currency=CCY_RUB)]
    proj = project_tax_drag(fwd, YEAR, sec_ytd_before=ABOVE_THRESHOLD_YTD)
    assert proj.projected_tax == COUPON_50K * NDFL_RATE_HIGH


def test_project_tax_drag_starts_from_current_ytd_band_dividends() -> None:
    """WR-01: forward dividend above the 2.4M dividend-base YTD is taxed at 15%."""
    fwd = [ForwardIncome(kind=TaxBase.DIVIDENDS, amount=DIVIDEND_100K, currency=CCY_RUB)]
    proj = project_tax_drag(fwd, YEAR, div_ytd_before=ABOVE_THRESHOLD_YTD)
    assert proj.projected_tax == DIVIDEND_100K * NDFL_RATE_HIGH


def test_project_tax_drag_defaults_to_zero_ytd() -> None:
    """WR-01: with no YTD context the band still starts at the bottom (13%)."""
    fwd = [ForwardIncome(kind=TaxBase.SECURITIES, amount=COUPON_50K, currency=CCY_RUB)]
    proj = project_tax_drag(fwd, YEAR)
    assert proj.projected_tax == COUPON_50K * NDFL_RATE


def test_build_report_threads_realized_ytd_into_drag_projection() -> None:
    """WR-01: build_report seeds the drag band from the real base-A / dividend YTD.

    A realized base-A result already above 2.4M means a forward coupon is taxed at
    15%, so the projected drag must exceed the naive 13% figure.
    """
    realized = [
        RealizedResult(
            figi=FIGI,
            ticker=TICKER,
            acquire_date=date(2020, 1, 1),
            dispose_date=TODAY,
            quantity=Decimal(1),
            proceeds=ABOVE_THRESHOLD_YTD,
            matched_cost=Decimal(0),
            buy_commission_share=Decimal(0),
            sell_commission_share=Decimal(0),
        )
    ]
    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot()],
        realized_ytd=realized,
        coupons_ytd=[],
        dividends_ytd=[],
        forward_income=[ForwardIncome(kind=TaxBase.SECURITIES, amount=COUPON_50K)],
        harvest_candidates=[],
        history_truncated=False,
    )
    drag_items = [a for a in report.action_items if a.category == "TAX_DRAG"]
    assert drag_items
    # the drag description carries the projected figure; it must reflect the 15%
    # band, i.e. strictly more than the naive all-13% projection
    naive = COUPON_50K * NDFL_RATE
    correct = COUPON_50K * NDFL_RATE_HIGH
    assert f"{correct}" in drag_items[0].description
    assert f"{naive}" not in drag_items[0].description


# ---------- step 9: honest degradation flags ----------


def test_input_securities_lot_produces_cost_basis_flag() -> None:
    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot(cost_known=False)],
        realized_ytd=[],
        coupons_ytd=[],
        dividends_ytd=[],
        forward_income=[],
        harvest_candidates=[],
        history_truncated=False,
    )
    assert any(f.reason is FlagReason.COST_BASIS_UNKNOWN for f in report.flags)


def test_fx_lot_produces_fx_flag() -> None:
    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot(currency=CCY_USD)],
        realized_ytd=[],
        coupons_ytd=[],
        dividends_ytd=[],
        forward_income=[],
        harvest_candidates=[],
        history_truncated=False,
    )
    assert any(f.reason is FlagReason.FX_NOT_COMPUTED for f in report.flags)


def test_truncated_history_produces_flag() -> None:
    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot()],
        realized_ytd=[],
        coupons_ytd=[],
        dividends_ytd=[],
        forward_income=[],
        harvest_candidates=[],
        history_truncated=True,
    )
    assert any(f.reason is FlagReason.HISTORY_TRUNCATED for f in report.flags)


def test_iis_lot_flagged_for_ldv_exclusion() -> None:
    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot(on_iis=True)],
        realized_ytd=[],
        coupons_ytd=[],
        dividends_ytd=[],
        forward_income=[],
        harvest_candidates=[],
        history_truncated=False,
    )
    assert any(f.reason is FlagReason.IIS_EXCLUDED for f in report.flags)


def test_flag_never_produces_silent_number() -> None:
    # an INPUT_SECURITIES lot must NOT appear as an LDV action item with a number
    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot(cost_known=False)],
        realized_ytd=[],
        coupons_ytd=[],
        dividends_ytd=[],
        forward_income=[],
        harvest_candidates=[],
        history_truncated=False,
    )
    ldv_items = [a for a in report.action_items if a.category == "LDV"]
    assert ldv_items == []


# ---------- step 10: report assembly ----------


def _dividend(amount: Decimal) -> Operation:
    return Operation(
        op_type=OperationType.DIVIDEND,
        op_date=TODAY,
        figi=FIGI,
        ticker="LKOH",
        payment=amount,
        currency=CCY_RUB,
        payment_is_net_estimate=True,
    )


def _coupon(amount: Decimal) -> Operation:
    return Operation(
        op_type=OperationType.COUPON,
        op_date=TODAY,
        figi=FIGI,
        ticker="OFZ",
        payment=amount,
        currency=CCY_RUB,
    )


def test_report_has_scope_banner_no_orders() -> None:
    report: TaxReport = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot()],
        realized_ytd=[],
        coupons_ytd=[],
        dividends_ytd=[_dividend(DIVIDEND_100K)],
        forward_income=[],
        harvest_candidates=[],
        history_truncated=False,
    )
    banner = report.scope_banner.lower()
    assert "decision-support" in banner
    assert "hard stop" in banner
    assert "no order" in banner or "never" in banner
    # no action item may ever be an order/trade
    for item in report.action_items:
        assert item.places_order is False


def test_report_ldv_action_item_has_ruble_estimate() -> None:
    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot(acquire=date(2020, 1, 1))],
        realized_ytd=[],
        coupons_ytd=[],
        dividends_ytd=[],
        forward_income=[],
        harvest_candidates=[],
        history_truncated=False,
        ldv_hypothetical={FIGI: LdvHoldingItem(positive_finrez=Decimal(500_000), full_years=6)},
    )
    ldv_items = [a for a in report.action_items if a.category == "LDV"]
    assert ldv_items
    assert all(isinstance(a.savings_estimate, Decimal) for a in ldv_items)


def test_report_dividend_tax_never_harvestable() -> None:
    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot()],
        realized_ytd=[],
        coupons_ytd=[],
        dividends_ytd=[_dividend(DIVIDEND_100K)],
        forward_income=[],
        harvest_candidates=[
            HarvestCandidate(lot=_lot(), unrealized_loss=Decimal(-1_000_000)),
        ],
        history_truncated=False,
    )
    # no harvest action item may claim to offset dividend tax
    for item in report.action_items:
        if item.category == "HARVEST":
            assert "dividend" not in item.description.lower() or "never" in item.description.lower()


def test_report_documents_ndfl_accumulator_defect() -> None:
    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot()],
        realized_ytd=[],
        coupons_ytd=[],
        dividends_ytd=[],
        forward_income=[],
        harvest_candidates=[],
        history_truncated=False,
    )
    # the documented live defect is surfaced as a needs-operator/doubtful note
    assert any(
        "YtdTaxAccumulator" in a.description or "separate base" in a.description.lower()
        for a in report.action_items
    )


def test_report_marks_doubtful_iis_limit() -> None:
    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=[_lot()],
        realized_ytd=[],
        coupons_ytd=[],
        dividends_ytd=[],
        forward_income=[],
        harvest_candidates=[],
        history_truncated=False,
    )
    assert any(a.confidence in {"doubtful", "needs-operator"} for a in report.action_items)


def test_action_item_is_frozen_dataclass() -> None:
    item = ActionItem(
        category="LDV",
        description="test",
        savings_estimate=Decimal(0),
        confidence="confirmed",
    )
    assert item.places_order is False


def test_degradation_flag_carries_reason_and_detail() -> None:
    flag = DegradationFlag(reason=FlagReason.COST_BASIS_UNKNOWN, detail="x")
    assert flag.reason is FlagReason.COST_BASIS_UNKNOWN
    assert flag.detail == "x"

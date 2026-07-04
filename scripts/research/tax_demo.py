"""Synthetic-portfolio demo for the tax-optimization decision-support engine.

Decision-support ONLY: builds an in-code synthetic portfolio and PRINTS the
assembled report so the numbers are inspectable. NEVER places an order, never
uses a token, never touches the network. Real money = HARD STOP.

Run:
    uv run --directory <WORKTREE> python scripts/research/tax_demo.py
"""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

# config/ lives at project root, not under src/; the tax package itself needs no
# such shim, but running as a loose script we ensure src is importable.
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from finalayze.tax.baskets import TaxBase, realized_ytd_base_a
from finalayze.tax.harvest import HarvestCandidate
from finalayze.tax.ldv import LdvHoldingItem, full_years_held
from finalayze.tax.lots import (
    Operation,
    OperationType,
    TaxLot,
    fifo_match,
)
from finalayze.tax.report import ForwardIncome, build_report

TODAY = date(2026, 6, 1)
YEAR = 2026
RUB = "RUB"
USD = "USD"


def _buy(
    figi: str,
    ticker: str,
    qty: Decimal,
    price: Decimal,
    commission: Decimal,
    on: date,
    *,
    currency: str = RUB,
) -> Operation:
    return Operation(
        op_type=OperationType.BUY,
        op_date=on,
        figi=figi,
        ticker=ticker,
        payment=-(qty * price) - commission,
        currency=currency,
        quantity=qty,
        price_per_unit=price,
        commission=commission,
    )


def _sell(
    figi: str,
    ticker: str,
    qty: Decimal,
    price: Decimal,
    commission: Decimal,
    on: date,
) -> Operation:
    return Operation(
        op_type=OperationType.SELL,
        op_date=on,
        figi=figi,
        ticker=ticker,
        payment=(qty * price) - commission,
        currency=RUB,
        quantity=qty,
        price_per_unit=price,
        commission=commission,
    )


def _income(op_type: OperationType, ticker: str, amount: Decimal) -> Operation:
    return Operation(
        op_type=op_type,
        op_date=TODAY,
        figi=f"FIGI-{ticker}",
        ticker=ticker,
        payment=amount,
        currency=RUB,
        payment_is_net_estimate=op_type in {OperationType.DIVIDEND, OperationType.COUPON},
    )


def _lot(
    figi: str,
    ticker: str,
    acquire: date,
    qty: Decimal,
    price: Decimal,
    commission: Decimal,
    *,
    currency: str = RUB,
    on_iis: bool = False,
    cost_known: bool = True,
    russian: bool = True,
) -> TaxLot:
    return TaxLot(
        figi=figi,
        ticker=ticker,
        acquire_date=acquire,
        quantity=qty,
        price_per_unit=price,
        commission_buy=commission,
        currency=currency,
        russian_issuer=russian,
        on_iis=on_iis,
        cost_basis_known=cost_known,
    )


def main() -> None:
    # ---- synthetic portfolio ----
    # 1) SBER: held > 3 years, LDV-eligible, sold at a gain this year (realized)
    sber_buy = _buy("F-SBER", "SBER", Decimal(1000), Decimal(200), Decimal(100), date(2021, 3, 1))
    sber_sell = _sell("F-SBER", "SBER", Decimal(400), Decimal(280), Decimal(120), date(2026, 4, 1))

    # 2) LKOH: still-open lot held > 3 years -> LDV headroom action item
    lkoh_lot = _lot("F-LKOH", "LKOH", date(2020, 2, 1), Decimal(50), Decimal(5000), Decimal(50))

    # 3) OFZ bond with coupons this year (base A, netted with the SBER gain)
    ofz_coupon = _income(OperationType.COUPON, "SU26240", Decimal(34_500))

    # 4) dividend payer (separate base, never netted/harvested)
    lkoh_div = _income(OperationType.DIVIDEND, "LKOH", Decimal(87_000))

    # 5) INPUT_SECURITIES lot: transferred-in, cost/date unknown -> FLAG
    gmkn_lot = _lot(
        "F-GMKN", "GMKN", date(2023, 8, 10), Decimal(30), Decimal(0), Decimal(0), cost_known=False
    )

    # 6) a position at a LOSS -> harvest candidate (also breaks an LDV clock)
    magn_lot = _lot("F-MAGN", "MAGN", date(2022, 9, 1), Decimal(200), Decimal(60), Decimal(20))
    magn_loss = Decimal(-40_000)

    # 7) a USD-denominated bond lot -> FX not computed FLAG
    usd_bond_lot = _lot(
        "F-USD1",
        "RU000A-USD",
        date(2022, 1, 1),
        Decimal(10),
        Decimal(90_000),
        Decimal(0),
        currency=USD,
    )

    open_lots = [lkoh_lot, gmkn_lot, magn_lot, usd_bond_lot]

    # realized YTD base A from FIFO replay of the SBER round-trip
    realized = fifo_match([sber_buy], [sber_sell])
    coupons = [ofz_coupon]
    dividends = [lkoh_div]

    base_a_via_helper = realized_ytd_base_a(realized, coupons)

    # LDV hypothetical finrez (what a full disposal would realize) per figi
    lkoh_full_years = full_years_held(lkoh_lot.acquire_date, TODAY)
    ldv_hyp = {
        "F-LKOH": LdvHoldingItem(positive_finrez=Decimal(500_000), full_years=lkoh_full_years),
    }

    harvest_candidates = [
        HarvestCandidate(lot=magn_lot, unrealized_loss=magn_loss, breaks_ldv_clock=True),
    ]

    forward_income = [
        ForwardIncome(kind=TaxBase.DIVIDENDS, amount=Decimal(20_000), currency=RUB),
        ForwardIncome(kind=TaxBase.SECURITIES, amount=Decimal(15_000), currency=RUB),
        ForwardIncome(kind=TaxBase.SECURITIES, amount=Decimal(5_000), currency=USD),
    ]

    report = build_report(
        today=TODAY,
        year=YEAR,
        open_lots=open_lots,
        realized_ytd=realized,
        coupons_ytd=coupons,
        dividends_ytd=dividends,
        forward_income=forward_income,
        harvest_candidates=harvest_candidates,
        history_truncated=True,
        ldv_hypothetical=ldv_hyp,
    )

    # ---- print the report ----
    print("=" * 88)
    print("SCOPE BANNER")
    print("=" * 88)
    print(report.scope_banner)
    print()
    print(f"Realized YTD base A (gains netted with coupons): {report.realized_ytd_base_a} RUB")
    print(f"  (helper realized_ytd_base_a agrees: {base_a_via_helper} RUB)")
    print(f"  (SBER realized: {sum(r.realized for r in realized)} RUB + OFZ coupon 34500 RUB)")
    print(f"Dividend base YTD (separate, never netted): {report.dividend_ytd} RUB")
    print()
    print("=" * 88)
    print(f"ACTION ITEMS ({len(report.action_items)})  -- NONE places an order")
    print("=" * 88)
    for i, item in enumerate(report.action_items, 1):
        print(
            f"[{i}] {item.category} "
            f"(confidence={item.confidence}, places_order={item.places_order})"
        )
        print(f"    savings_estimate = {item.savings_estimate} RUB")
        print(f"    {item.description}")
        print()
    print("=" * 88)
    print(f"HONEST DEGRADATION FLAGS ({len(report.flags)})")
    print("=" * 88)
    for i, flag in enumerate(report.flags, 1):
        print(f"[{i}] {flag.reason.value}: {flag.detail}")
    print()
    # explicit no-order assertion
    assert all(not a.places_order for a in report.action_items)
    print("INVARIANT CHECK: no action item places an order -> OK")


if __name__ == "__main__":
    main()

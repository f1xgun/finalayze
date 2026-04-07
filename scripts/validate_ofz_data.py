"""Phase 0: Validate T-Bank API provides OFZ bond data for backtest.

Tasks validated:
  0.1: OFZ-PD candle data back to 2022-01-01
  0.2: OFZ coupon schedules
  0.3: Bond instrument metadata (face value, maturity, coupon freq, floating flag)
  0.4: OFZ-PK floater data
  0.5: Accrued interest (NKD) data
  0.6: Data gaps assessment (Feb-Mar 2022 MOEX closure)

Usage:
    uv run python scripts/validate_ofz_data.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# gRPC env vars MUST be set before importing grpc (via t_tech.invest).
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GRPC_ROOTS = _PROJECT_ROOT / "certs" / "grpc_roots.pem"
if _GRPC_ROOTS.exists():
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(_GRPC_ROOTS))

sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from t_tech.invest import AsyncClient, CandleInterval
from t_tech.invest.schemas import InstrumentIdType

_TBANK_GRPC_TARGET = "invest-public-api.tbank.ru:443"
_NANO_DIVISOR = Decimal(1_000_000_000)

# Target OFZ bonds from the plan
OFZ_PD_TICKERS = [
    "SU26238RMFS4",
    "SU26239RMFS2",
    "SU26241RMFS8",
    "SU26243RMFS4",
    "SU26244RMFS2",
    "SU26246RMFS7",
    "SU26252RMFS5",
    "SU26253RMFS3",
]

OFZ_PK_TICKERS = [
    "SU29007RMFS0",
    "SU29008RMFS8",
    "SU29009RMFS6",
    "SU29010RMFS4",
]


def quotation_to_decimal(q) -> Decimal:
    """Convert Tinkoff Quotation(units, nano) to Decimal."""
    return Decimal(q.units) + Decimal(q.nano) / _NANO_DIVISOR


def money_to_decimal(m) -> Decimal:
    """Convert Tinkoff MoneyValue to Decimal."""
    return Decimal(m.units) + Decimal(m.nano) / _NANO_DIVISOR


async def validate_ofz_data() -> dict:  # noqa: PLR0912, PLR0915
    """Run all Phase 0 validation checks."""
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    if not token:
        print("ERROR: FINALAYZE_TINKOFF_TOKEN not set in .env")
        sys.exit(1)

    results: dict = {
        "bonds_found": {},
        "bonds_not_found": [],
        "candle_data": {},
        "coupon_data": {},
        "accrued_interest": {},
        "data_gaps": {},
        "errors": [],
    }

    client = AsyncClient(token, target=_TBANK_GRPC_TARGET)
    async with client as services:
        # ================================================================
        # Task 0.3: Bond instrument metadata
        # ================================================================
        print("\n" + "=" * 70)
        print("TASK 0.3: Bond instrument metadata")
        print("=" * 70)

        all_tickers = OFZ_PD_TICKERS + OFZ_PK_TICKERS
        for ticker in all_tickers:
            try:
                resp = await services.instruments.bond_by(
                    id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                    class_code="TQOB",
                    id=ticker,
                )
                bond = resp.instrument
                nominal = money_to_decimal(bond.nominal)
                initial_nominal = money_to_decimal(bond.initial_nominal)
                aci = money_to_decimal(bond.aci_value)

                info = {
                    "ticker": bond.ticker,
                    "name": bond.name,
                    "figi": bond.figi,
                    "isin": bond.isin,
                    "uid": bond.uid,
                    "nominal": float(nominal),
                    "initial_nominal": float(initial_nominal),
                    "coupon_freq": bond.coupon_quantity_per_year,
                    "floating_coupon": bond.floating_coupon_flag,
                    "maturity_date": str(bond.maturity_date),
                    "placement_date": str(bond.placement_date),
                    "first_candle_date": str(bond.first_1day_candle_date),
                    "currency": bond.currency,
                    "lot": bond.lot,
                    "class_code": bond.class_code,
                    "aci_value": float(aci),
                    "amortization": bond.amortization_flag,
                }
                results["bonds_found"][ticker] = info
                is_pk = "ОФЗ-ПК" if bond.floating_coupon_flag else "ОФЗ-ПД"
                print(
                    f"  OK {ticker}: {bond.name} ({is_pk}), "
                    f"nominal={nominal}, coupon_freq={bond.coupon_quantity_per_year}/yr, "
                    f"maturity={bond.maturity_date}, figi={bond.figi}"
                )
            except Exception as exc:
                results["bonds_not_found"].append(ticker)
                results["errors"].append(f"bond_by({ticker}): {exc}")
                print(f"  FAIL {ticker}: {exc}")

        # ================================================================
        # Task 0.1: OFZ-PD candle data back to 2022-01-01
        # ================================================================
        print("\n" + "=" * 70)
        print("TASK 0.1: OFZ candle data availability (2022-01-01 to now)")
        print("=" * 70)

        start_date = datetime(2022, 1, 1, tzinfo=UTC)
        end_date = datetime.now(tz=UTC)

        for ticker in all_tickers:
            if ticker not in results["bonds_found"]:
                continue

            figi = results["bonds_found"][ticker]["figi"]
            try:
                # T-Bank API limits candle requests to 1 year for daily candles
                # Fetch in yearly chunks
                all_candles = []
                chunk_start = start_date
                while chunk_start < end_date:
                    chunk_end = min(chunk_start + timedelta(days=365), end_date)
                    resp = await services.market_data.get_candles(
                        figi=figi,
                        from_=chunk_start,
                        to=chunk_end,
                        interval=CandleInterval.CANDLE_INTERVAL_DAY,
                    )
                    all_candles.extend(resp.candles)
                    chunk_start = chunk_end

                if all_candles:
                    first_date = all_candles[0].time
                    last_date = all_candles[-1].time
                    first_close = quotation_to_decimal(all_candles[0].close)
                    last_close = quotation_to_decimal(all_candles[-1].close)

                    results["candle_data"][ticker] = {
                        "total_bars": len(all_candles),
                        "first_date": str(first_date),
                        "last_date": str(last_date),
                        "first_close": float(first_close),
                        "last_close": float(last_close),
                    }
                    print(
                        f"  OK {ticker}: {len(all_candles)} daily bars, "
                        f"from {first_date.date()} to {last_date.date()}, "
                        f"close range {first_close:.2f}% - {last_close:.2f}%"
                    )

                    # Check for 2022 MOEX closure gap
                    dates = [c.time.date() for c in all_candles]
                    gap_start = datetime(2022, 2, 24, tzinfo=UTC).date()
                    gap_end = datetime(2022, 3, 28, tzinfo=UTC).date()
                    gap_bars = [d for d in dates if gap_start <= d <= gap_end]
                    if not gap_bars:
                        results["data_gaps"][ticker] = "No data Feb 24 - Mar 28, 2022"
                        print("         GAP: No data during MOEX closure (Feb 24 - Mar 28, 2022)")
                    else:
                        print(f"         Note: {len(gap_bars)} bars during Feb 24 - Mar 28 period")
                else:
                    results["candle_data"][ticker] = {"total_bars": 0}
                    results["errors"].append(f"No candles for {ticker}")
                    print(f"  FAIL {ticker}: No candle data returned")

            except Exception as exc:
                results["errors"].append(f"candles({ticker}): {exc}")
                print(f"  FAIL {ticker}: {exc}")

        # ================================================================
        # Task 0.2: OFZ coupon schedules
        # ================================================================
        print("\n" + "=" * 70)
        print("TASK 0.2: OFZ coupon schedules")
        print("=" * 70)

        for ticker in all_tickers:
            if ticker not in results["bonds_found"]:
                continue

            figi = results["bonds_found"][ticker]["figi"]
            try:
                resp = await services.instruments.get_bond_coupons(
                    figi=figi,
                    from_=start_date,
                    to=datetime(2035, 12, 31, tzinfo=UTC),
                )
                coupons = resp.events
                if coupons:
                    paid_coupons = [
                        c for c in coupons if c.coupon_date.replace(tzinfo=UTC) < end_date
                    ]
                    future_coupons = [
                        c for c in coupons if c.coupon_date.replace(tzinfo=UTC) >= end_date
                    ]

                    # Sample coupon amounts
                    sample = coupons[:3]
                    amounts = [float(money_to_decimal(c.pay_one_bond)) for c in sample]

                    results["coupon_data"][ticker] = {
                        "total_coupons": len(coupons),
                        "paid_coupons": len(paid_coupons),
                        "future_coupons": len(future_coupons),
                        "sample_amounts": amounts,
                        "first_coupon_date": str(coupons[0].coupon_date),
                        "last_coupon_date": str(coupons[-1].coupon_date),
                        "is_floating": results["bonds_found"][ticker]["floating_coupon"],
                    }
                    print(
                        f"  OK {ticker}: {len(coupons)} coupons total "
                        f"({len(paid_coupons)} paid, {len(future_coupons)} future), "
                        f"sample amounts: {amounts}"
                    )
                else:
                    results["coupon_data"][ticker] = {"total_coupons": 0}
                    print(f"  WARN {ticker}: No coupon data returned")

            except Exception as exc:
                results["errors"].append(f"coupons({ticker}): {exc}")
                print(f"  FAIL {ticker}: {exc}")

        # ================================================================
        # Task 0.5: Accrued interest (NKD) data
        # ================================================================
        print("\n" + "=" * 70)
        print("TASK 0.5: Accrued interest (NKD) data")
        print("=" * 70)

        # Check NKD for a recent period
        nkd_start = end_date - timedelta(days=30)
        nkd_end = end_date

        for ticker in all_tickers[:4]:  # Sample first 4
            if ticker not in results["bonds_found"]:
                continue

            figi = results["bonds_found"][ticker]["figi"]
            try:
                resp = await services.instruments.get_accrued_interests(
                    figi=figi,
                    from_=nkd_start,
                    to=nkd_end,
                )
                nkd_list = resp.accrued_interests
                if nkd_list:
                    sample = nkd_list[-3:]
                    nkd_values = [
                        {
                            "date": str(n.date),
                            "value": float(money_to_decimal(n.value)),
                            "value_pct": float(quotation_to_decimal(n.value_percent)),
                        }
                        for n in sample
                    ]
                    results["accrued_interest"][ticker] = {
                        "total_records": len(nkd_list),
                        "sample": nkd_values,
                    }
                    print(f"  OK {ticker}: {len(nkd_list)} NKD records, latest: {nkd_values[-1]}")
                else:
                    print(f"  WARN {ticker}: No NKD data returned")
            except Exception as exc:
                results["errors"].append(f"nkd({ticker}): {exc}")
                print(f"  FAIL {ticker}: {exc}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 0 VALIDATION SUMMARY")
    print("=" * 70)

    total_bonds = len(all_tickers)
    found = len(results["bonds_found"])
    not_found = len(results["bonds_not_found"])
    with_candles = sum(1 for v in results["candle_data"].values() if v.get("total_bars", 0) > 0)
    with_coupons = sum(1 for v in results["coupon_data"].values() if v.get("total_coupons", 0) > 0)
    with_nkd = len(results["accrued_interest"])

    print(f"\n  Bonds found:       {found}/{total_bonds}")
    print(f"  Bonds NOT found:   {not_found}/{total_bonds}")
    if results["bonds_not_found"]:
        print(f"    Missing: {results['bonds_not_found']}")
    print(f"  With candle data:  {with_candles}/{found}")
    print(f"  With coupon data:  {with_coupons}/{found}")
    print(f"  With NKD data:     {with_nkd}/4 (sampled)")

    # Duration assessment
    print("\n  Duration assessment (from maturity dates):")
    today = datetime.now(tz=UTC).date()
    for ticker, info in results["bonds_found"].items():
        mat_str = info["maturity_date"]
        if mat_str and mat_str != "None":
            # Parse "2031-05-14 00:00:00+00:00" style
            mat_date = datetime.fromisoformat(mat_str).date()
            years_to_mat = (mat_date - today).days / 365.25
            max_maturity_years = 5.0
            flag = " *** EXCEEDS 5Y CAP" if years_to_mat > max_maturity_years else ""
            print(f"    {ticker}: {years_to_mat:.1f}Y to maturity{flag}")

    if results["data_gaps"]:
        print("\n  Data gaps (MOEX closure):")
        for ticker, gap in results["data_gaps"].items():
            print(f"    {ticker}: {gap}")

    if results["errors"]:
        print(f"\n  Errors ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"    - {err}")

    # Final verdict
    print("\n" + "-" * 70)
    if found >= total_bonds - 2 and with_candles >= found - 2 and with_coupons >= found - 2:
        print("  VERDICT: PASS — T-Bank API provides sufficient OFZ data for backtesting")
    elif found >= total_bonds // 2:
        print("  VERDICT: PARTIAL — Some bonds missing, plan may need adjustment")
    else:
        print("  VERDICT: FAIL — Insufficient data from T-Bank API")
    print("-" * 70)

    return results


if __name__ == "__main__":
    asyncio.run(validate_ofz_data())

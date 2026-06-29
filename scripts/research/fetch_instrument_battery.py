"""Fetch the instrument-battery panel (iter 1), token-free (autonomous diversification program).

A battery of candidate instruments spanning risk tiers, all from the PUBLIC MOEX ISS REST API
(NO Tinkoff token, NO cert):

- RGBITR  — fixed-coupon OFZ total-return index (duration carry; index endpoint).
- RUCBITR — investment-grade corporate-bond total-return index (credit carry; index endpoint).
- RUCBHYTR — high-yield (ВДО) corporate-bond total-return index (high credit carry; index endpoint).
- LQDT    — money-market fund (near-cash; shares/TQTF board via fetch_close_history(board=...)).
- CNYRUB  — yuan/RUB spot (FX diversifier; currency/selt CETS board).
- MCFTRR  — net equity total-return index (the gate's equity baseline leg; index endpoint).

The index/ETF legs carry the MSK T-1 date convention; the cert shifts them +1 to the true date.
CNYRUB (currency) is already true-dated. Writes a committed snapshot so the gate reproduces offline.

    uv run python scripts/research/fetch_instrument_battery.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from finalayze.data.fetchers.moex_iss import MoexISSFetcher
from finalayze.data.loader import load_mcftr_series

_LOG = structlog.get_logger(__name__)
_OUT = Path("results/research/instrument_battery/panel_snapshot.json")
_INDEX_SECIDS = {
    "rgbitr_ofz_fixed": "RGBITR",
    "rucbitr_corp_ig": "RUCBITR",
    "rucbhytr_corp_hy": "RUCBHYTR",
    "equity_mcftrr": "MCFTRR",
}
_START = datetime(2022, 1, 1, tzinfo=UTC)
_END = datetime(2026, 6, 11, tzinfo=UTC)  # exclusive -> last usable bar 2026-06-10


def main() -> None:
    legs: dict[str, list[list[str]]] = {}
    for key, secid in _INDEX_SECIDS.items():
        rows = load_mcftr_series(secid, _START, _END)
        legs[key] = [[d.isoformat(), str(c)] for d, c in rows]
        _LOG.info("battery_index_fetched", key=key, secid=secid, bars=len(rows))
    with MoexISSFetcher() as fetcher:
        lqdt = fetcher.fetch_close_history("LQDT", _START, _END, board="TQTF")
        legs["lqdt_money_market"] = [[d.isoformat(), str(c)] for d, c in lqdt]
        _LOG.info("battery_lqdt_fetched", bars=len(lqdt))
        cny = fetcher.fetch_currency_close_history("CNYRUB_TOM", _START, _END)
        legs["cnyrub_fx"] = [[d.isoformat(), str(c)] for d, c in cny]
        _LOG.info("battery_cny_fetched", bars=len(cny))

    if not legs["equity_mcftrr"] or not legs["rgbitr_ofz_fixed"]:
        msg = "empty battery fetch"
        raise SystemExit(msg)

    snapshot = {
        "meta": {
            "source": "moex_iss_rest_public (token-free)",
            "index_secids": _INDEX_SECIDS,
            "lqdt": "shares/TQTF board (money-market ETF)",
            "cnyrub": "currency/selt CETS CNYRUB_TOM",
            "note": (
                "index/ETF legs carry the MSK->UTC T-1 convention; the cert shifts index legs +1 "
                "to the true date. LQDT (shares fetch_close_history) + CNYRUB (currency) are "
                "already true-dated. LQDT starts 2023 (tail un-backtestable)."
            ),
        },
        "legs": legs,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    print("wrote", _OUT)
    for key, bars in legs.items():
        span = f"{bars[0][0]}..{bars[-1][0]}" if bars else "EMPTY"
        print(f"  {key}: {len(bars)} bars [{span}]")


if __name__ == "__main__":
    main()

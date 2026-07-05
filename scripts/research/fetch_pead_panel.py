"""Fetch the PEAD panel: MOEX earnings report DATES + daily OHLC (token-gated once).

Post-earnings-announcement-drift (PEAD) test data. Two sources:
- earnings report DATES: Tinkoff ``get_asset_reports`` (calendar-only, NO EPS -- MOEX has
  no consensus feed, D-01). Requires ``FINALAYZE_TINKOFF_TOKEN`` (readonly reference data,
  no orders). Report dates are public corporate-calendar facts, committed to the snapshot.
- daily OHLC: token-free public MOEX ISS-REST for the shares + IMOEX + MCFTRR.

Like ``scripts/build_event_data.py`` this fetch is token-gated and run ONCE; the committed
snapshot lets the cert (``run_pead_gate.py``) reproduce offline & token-free.

A ``t_tech`` SDK bug (``ts_to_datetime`` chokes on an int -- an unset period field the SDK
misdeclares as a Timestamp) is worked around by tolerating ints; ``report_date`` (a real
Timestamp) parses cleanly.

Run:  GRPC_DNS_RESOLVER=native uv run python scripts/research/fetch_pead_panel.py
"""

from __future__ import annotations

import os

# Force gRPC to use the OS resolver BEFORE any grpc import (c-ares fails for tbank.ru).
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import t_tech.invest._grpc_helpers as _gh
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "results" / "research" / "pead" / "pead_panel.json"

# MOEX blue chips + registry FIGIs (from build_event_data.MOEX_FIGIS).
FIGIS: dict[str, str] = {
    "SBER": "BBG004730N88",
    "GAZP": "BBG004730RP0",
    "LKOH": "BBG004731032",
    "VTBR": "BBG004730ZJ9",
    "SBERP": "BBG0047315Y7",
    "ROSN": "BBG004731354",
    "TATN": "BBG004RVFFC0",
    "NVTK": "BBG00475KKY8",
    "GMKN": "BBG004731489",
    "MGNT": "BBG004RVFCY3",
    "ALRS": "BBG004S68B31",
    "SNGS": "BBG004S681W1",
    "TRNFP": "BBG00475K6C3",
    "IRAO": "BBG004S68473",
}
_START = "2024-01-01"
_END = "2026-07-05"
_PAGE = 100
_TIMEOUT_S = 30
_SHARE_BASE = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities"
_INDEX_BASE = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities"

# ── SDK bug workaround: tolerate an int where a Timestamp is expected. ─────────
_orig_ts = _gh.ts_to_datetime


def _safe_ts(value: object) -> object:
    return None if isinstance(value, int) else _orig_ts(value)


_gh.ts_to_datetime = _safe_ts  # type: ignore[assignment]

from t_tech.invest.schemas import GetAssetReportsRequest

from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import build_default_registry


def _iss_ohlc(base: str, secid: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    start = 0
    cols = "TRADEDATE,OPEN,CLOSE"
    while True:
        params = urllib.parse.urlencode(
            {
                "from": _START,
                "till": _END,
                "iss.only": "history",
                "iss.meta": "off",
                "history.columns": cols,
                "start": start,
            }
        )
        url = f"{base}/{secid}.json?{params}"
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            payload = json.load(resp)
        page = payload["history"]["data"]
        if not page:
            break
        for tradedate, open_, close in page:
            if close is None:
                continue
            rows.append(
                {"d": tradedate, "o": str(open_ if open_ is not None else close), "c": str(close)}
            )
        if len(page) < _PAGE:
            break
        start += len(page)
    return rows


def _fetch_earnings_dates(fetcher: TinkoffFetcher, figi: str) -> list[str]:
    async def _run() -> list[str]:
        services = await fetcher._get_services_async()
        now = datetime.now(tz=UTC)
        resp = await services.instruments.get_asset_reports(  # type: ignore[attr-defined]
            GetAssetReportsRequest(
                instrument_id=figi, from_=now - timedelta(days=730), to=now + timedelta(days=90)
            )
        )
        out: list[str] = []
        for ev in resp.events:
            rd = getattr(ev, "report_date", None)
            if rd is not None:
                out.append(rd.date().isoformat())
        return sorted(set(out))

    return fetcher._run_async(_run())  # type: ignore[return-value]


def main() -> None:
    token = dotenv_values(PROJECT_ROOT / ".env").get("FINALAYZE_TINKOFF_TOKEN") or os.environ.get(
        "FINALAYZE_TINKOFF_TOKEN"
    )
    if not token:
        msg = "FINALAYZE_TINKOFF_TOKEN required for the earnings-date calendar"
        raise SystemExit(msg)

    registry = build_default_registry()
    fetcher = TinkoffFetcher(token=token, registry=registry, sandbox=False)

    earnings: dict[str, list[str]] = {}
    prices: dict[str, list[dict[str, str]]] = {}
    try:
        for sym, figi in FIGIS.items():
            try:
                dates = _fetch_earnings_dates(fetcher, figi)
            except Exception as exc:
                print(f"  {sym}: earnings-date fetch FAILED ({type(exc).__name__}); skipping dates")
                dates = []
            earnings[sym] = dates
            prices[sym] = _iss_ohlc(_SHARE_BASE, sym)
            print(f"  {sym}: {len(dates)} report dates, {len(prices[sym])} price bars")
    finally:
        fetcher.close()

    imoex = _iss_ohlc(_INDEX_BASE, "IMOEX")
    mcftrr = _iss_ohlc(_INDEX_BASE, "MCFTRR")
    print(f"  IMOEX: {len(imoex)} bars | MCFTRR: {len(mcftrr)} bars")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "source": "Tinkoff get_asset_reports (dates) + MOEX ISS-REST (prices)",
        "span": {"from": _START, "till": _END},
        "earnings_dates": earnings,
        "prices": prices,
        "imoex": imoex,
        "mcftrr": mcftrr,
    }
    OUT.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True) + "\n")
    total_events = sum(len(v) for v in earnings.values())
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} ({total_events} total report dates)")


if __name__ == "__main__":
    main()

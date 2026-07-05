"""Fetch the daily OHLC panel for the news-event study (token-free MOEX ISS-REST).

Pulls daily open/low/high/close for every share and the IMOEX benchmark touched by the
pre-registered event set, over one span covering all events, and writes a committed
snapshot the deterministic cert (:mod:`scripts.research.run_event_study`) reads offline.

NO Tinkoff token, NO real money, NO orders -- public ISS-REST only. The event
definitions (dates, tickers, the NAIVE predicted direction a retail reader would bet)
live here so the snapshot is the single source of truth for the cert.

Run:  uv run python scripts/research/fetch_event_study_panel.py
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "results" / "research" / "event_study" / "panel_snapshot.json"

# ── Pre-registered event set ──────────────────────────────────────────────────
# direction = the NAIVE prediction a retail reader would make from the headline
# (+1 up, -1 down). The cert MEASURES whether that prediction was even correct and,
# if so, whether it was tradeable -- direction never presupposes the answer.
# entry_mode: "intraday" (headline broke during the anchor session) or "overnight"
# (broke after close / over a weekend -> price first moves at the anchor's OPEN gap).
EVENTS = [
    {
        "key": "gazp_div_cancel_2022",
        "label": "GAZP 2021 dividend cancelled (AGM)",
        "anchor": "2022-06-30",
        "tickers": ["GAZP"],
        "benchmark": "IMOEX",
        "direction": -1,
        "entry_mode": "intraday",
        "note": "Shareholders voted down the record 2021 dividend intraday; single-name shock.",
    },
    {
        "key": "fuel_export_ban_2023",
        "label": "Gasoline/diesel export ban",
        "anchor": "2023-09-21",
        "tickers": ["ROSN", "LKOH", "SIBN", "TATN", "SNGS", "GAZP"],
        "benchmark": "IMOEX",
        "direction": -1,
        "entry_mode": "intraday",
        "note": "Government banned fuel exports (operator's fuel example); DIRECTION AMBIGUOUS.",
    },
    {
        "key": "vk_appstore_2022",
        "label": "VK apps pulled from Apple App Store",
        "anchor": "2022-09-26",
        "tickers": ["VKCO"],
        "benchmark": "IMOEX",
        "direction": -1,
        "entry_mode": "intraday",
        "note": "Operator's VK example; CONFOUNDED by the 21 Sep mobilisation crash.",
    },
    {
        "key": "sber_div_reco_2023",
        "label": "SBER record dividend recommended",
        "anchor": "2023-03-17",
        "tickers": ["SBER"],
        "benchmark": "IMOEX",
        "direction": 1,
        "entry_mode": "intraday",
        "note": "Supervisory board recommended a record dividend; positive single-name shock.",
    },
    {
        "key": "wagner_mutiny_2023",
        "label": "Wagner mutiny weekend",
        "anchor": "2023-06-26",
        "tickers": ["IMOEX"],
        "benchmark": None,
        "direction": -1,
        "entry_mode": "overnight",
        "note": "Broke Fri night / Sat 24 Jun; market first trades Mon 26 Jun -> pure open gap.",
    },
]

_SHARE_BASE = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities"
_INDEX_BASE = "https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities"
_SHARE_SECIDS = ["GAZP", "ROSN", "LKOH", "SIBN", "TATN", "SNGS", "VKCO", "SBER"]
_INDEX_SECIDS = ["IMOEX"]
_START = "2022-06-01"
_END = "2023-11-15"
_PAGE = 100
_TIMEOUT_S = 30


def _fetch_history(base: str, secid: str) -> list[dict[str, str]]:
    """Fetch paginated daily history for one secid; return list of str-valued OHLC rows."""
    rows: list[dict[str, str]] = []
    start = 0
    while True:
        params = urllib.parse.urlencode(
            {
                "from": _START,
                "till": _END,
                "iss.only": "history",
                "iss.meta": "off",
                "history.columns": "TRADEDATE,OPEN,LOW,HIGH,CLOSE,VOLUME",
                "start": start,
            }
        )
        url = f"{base}/{secid}.json?{params}"
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:  # noqa: S310 (trusted ISS host)
            payload = json.load(resp)
        page = payload["history"]["data"]
        if not page:
            break
        for tradedate, open_, low, high, close, volume in page:
            if close is None or open_ is None:
                continue  # a no-trade session -> skip rather than fabricate a price
            rows.append(
                {
                    "d": tradedate,
                    "o": str(open_),
                    "l": str(low),
                    "h": str(high),
                    "c": str(close),
                    "v": str(volume),
                }
            )
        if len(page) < _PAGE:
            break
        start += len(page)
    return rows


def main() -> None:
    prices: dict[str, list[dict[str, str]]] = {}
    for secid in _SHARE_SECIDS:
        prices[secid] = _fetch_history(_SHARE_BASE, secid)
        print(f"  {secid}: {len(prices[secid])} bars")
    for secid in _INDEX_SECIDS:
        prices[secid] = _fetch_history(_INDEX_BASE, secid)
        print(f"  {secid}: {len(prices[secid])} bars")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "source": "MOEX ISS-REST (token-free public history)",
        "span": {"from": _START, "till": _END},
        "events": EVENTS,
        "prices": prices,
    }
    OUT.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

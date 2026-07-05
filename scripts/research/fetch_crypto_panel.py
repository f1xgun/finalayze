"""Fetch the crypto research panel — token-free, public, READ-ONLY. Run ONCE, commit the snapshot.

Two independent measurements need three public data pulls (no API keys, no orders — real money is a
hard stop):

  1. CROSS-EXCHANGE ARBITRAGE — poll the top-of-book (best bid/ask) for BTC/USDT across five public
     venues (Kraken, Coinbase, Bybit, OKX, Binance) over a short window. Yields the distribution of
     the best realisable cross-venue spread, which the cert compares against round-trip taker fees +
     withdrawal cost + the capital-lockup carry (the deposit forgone on pre-positioned funds).

  2. TREND SLEEVE — Binance daily klines for BTC/ETH (2021+) give a RUB investor's crypto price
     path once converted with the CBR official USD/RUB rate. The cert builds a time-series-momentum
     sleeve, nets costs + 13% NDFL, and runs it through the canonical Instrument Integration Gate
     against the same MCFTRR equity leg the gold/ZO/real-estate battery used.

This writes ``results/research/crypto/crypto_panel.json`` deterministically consumed offline by
``run_crypto_gate.py``. The arb poll is a within-session microstructure snapshot (calm market,
single time-of-day) — the cert reports that limit honestly; the structural conclusion (fees vs
top-of-book spread) does not depend on the window.

    uv run python scripts/research/fetch_crypto_panel.py
"""

from __future__ import annotations

import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

_DIR = Path("results/research/crypto")
_OUT = _DIR / "crypto_panel.json"

_ARB_ROUNDS = 45
_ARB_SLEEP_S = 2.0
_KLINE_START = "2021-01-01"
_HTTP_TIMEOUT = 15.0
_MS_PER_DAY = 86_400_000


def _get(url: str, encoding: str = "utf-8") -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "finalayze-research/1.0"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310 (public GET)
        return resp.read().decode(encoding, "replace")


# ── 1. cross-venue top-of-book (best bid/ask) ────────────────────────────────
def _kraken() -> tuple[float, float]:
    r = json.loads(_get("https://api.kraken.com/0/public/Ticker?pair=XBTUSDT"))["result"]
    row = next(iter(r.values()))
    return float(row["b"][0]), float(row["a"][0])


def _coinbase() -> tuple[float, float]:
    r = json.loads(_get("https://api.exchange.coinbase.com/products/BTC-USDT/ticker"))
    return float(r["bid"]), float(r["ask"])


def _bybit() -> tuple[float, float]:
    r = json.loads(_get("https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT"))
    row = r["result"]["list"][0]
    return float(row["bid1Price"]), float(row["ask1Price"])


def _okx() -> tuple[float, float]:
    r = json.loads(_get("https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT"))["data"][0]
    return float(r["bidPx"]), float(r["askPx"])


def _binance_book() -> tuple[float, float]:
    r = json.loads(_get("https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT"))
    return float(r["bidPrice"]), float(r["askPrice"])


_VENUES = {
    "kraken": _kraken,
    "coinbase": _coinbase,
    "bybit": _bybit,
    "okx": _okx,
    "binance": _binance_book,
}


def _poll_arb() -> list[dict[str, object]]:
    rounds: list[dict[str, object]] = []
    for i in range(_ARB_ROUNDS):
        quotes: dict[str, list[float]] = {}
        for name, fn in _VENUES.items():
            try:
                bid, ask = fn()
                quotes[name] = [bid, ask]
            except Exception:
                continue
        rounds.append({"round": i, "quotes": quotes})
        if i < _ARB_ROUNDS - 1:
            time.sleep(_ARB_SLEEP_S)
    return rounds


# ── 2. Binance daily klines (USD price path) ─────────────────────────────────
def _klines(symbol: str) -> list[list[str]]:
    start_ms = int(datetime.fromisoformat(_KLINE_START).replace(tzinfo=UTC).timestamp() * 1000)
    out: list[list[str]] = []
    while True:
        url = (
            f"https://api.binance.com/api/v3/klines?symbol={symbol}"
            f"&interval=1d&startTime={start_ms}&limit=1000"
        )
        batch = json.loads(_get(url))
        if not batch:
            break
        for row in batch:
            d = datetime.fromtimestamp(row[0] / 1000, tz=UTC).date().isoformat()
            out.append([d, str(row[4])])  # close
        last_open = batch[-1][0]
        if len(batch) < 1000:  # noqa: PLR2004 (Binance page size)
            break
        start_ms = last_open + _MS_PER_DAY
        time.sleep(0.3)
    # de-dup on date (paging boundary safety), keep last
    dedup: dict[str, str] = dict(out)
    return [[d, dedup[d]] for d in sorted(dedup)]


# ── 3. CBR official USD/RUB (RUB numeraire) ──────────────────────────────────
def _usdrub() -> list[list[str]]:
    d1 = datetime.fromisoformat(_KLINE_START).strftime("%d/%m/%Y")
    d2 = datetime.now(tz=UTC).strftime("%d/%m/%Y")
    url = (
        f"https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={d1}&date_req2={d2}&VAL_NM_RQ=R01235"
    )
    body = _get(url, encoding="windows-1251")
    body = body.split("?>", 1)[-1]  # drop the windows-1251 XML decl for the utf parser
    root = ET.fromstring(body)  # noqa: S314 (trusted CBR endpoint, no entities)
    rows: list[list[str]] = []
    for rec in root.findall("Record"):
        dd = rec.get("Date")
        rate = rec.findtext("VunitRate")
        if not dd or not rate:
            continue
        iso = datetime.strptime(dd, "%d.%m.%Y").date().isoformat()  # noqa: DTZ007 (date-only)
        rows.append([iso, rate.replace(",", ".")])
    return sorted(rows)


def main() -> None:
    print("polling cross-venue top-of-book...")
    arb = _poll_arb()
    print(f"  {len(arb)} rounds")
    print("fetching daily klines...")
    ohlc = {sym: _klines(sym) for sym in ("BTCUSDT", "ETHUSDT")}
    for sym, rows in ohlc.items():
        print(f"  {sym}: {len(rows)} bars {rows[0][0]}..{rows[-1][0]}")
    print("fetching CBR USD/RUB...")
    fx = _usdrub()
    print(f"  usdrub: {len(fx)} bars {fx[0][0]}..{fx[-1][0]}")

    panel = {
        "meta": {
            "fetched": datetime.now(tz=UTC).date().isoformat(),
            "sources": {
                "arb_top_of_book": list(_VENUES),
                "klines": "binance /api/v3/klines interval=1d (USDT-quoted)",
                "usdrub": "CBR XML_dynamic R01235 (official daily fixing)",
            },
            "disclaimer": (
                "Public read-only market data. No orders, no API keys. The arb poll is a "
                "within-session microstructure snapshot; the klines/fx are the RUB investor's "
                "real price path. Real-money execution is a hard stop."
            ),
            "arb_rounds": _ARB_ROUNDS,
            "arb_sleep_s": _ARB_SLEEP_S,
        },
        "arb_snapshots": arb,
        "ohlc": ohlc,
        "usdrub": fx,
    }
    _DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(panel, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {_OUT} ({_OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

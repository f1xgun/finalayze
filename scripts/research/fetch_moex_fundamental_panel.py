"""Fetch the MOEX fundamental factor panel: SmartLab IFRS statements + ISS prices/dividends.

READ-ONLY, public sources only:
- SmartLab (smart-lab.ru) annual МСФО (IFRS) financials — robots-gated; the same
  source our production ``smartlab_fundamentals.py`` already uses. Gives a consistent
  IFRS-consolidated basis for the whole IMOEX universe (incl. metals & miners that
  are absent from RFSD RAS). Depth is shallow (~5-6 recent fiscal years).
- MOSCOW EXCHANGE ISS (iss.moex.com) public REST — daily close history + dividends
  for forward-return construction. Official exchange data, token-free (the same
  public ISS path used by the Phase-73/74 live certs). NOT yfinance.

Output: a committed snapshot ``results/research/moex_fundamental/panel.json`` so the
study re-runs offline. Nothing here places an order or touches a broker.

Honest caveats baked into the data (surfaced by the study, not hidden here):
SmartLab is survivor-biased (no delisted tickers), serves as-of-today / possibly
restated values (only the "Дата отчёта" disclosure date is frozen), and its shallow
depth pins the panel to the post-2022 regime.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx
from lxml import html as lxml_html

_HTTP_OK = 200
_ISS_PAGE = 100
_MIN_YEAR_CELLS = 2

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_SL_BASE = "https://smart-lab.ru"
_SL_TMPL = "https://smart-lab.ru/q/{t}/f/y/MSFO/"
_ISS = "https://iss.moex.com/iss"
_PRICE_FROM = "2020-01-01"
_PRICE_TILL = "2026-07-31"

# SmartLab @field rows we keep. GP/A = (revenue - cost_of_production)/assets.
_FIELDS = (
    "revenue",
    "cost_of_production",
    "operating_income",
    "ebitda",
    "net_income",
    "assets",
    "net_assets",
    "net_debt",
    "market_cap",
    "ev_ebitda",
    "roe",
    "roa",
)
_YEAR_RE = re.compile(r"^\d{4}$")
_SEPS = ("\u00a0", "\u2009", " ")  # nbsp, thin space, regular space

_OUT = Path("results/research/moex_fundamental")


def _num(s: str) -> float | None:
    s = (s or "").strip()
    for sep in _SEPS:
        s = s.replace(sep, "")
    if s in ("", "-", "n/a", "N/A", "?"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _robots_ok(path: str) -> bool:
    rp = RobotFileParser()
    rp.set_url(f"{_SL_BASE}/robots.txt")
    try:
        rp.read()
    except Exception:  # if robots unreadable, fail closed
        return False
    return rp.can_fetch(_UA, f"{_SL_BASE}{path}")


def _parse_smartlab(html: str) -> list[dict[str, object]]:
    doc = lxml_html.fromstring(html)
    tabs = doc.xpath("//table[contains(@class,'financials')]")
    if not tabs:
        return []
    tb = tabs[0]

    # Year header: the row whose cells contain >=2 bare 4-digit fiscal years.
    years: list[int] = []
    for tr in tb.xpath(".//tr"):
        cells = [c.text_content().strip() for c in tr.xpath("./*")]
        yrs = [int(c) for c in cells if _YEAR_RE.match(c)]
        if len(yrs) >= _MIN_YEAR_CELLS:
            years = yrs
            break
    if not years:
        return []

    def _row_cells(field: str) -> list[str]:
        trs = tb.xpath(f".//tr[@field='{field}']")
        if not trs:
            return []
        return [td.text_content().strip() for td in trs[0].xpath("./td")]

    dates = _row_cells("date")
    field_cells = {f: _row_cells(f) for f in _FIELDS}

    records: list[dict[str, object]] = []
    for i, fy in enumerate(years):
        as_of = dates[i].strip() if i < len(dates) else ""
        # cross-check: annual report disclosed in FY+1 (skip if wildly off, keep FY from header)
        rec: dict[str, object] = {"fiscal_year": fy, "as_of": as_of}
        for f in _FIELDS:
            cells = field_cells[f]
            rec[f] = _num(cells[i]) if i < len(cells) else None
        records.append(rec)
    return records


def _fetch_smartlab(client: httpx.Client, ticker: str) -> list[dict[str, object]]:
    path = f"/q/{ticker}/f/y/MSFO/"
    if not _robots_ok(path):
        print(f"  robots.txt disallows {path} — skipping", file=sys.stderr)
        return []
    r = client.get(_SL_TMPL.format(t=ticker), headers={"User-Agent": _UA}, timeout=30.0)
    if r.status_code != _HTTP_OK:
        print(f"  smartlab {ticker} HTTP {r.status_code}", file=sys.stderr)
        return []
    return _parse_smartlab(r.text)


def _iss_prices(client: httpx.Client, ticker: str) -> dict[str, float]:
    out: dict[str, float] = {}
    start = 0
    while True:
        url = (
            f"{_ISS}/history/engines/stock/markets/shares/securities/{ticker}.json"
            f"?iss.meta=off&iss.only=history"
            f"&history.columns=TRADEDATE,CLOSE,LEGALCLOSEPRICE"
            f"&from={_PRICE_FROM}&till={_PRICE_TILL}&start={start}"
        )
        r = client.get(url, timeout=30.0)
        if r.status_code != _HTTP_OK:
            break
        data = r.json().get("history", {}).get("data", [])
        if not data:
            break
        for row in data:
            d, close, legal = row[0], row[1], row[2]
            px = close if close is not None else legal
            if px is not None:
                out[d] = float(px)
        start += len(data)
        if len(data) < _ISS_PAGE:
            break
        time.sleep(0.1)
    return out


def _iss_dividends(client: httpx.Client, ticker: str) -> list[dict[str, object]]:
    url = f"{_ISS}/securities/{ticker}/dividends.json?iss.meta=off"
    r = client.get(url, timeout=30.0)
    if r.status_code != _HTTP_OK:
        return []
    blk = r.json().get("dividends", {})
    cols = blk.get("columns", [])
    if not cols:
        return []
    ci = {c: i for i, c in enumerate(cols)}
    out: list[dict[str, object]] = []
    for row in blk.get("data", []):
        cur = row[ci["currencyid"]] if "currencyid" in ci else "RUB"
        if cur not in ("RUB", "SUR"):
            continue
        out.append(
            {
                "registryclosedate": row[ci.get("registryclosedate", 0)],
                "value": row[ci.get("value", 0)],
            }
        )
    return out


def _tickers_from_crosswalk() -> list[str]:
    path = Path("docs/research/moex_fundamental_crosswalk.csv")
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        tickers.append(line.split(",", 1)[0].strip())
    return tickers


def main() -> None:
    tickers = _tickers_from_crosswalk()
    print(f"fetching {len(tickers)} tickers")
    fundamentals: dict[str, list[dict[str, object]]] = {}
    prices: dict[str, dict[str, float]] = {}
    dividends: dict[str, list[dict[str, object]]] = {}
    with httpx.Client(follow_redirects=True) as client:
        for t in tickers:
            print(f"- {t}")
            fundamentals[t] = _fetch_smartlab(client, t)
            time.sleep(0.3)
            prices[t] = _iss_prices(client, t)
            dividends[t] = _iss_dividends(client, t)
            time.sleep(0.2)
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "panel.json").write_text(
        json.dumps(
            {
                "meta": {
                    "source_fundamentals": "smart-lab.ru МСФО (IFRS) annual",
                    "source_prices": "iss.moex.com history (official, public)",
                    "price_window": [_PRICE_FROM, _PRICE_TILL],
                    "n_tickers": len(tickers),
                },
                "fundamentals": fundamentals,
                "prices": prices,
                "dividends": dividends,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    n_fund = sum(1 for t in tickers if fundamentals.get(t))
    n_px = sum(1 for t in tickers if prices.get(t))
    print(f"done: fundamentals for {n_fund}/{len(tickers)}, prices for {n_px}/{len(tickers)}")
    print(f"-> {_OUT / 'panel.json'}")


if __name__ == "__main__":
    main()

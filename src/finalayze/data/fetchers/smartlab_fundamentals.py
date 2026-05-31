"""SmartLab fundamentals fetcher (Layer 2).

Scrapes per-quarter company fundamentals from smart-lab.ru and parses them into
look-ahead-safe ``FundamentalSnapshot`` rows (BACKFILL-H-01).

Mirrors ``cbr.py``'s shape: httpx + lxml, sync, browser User-Agent, retry loop,
optional ``RateLimiter``, ``__enter__/__exit__``. Does NOT extend BaseFetcher
(SmartLab is an HTML scrape, not the candle contract).

Robustness seams:
- Parse by the stable ``@field`` attribute, never by Russian label or column index
  (RESEARCH Pattern 1 / A2). Banks (SBER) and industrials (LKOH) differ in row set:
  ``.get()`` per field → absent rows leave the schema field ``None`` (never fabricated).
- A runtime ``robots.txt`` gate (``urllib.robotparser``) HARD-STOPS with
  ``DataFetchError`` before any pull if the path is disallowed (BACKFILL-H-04).
- On-disk HTML cache so re-runs do not re-hit the network (D-02).
- Per-quarter ``as_of`` = the explicit "Дата отчета" cell when present, else the
  +75d publication-lag fallback — NEVER the fiscal-quarter-end (BACKFILL-H-02).
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.robotparser import RobotFileParser

import httpx
import structlog
from lxml import html as lxml_html

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import FundamentalSnapshot
from finalayze.data.fundamental_publication_dates import (
    get_effective_annual_disclosure_date,
    get_effective_disclosure_date,
)

if TYPE_CHECKING:
    from finalayze.data.rate_limiter import RateLimiter

_log = structlog.get_logger()

_BASE_URL = "https://smart-lab.ru"
_ROBOTS_URL = f"{_BASE_URL}/robots.txt"
_URL_TMPL = "https://smart-lab.ru/q/{symbol}/f/q/{statement}/"  # statement in {MSFO, RSBU}
# Annual route mirrors the quarterly one but on /f/y/ — exposes ~10 fiscal years
# of DEPTH vs the quarterly page's ~5 recent quarters (BACKFILL-Y-01).
_URL_TMPL_ANNUAL = "https://smart-lab.ru/q/{symbol}/f/y/{statement}/"
_FINANCIALS_XPATH = "//table[contains(@class,'financials')]"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0
_HTTP_TIMEOUT = 30.0
_CACHE_DIR = Path(".cache/smartlab")

# Fiscal quarter / LTM column labels. Only the 4 fiscal quarters become snapshots;
# the LTM column is not a point-in-time fiscal period.
_QUARTER_RE = re.compile(r"^\d{4}Q[1-4]$")

# Annual column header: a bare 4-digit fiscal year. The trailing LTM/TTM column
# is NOT a fiscal year (it fails ^\d{4}$) and is therefore excluded.
_YEAR_RE = re.compile(r"^\d{4}$")

# Billions-of-rubles labelled cells (label "mlrd rub") scale to raw RUB for the
# schema's Numeric(20,2) fields (revenue_ttm, market_cap).
_BILLION = Decimal("1e9")

# Thousands separators SmartLab uses: U+00A0 (nbsp), U+2009 (thin space), plain space.
_THOUSANDS_SEPARATORS = ("\u00a0", "\u2009", " ")

# A bare-zero dividend yield ("0.0%") means "no dividend reported"; treat as absent
# rather than fabricating a 0.0 yield (RESEARCH Field Mapping: ignore 0.0%).
_ZERO_YIELD = Decimal(0)


class SmartlabFundamentalsFetcher:
    """SmartLab per-quarter fundamentals scraper. Sync. Does NOT extend BaseFetcher."""

    def __init__(self, rate_limiter: RateLimiter | None = None) -> None:
        self._client = httpx.Client(timeout=_HTTP_TIMEOUT, headers=_HEADERS)
        self._rate_limiter = rate_limiter
        self._user_agent = _HEADERS["User-Agent"]

    def close(self) -> None:
        """Close underlying httpx.Client."""
        self._client.close()

    def __enter__(self) -> SmartlabFundamentalsFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── robots.txt runtime gate (BACKFILL-H-04) ──────────────────────────────

    def assert_robots_allowed(self, path: str) -> None:
        """Hard-stop with ``DataFetchError`` if robots.txt disallows *path*.

        Parses smart-lab.ru/robots.txt at runtime and checks ``can_fetch`` for the
        configured User-Agent. MUST be called BEFORE any GET (RESEARCH Pattern 2).
        """
        rp = RobotFileParser()
        rp.set_url(_ROBOTS_URL)
        rp.read()
        if not rp.can_fetch(self._user_agent, f"{_BASE_URL}{path}"):
            msg = f"robots.txt disallows {path} — aborting (BACKFILL-H-04)"
            raise DataFetchError(msg)

    # ── Fetch (polite + cached) ───────────────────────────────────────────────

    def fetch_html(self, symbol: str, statement: str = "MSFO") -> str:
        """Fetch *symbol*'s raw fundamentals HTML (robots-gated + cached).

        Robots gate runs FIRST (hard-stop before any pull, BACKFILL-H-04). On-disk
        cache short-circuits a re-run so it does not re-hit the network (D-02).
        Returns the HTML string; parsing is the caller's concern (see :meth:`fetch`
        and the Plan-04 backfill driver, which composes ``fetch_html`` + ``parse_html``).
        """
        path = f"/q/{symbol}/f/q/{statement}/"
        self.assert_robots_allowed(path)

        cached = self._read_cache(symbol, statement)
        if cached is not None:
            return cached

        url = _URL_TMPL.format(symbol=symbol, statement=statement)
        content = self._request("GET", url).decode("utf-8", errors="replace")
        self._write_cache(symbol, statement, content)
        return content

    def fetch(self, symbol: str, statement: str = "MSFO") -> list[FundamentalSnapshot]:
        """Fetch and parse *symbol*'s per-quarter fundamentals.

        Thin composition of :meth:`fetch_html` (robots gate + cache + GET) and
        :meth:`parse_html`.
        """
        return self.parse_html(self.fetch_html(symbol, statement), symbol)

    def fetch_html_annual(self, symbol: str, statement: str = "MSFO") -> str:
        """Fetch *symbol*'s raw ANNUAL fundamentals HTML (robots-gated + cached).

        Mirrors :meth:`fetch_html` but on the ``/f/y/`` annual route. The ``/f/y/``
        robots gate runs FIRST, before any pull (BACKFILL-Y-04) — the quarterly
        gate only covered ``/f/q/``. The annual cache key is namespaced (``period="y"``)
        so it never clobbers the quarterly cache file (RESEARCH Pitfall 1).
        """
        path = f"/q/{symbol}/f/y/{statement}/"
        self.assert_robots_allowed(path)

        cached = self._read_cache(symbol, statement, period="y")
        if cached is not None:
            return cached

        url = _URL_TMPL_ANNUAL.format(symbol=symbol, statement=statement)
        content = self._request("GET", url).decode("utf-8", errors="replace")
        self._write_cache(symbol, statement, content, period="y")
        return content

    def _cache_path(self, symbol: str, statement: str, period: str = "q") -> Path:
        # Quarterly stays "SBER_MSFO.html"; annual becomes "SBER_MSFO_y.html".
        suffix = "" if period == "q" else f"_{period}"
        return _CACHE_DIR / f"{symbol}_{statement}{suffix}.html"

    def _read_cache(self, symbol: str, statement: str, period: str = "q") -> str | None:
        path = self._cache_path(symbol, statement, period)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None

    def _write_cache(self, symbol: str, statement: str, content: str, period: str = "q") -> None:
        path = self._cache_path(symbol, statement, period)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _request(self, method: str, url: str, **kwargs: Any) -> bytes:
        """HTTP request with rate limiting, retry, and error wrapping (mirrors cbr.py)."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            if self._rate_limiter:
                self._rate_limiter.acquire()
            try:
                resp = self._client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp.content
            except httpx.TimeoutException as exc:
                last_exc = exc
                _log.warning("smartlab_timeout", attempt=attempt + 1, url=url)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                _log.warning(
                    "smartlab_http_error", status=exc.response.status_code, attempt=attempt + 1
                )
            except httpx.RequestError as exc:
                last_exc = exc
                _log.warning("smartlab_network_error", attempt=attempt + 1, url=url)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE * (2**attempt))

        cause = "network_error"
        if isinstance(last_exc, httpx.TimeoutException):
            cause = "timeout"
        elif isinstance(last_exc, httpx.HTTPStatusError):
            cause = "http_error"
        msg = f"SmartLab {cause}: {last_exc}"
        raise DataFetchError(msg) from last_exc

    # ── @field parser (RESEARCH Pattern 1) ────────────────────────────────────

    def parse_html(self, content: str, symbol: str) -> list[FundamentalSnapshot]:
        """Parse SmartLab fundamentals HTML into one ``FundamentalSnapshot`` per quarter.

        Rows are keyed by their stable ``@field`` attribute (drift-resilient); quarter
        columns are located by quarter-regex on the period-header row (NOT a hard-coded
        index). Banks lack ``revenue``/``ev_ebitda`` rows → those fields stay ``None``.
        """
        doc = lxml_html.fromstring(content)
        tables: Any = doc.xpath(_FINANCIALS_XPATH)
        if not tables:
            return []
        rows: Any = tables[0].xpath(".//tr")

        # Locate the period-header row defensively: the row whose cells include
        # quarter labels like "2025Q1" (RESEARCH A2 — do not hard-code an index).
        quarter_cols: dict[int, str] = {}
        for row in rows:
            cells = [self._cell_text(c) for c in row.xpath("./th|./td")]
            candidate = {i: lbl for i, lbl in enumerate(cells) if _QUARTER_RE.match(lbl)}
            if candidate:
                quarter_cols = candidate
                break
        if not quarter_cols:
            return []

        # Build {field: [cell, ...]} skipping rows without a @field attribute.
        by_field: dict[str, list[str]] = {}
        for row in rows:
            field = row.get("field")
            if not field:
                continue
            by_field[field] = [self._cell_text(c) for c in row.xpath("./th|./td")]

        date_cells = by_field.get("date", [])

        snapshots: list[FundamentalSnapshot] = []
        for col_idx, period in quarter_cols.items():
            as_of = self._resolve_as_of(date_cells, col_idx, symbol, period)
            snapshots.append(
                FundamentalSnapshot(
                    symbol=symbol,
                    as_of=as_of,
                    pe_ratio=self._number(by_field, "p_e", col_idx),
                    ev_ebitda=self._number(by_field, "ev_ebitda", col_idx),
                    revenue_ttm=self._billions(by_field, "revenue", col_idx),
                    net_margin=self._percent(by_field, "net_margin", col_idx),
                    roe=self._percent(by_field, "roe", col_idx),
                    eps_ttm=self._number(by_field, "eps", col_idx),
                    dividend_yield=self._yield(by_field, "div_yield", col_idx),
                    market_cap=self._billions(by_field, "market_cap", col_idx),
                    currency=self._text(by_field, "currency", col_idx),
                )
            )
        return snapshots

    def parse_html_annual(self, content: str, symbol: str) -> list[FundamentalSnapshot]:
        """Parse SmartLab ANNUAL fundamentals HTML into one snapshot per fiscal year.

        Mirrors :meth:`parse_html` but locates columns with the bare-year regex
        (``_YEAR_RE``) instead of the quarter regex, keys each ``period`` on the
        4-digit year token, and resolves ``as_of`` via the annual +120d helper.
        The trailing LTM column is excluded (it fails ``^\\d{4}$``). Banks lack
        ``revenue``/``ev_ebitda`` rows → those fields stay ``None`` (Pitfall 5,
        guaranteed by the shared ``.get()`` extractors).
        """
        doc = lxml_html.fromstring(content)
        tables: Any = doc.xpath(_FINANCIALS_XPATH)
        if not tables:
            return []
        rows: Any = tables[0].xpath(".//tr")

        # Locate the year-header row defensively by year-regex (NOT a fixed index):
        # the leading "Показатель" label column would break index-based parsing.
        year_cols: dict[int, str] = {}
        for row in rows:
            cells = [self._cell_text(c) for c in row.xpath("./th|./td")]
            candidate = {i: lbl for i, lbl in enumerate(cells) if _YEAR_RE.match(lbl)}
            if candidate:
                year_cols = candidate
                break
        if not year_cols:
            return []

        by_field: dict[str, list[str]] = {}
        for row in rows:
            field = row.get("field")
            if not field:
                continue
            by_field[field] = [self._cell_text(c) for c in row.xpath("./th|./td")]

        date_cells = by_field.get("date", [])

        snapshots: list[FundamentalSnapshot] = []
        for col_idx, period in year_cols.items():
            as_of = self._resolve_annual_as_of(date_cells, col_idx, symbol, period)
            snapshots.append(
                FundamentalSnapshot(
                    symbol=symbol,
                    as_of=as_of,
                    pe_ratio=self._number(by_field, "p_e", col_idx),
                    ev_ebitda=self._number(by_field, "ev_ebitda", col_idx),
                    revenue_ttm=self._billions(by_field, "revenue", col_idx),
                    net_margin=self._percent(by_field, "net_margin", col_idx),
                    roe=self._percent(by_field, "roe", col_idx),
                    eps_ttm=self._number(by_field, "eps", col_idx),
                    dividend_yield=self._yield(by_field, "div_yield", col_idx),
                    market_cap=self._billions(by_field, "market_cap", col_idx),
                    currency=self._text(by_field, "currency", col_idx),
                )
            )
        return snapshots

    @staticmethod
    def _cell_text(cell: Any) -> str:
        """First non-empty line of a cell's text content, stripped."""
        text = str(cell.text_content())
        return text.strip().split("\n")[0].strip()

    def _resolve_as_of(
        self, date_cells: list[str], col_idx: int, symbol: str, period: str
    ) -> datetime:
        """Resolve a column's ``as_of``: the explicit date cell or the +75d lag fallback.

        NEVER returns a fiscal-quarter-end date directly (look-ahead trap, H-02).
        """
        raw = date_cells[col_idx] if col_idx < len(date_cells) else ""
        if raw:
            try:
                return datetime.strptime(raw, "%d.%m.%Y").replace(tzinfo=UTC)
            except ValueError:
                _log.warning("smartlab_bad_date_cell", symbol=symbol, period=period, raw=raw)
        # Empty / unparseable date cell → conservative publication-lag fallback.
        _log.info("smartlab_lag_approximated", symbol=symbol, period=period)
        effective = get_effective_disclosure_date(symbol, period)
        return datetime(effective.year, effective.month, effective.day, tzinfo=UTC)

    def _resolve_annual_as_of(
        self, date_cells: list[str], col_idx: int, symbol: str, period: str
    ) -> datetime:
        """Resolve an annual column's ``as_of``: the explicit date cell or +120d fallback.

        Mirrors :meth:`_resolve_as_of` but uses the annual +120d helper. NEVER
        returns a bare fiscal-year-end date (look-ahead trap, BACKFILL-Y-02).
        """
        raw = date_cells[col_idx] if col_idx < len(date_cells) else ""
        if raw:
            try:
                return datetime.strptime(raw, "%d.%m.%Y").replace(tzinfo=UTC)
            except ValueError:
                _log.warning("smartlab_bad_date_cell", symbol=symbol, period=period, raw=raw)
        # Empty / unparseable date cell → conservative annual publication-lag fallback.
        _log.info("smartlab_lag_approximated", symbol=symbol, period=period, route="annual")
        effective = get_effective_annual_disclosure_date(symbol, period)
        return datetime(effective.year, effective.month, effective.day, tzinfo=UTC)

    @staticmethod
    def _raw(by_field: dict[str, list[str]], field: str, col_idx: int) -> str:
        cells = by_field.get(field)
        if cells is None or col_idx >= len(cells):
            return ""
        return cells[col_idx]

    def _decimal(self, raw: str) -> Decimal | None:
        """Parse a SmartLab numeric string to Decimal, or None if blank/unparseable.

        Strips thousands separators (U+00A0/U+2009/space) and converts comma decimals.
        Skip-not-fabricate on a bad numeric (V5 input validation).
        """
        if not raw:
            return None
        cleaned = raw
        for sep in _THOUSANDS_SEPARATORS:
            cleaned = cleaned.replace(sep, "")
        cleaned = cleaned.replace(",", ".")
        try:
            return Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return None

    def _number(self, by_field: dict[str, list[str]], field: str, col_idx: int) -> float | None:
        value = self._decimal(self._raw(by_field, field, col_idx))
        return float(value) if value is not None else None

    def _billions(self, by_field: dict[str, list[str]], field: str, col_idx: int) -> float | None:
        """A billions-of-rubles ("mlrd rub") value scaled to raw RUB (x1e9)."""
        value = self._decimal(self._raw(by_field, field, col_idx))
        return float(value * _BILLION) if value is not None else None

    def _percent(self, by_field: dict[str, list[str]], field: str, col_idx: int) -> float | None:
        """A "21,4%" cell → 0.214 fraction."""
        value = self._decimal(self._raw(by_field, field, col_idx).replace("%", "").strip())
        return float(value / Decimal(100)) if value is not None else None

    def _yield(self, by_field: dict[str, list[str]], field: str, col_idx: int) -> float | None:
        """Dividend yield "%" → fraction; a bare 0.0% reads as absent (no fabricated 0)."""
        value = self._decimal(self._raw(by_field, field, col_idx).replace("%", "").strip())
        if value is None or value == _ZERO_YIELD:
            return None
        return float(value / Decimal(100))

    @staticmethod
    def _text(by_field: dict[str, list[str]], field: str, col_idx: int) -> str | None:
        cells = by_field.get(field)
        if cells is None or col_idx >= len(cells) or not cells[col_idx]:
            return None
        return cells[col_idx]

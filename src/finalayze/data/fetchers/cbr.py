"""CBR (Central Bank of Russia) API fetcher (Layer 2).

Fetches FX rates (REST XML) and key rate (SOAP XML) from cbr.ru.
Sync only — do NOT call from async code without asyncio.to_thread().
Uses httpx + lxml. No third-party CBR libraries.
Does NOT extend BaseFetcher — CBR API structure is fundamentally different.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from lxml import etree
from lxml import html as lxml_html

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import (
    RATE_REGIME_EASING,
    RATE_REGIME_HIGH_RATE,
    FXRate,
    KeyRateRecord,
)

if TYPE_CHECKING:
    from finalayze.data.rate_limiter import RateLimiter

_log = structlog.get_logger()

_FX_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
_KEY_RATE_URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"

_CURRENCY_CODES: dict[str, str] = {"USD": "R01235", "EUR": "R01239"}
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0
_ONE_DAY = timedelta(days=1)

_KEY_RATE_SOAP = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"
               xmlns:web="http://web.cbr.ru/">
  <soap:Body>
    <web:KeyRateXML>
      <web:fromDate>{from_date}</web:fromDate>
      <web:ToDate>{to_date}</web:ToDate>
    </web:KeyRateXML>
  </soap:Body>
</soap:Envelope>"""

_KEY_RATE_PERCENT_DIVISOR = Decimal(100)  # CBR returns percentage points
_ZCYC_URL = "https://www.cbr.ru/hd_base/zcyc_params/"
_ZCYC_MATURITIES = ("0.25", "0.50", "0.75", "1", "2", "3", "5", "7", "10", "15", "20", "30")
_INDEXATION_URL = "https://www.cbr.ru/hd_base/ostat_depo_new/"
# CBR inflation dynamics page (YoY CPI). RESEARCH A2: the live HTML table layout is
# UNCONFIRMED (the page may be JS-rendered/PDF); _parse_inflation_html's xpath is a
# best-known guess and may need a one-off MANUAL-verify pass — see the single
# clearly-commented xpath block in _parse_inflation_html and 59-VALIDATION.md
# "Manual-Only Verifications" (CPI-01).
_INFLATION_URL = "https://www.cbr.ru/eng/analytics/dkp/dinamic/"


class CBRFetcher:
    """CBR XML API client. Sync. Does NOT extend BaseFetcher."""

    def __init__(self, rate_limiter: RateLimiter | None = None) -> None:
        self._client = httpx.Client(timeout=30.0)
        self._rate_limiter = rate_limiter

    def close(self) -> None:
        """Close underlying httpx.Client."""
        self._client.close()

    def __enter__(self) -> CBRFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch_fx_rates(
        self,
        currency: str,
        start: datetime,
        end: datetime,
    ) -> list[FXRate]:
        """Fetch daily official FX rates. end is exclusive (CBR API is inclusive)."""
        code = _CURRENCY_CODES.get(currency)
        if code is None:
            msg = f"Unsupported currency: {currency}. Supported: {list(_CURRENCY_CODES)}"
            raise DataFetchError(msg)

        pair = f"{currency}RUB"
        # CBR API uses inclusive date ranges; our end is exclusive → subtract 1 day
        cbr_end = end - _ONE_DAY
        params = {
            "date_req1": start.strftime("%d/%m/%Y"),
            "date_req2": cbr_end.strftime("%d/%m/%Y"),
            "VAL_NM_RQ": code,
        }
        content = self._request("GET", _FX_URL, params=params)
        return self._parse_fx_xml(content, pair)

    def fetch_yield_curve(self, as_of: date) -> dict[str, Decimal] | None:
        """Fetch zero-coupon yield curve from CBR for *as_of* date.

        Returns dict of maturity -> yield (percentage points), or None if no data
        (weekends, holidays, HTTP errors).
        """
        params = {
            "DateReq": as_of.strftime("%d.%m.%Y"),
        }
        try:
            content = self._request("GET", _ZCYC_URL, params=params)
        except DataFetchError:
            _log.warning("cbr_yield_curve_fetch_failed", as_of=str(as_of))
            return None
        return self._parse_zcyc_html(content.decode("utf-8", errors="replace"))

    @staticmethod
    def _parse_zcyc_html(content: str) -> dict[str, Decimal] | None:
        """Parse CBR ZCYC HTML table into maturity -> yield dict.

        Returns None if no data rows found (weekends/holidays).
        """
        doc = lxml_html.fromstring(content)
        tables: Any = doc.xpath("//table[contains(@class, 'data')]")
        if not tables:
            return None

        table: Any = tables[0]
        rows: Any = table.xpath(".//tr")
        if len(rows) < 2:  # noqa: PLR2004 -- header + at least 1 data row
            return None

        # Header row: extract maturity labels
        headers: list[str] = [th.text_content().strip() for th in rows[0].xpath("th|td")]
        # Data row: first data row after header
        data_cells: list[str] = [td.text_content().strip() for td in rows[1].xpath("td")]

        if len(data_cells) < 2:  # noqa: PLR2004 -- date + at least 1 value
            return None

        # Skip first cell (date), map remaining to headers
        result: dict[str, Decimal] = {}
        for i, val_str in enumerate(data_cells[1:], start=1):
            if i < len(headers) and val_str:
                try:
                    maturity = headers[i].strip()
                    result[maturity] = Decimal(val_str.replace(",", "."))
                except (ValueError, ArithmeticError):
                    continue

        return result or None

    def fetch_ofzin_indexation_coefficient(self, as_of: date) -> Decimal | None:
        """Fetch OFZ-IN daily indexation coefficient from CBR for *as_of* date.

        The coefficient represents cumulative CPI adjustment since issuance
        (e.g. 1.0523 = 5.23% cumulative inflation). Used for adjusting
        OFZ-IN face value: adjusted_nominal = original_nominal * coefficient.

        Returns None if no data available (weekends, holidays, HTTP errors).
        """
        params = {"DateReq": as_of.strftime("%d.%m.%Y")}
        try:
            content = self._request("GET", _INDEXATION_URL, params=params)
        except DataFetchError:
            _log.warning("cbr_indexation_fetch_failed", as_of=str(as_of))
            return None
        return self._parse_indexation_response(content.decode("utf-8", errors="replace"))

    @staticmethod
    def _parse_indexation_response(content: str) -> Decimal | None:
        """Parse CBR OFZ-IN indexation HTML response.

        Returns the indexation coefficient as Decimal, or None if no data.
        """
        doc = lxml_html.fromstring(content)
        tables: Any = doc.xpath("//table[contains(@class, 'data')]")
        if not tables:
            return None

        table: Any = tables[0]
        rows: Any = table.xpath(".//tr")
        if len(rows) < 2:  # noqa: PLR2004 -- header + at least 1 data row
            return None

        # Data row: second row (first is header)
        data_cells: list[str] = [td.text_content().strip() for td in rows[1].xpath("td")]
        if len(data_cells) < 2:  # noqa: PLR2004 -- date + coefficient
            return None

        coeff_str = data_cells[1].replace(",", ".")
        try:
            return Decimal(coeff_str)
        except (ValueError, ArithmeticError):
            return None

    def fetch_cpi_yoy(self, as_of: date) -> dict[str, Decimal] | None:
        """Fetch YoY CPI from CBR for *as_of*, mirroring fetch_yield_curve.

        Returns a dict of covered-month (``YYYY-MM``) -> YoY value in **percentage
        points** (the ``_CPI_DATA`` unit), or ``None`` if the page is unreachable /
        unparseable (no fabrication — ``_CPI_DATA`` stays the seeded fallback).

        NO unit conversion here: ``get_cpi_yoy_fraction`` already does the ``/ 100``
        fraction conversion at its read boundary; do NOT double-convert.

        Sync only — call from async via ``asyncio.to_thread`` (CBRFetcher is sync).
        """
        params = {"DateReq": as_of.strftime("%d.%m.%Y")}
        try:
            content = self._request("GET", _INFLATION_URL, params=params)
        except DataFetchError:
            _log.warning("cbr_cpi_fetch_failed", as_of=str(as_of))
            return None
        return self._parse_inflation_html(content.decode("utf-8", errors="replace"))

    @staticmethod
    def _parse_inflation_html(content: str) -> dict[str, Decimal] | None:
        """Parse the CBR inflation HTML table into covered-month -> YoY pct points.

        Mirrors ``_parse_zcyc_html``: ``lxml_html.fromstring`` -> the ``data`` table ->
        per-row (covered_month, YoY) extraction -> ``Decimal(val.replace(",", "."))``.
        Returns ``None`` if no data rows are found (mirrors the "no rows -> None" rule).

        RESEARCH A2: the live table layout is UNCONFIRMED — the row/cell xpath below is
        kept as a single, clearly-commented, easily-adjustable block (the MANUAL-verify
        seam). If the live HTML differs, adjust THIS block and re-run Plan 02 tests.
        """
        doc = lxml_html.fromstring(content)
        tables: Any = doc.xpath("//table[contains(@class, 'data')]")
        if not tables:
            return None

        table: Any = tables[0]
        rows: Any = table.xpath(".//tr")
        if len(rows) < 2:  # noqa: PLR2004 -- header + at least 1 data row
            return None

        # ── MANUAL-verify xpath seam (A2): covered_month in col 0, YoY % in col 1 ──
        result: dict[str, Decimal] = {}
        for row in rows[1:]:  # skip header
            cells: list[str] = [c.text_content().strip() for c in row.xpath("td")]
            if len(cells) < 2:  # noqa: PLR2004 -- need month + value
                continue
            month_key, val_str = cells[0], cells[1]
            if not month_key or not val_str:
                continue
            try:
                result[month_key] = Decimal(val_str.replace(",", "."))
            except (ValueError, ArithmeticError):
                continue
        # ──────────────────────────────────────────────────────────────────────────

        return result or None

    def fetch_key_rate(self, start: datetime, end: datetime) -> list[KeyRateRecord]:
        """Fetch CBR key rate history via SOAP. Rate normalized to decimal fraction."""
        body = _KEY_RATE_SOAP.format(
            from_date=start.strftime("%Y-%m-%dT00:00:00"),
            to_date=end.strftime("%Y-%m-%dT00:00:00"),
        )
        content = self._request(
            "POST",
            _KEY_RATE_URL,
            content=body.encode("utf-8"),
            headers={
                "Content-Type": "text/xml; charset=utf-8",
                "SOAPAction": "http://web.cbr.ru/KeyRateXML",
            },
        )
        return self._parse_key_rate_xml(content)

    def _request(self, method: str, url: str, **kwargs: Any) -> bytes:
        """HTTP request with rate limiting, retry, and error wrapping."""
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
                _log.warning("cbr_timeout", attempt=attempt + 1, url=url)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                _log.warning("cbr_http_error", status=exc.response.status_code, attempt=attempt + 1)
            except httpx.RequestError as exc:
                last_exc = exc
                _log.warning("cbr_network_error", attempt=attempt + 1, url=url)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE * (2**attempt))

        cause = "network_error"
        if isinstance(last_exc, httpx.TimeoutException):
            cause = "timeout"
        elif isinstance(last_exc, httpx.HTTPStatusError):
            cause = "http_error"
        msg = f"CBR {cause}: {last_exc}"
        raise DataFetchError(msg) from last_exc

    @staticmethod
    def _parse_fx_xml(content: bytes, pair: str) -> list[FXRate]:
        """Parse CBR XML_dynamic.asp response."""
        try:
            tree = etree.fromstring(content)
        except etree.XMLSyntaxError as exc:
            msg = f"CBR parse_error: {exc}"
            raise DataFetchError(msg) from exc

        rates: list[FXRate] = []
        for record in tree.findall(".//Record"):
            date_str = record.get("Date", "")
            nominal_el = record.find("Nominal")
            value_el = record.find("Value")
            if not date_str or nominal_el is None or value_el is None:
                continue
            if nominal_el.text is None or value_el.text is None:
                continue
            dt = datetime.strptime(date_str, "%d.%m.%Y").replace(tzinfo=UTC)
            nominal = Decimal(nominal_el.text.replace(",", "."))
            value = Decimal(value_el.text.replace(",", "."))
            rate = value / nominal if nominal else value
            rates.append(FXRate(timestamp=dt, pair=pair, rate=rate))
        return sorted(rates, key=lambda r: r.timestamp)

    @staticmethod
    def _parse_key_rate_xml(content: bytes) -> list[KeyRateRecord]:
        """Parse CBR SOAP KeyRateXML response. Rate divided by 100 (% → decimal)."""
        try:
            tree = etree.fromstring(content)
        except etree.XMLSyntaxError as exc:
            msg = f"CBR parse_error: {exc}"
            raise DataFetchError(msg) from exc

        records: list[KeyRateRecord] = []
        for kr in tree.iter():
            if kr.tag.endswith("}KR") or kr.tag == "KR":
                dt_el = rate_el = None
                for child in kr:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if tag == "DT":
                        dt_el = child
                    elif tag == "Rate":
                        rate_el = child
                if dt_el is not None and dt_el.text and rate_el is not None and rate_el.text:
                    dt = datetime.fromisoformat(dt_el.text).replace(tzinfo=UTC)
                    # CBR returns percentage points (16.00 = 16%). Normalize to decimal fraction.
                    rate = Decimal(rate_el.text) / _KEY_RATE_PERCENT_DIVISOR
                    records.append(KeyRateRecord(timestamp=dt, rate=rate))
        return sorted(records, key=lambda r: r.timestamp)


# ── CBR Meeting Calendar (static data for backtesting) ──────────────────────


@dataclass(frozen=True)
class CBRMeeting:
    """CBR Board of Directors rate meeting."""

    date: date
    meeting_type: str  # "core", "interim", or "emergency"
    decision: str | None = None  # "cut", "hold", "hike", None (future/unknown)
    rate_after: Decimal | None = None  # key rate (%) after decision


# Historical + scheduled meetings for backtesting (2022-2026).
# Sources: CBR press releases, official schedule.
# Future meetings (decision=None) use the published CBR schedule.
CBR_MEETINGS: tuple[CBRMeeting, ...] = (
    # 2022
    CBRMeeting(date(2022, 2, 28), "emergency", "hike", Decimal("20.00")),
    CBRMeeting(date(2022, 4, 8), "interim", "cut", Decimal("17.00")),
    CBRMeeting(date(2022, 4, 29), "core", "cut", Decimal("14.00")),
    CBRMeeting(date(2022, 6, 10), "interim", "cut", Decimal("9.50")),
    CBRMeeting(date(2022, 7, 22), "core", "cut", Decimal("8.00")),
    CBRMeeting(date(2022, 9, 16), "interim", "cut", Decimal("7.50")),  # 8.00->7.50 is a cut
    CBRMeeting(date(2022, 10, 28), "core", "hold", Decimal("7.50")),
    CBRMeeting(date(2022, 12, 16), "interim", "hold", Decimal("7.50")),
    # 2023
    CBRMeeting(date(2023, 2, 10), "core", "hold", Decimal("7.50")),
    CBRMeeting(date(2023, 3, 17), "interim", "hold", Decimal("7.50")),
    CBRMeeting(date(2023, 4, 28), "core", "hold", Decimal("7.50")),
    CBRMeeting(date(2023, 6, 9), "interim", "hold", Decimal("7.50")),
    CBRMeeting(date(2023, 7, 21), "core", "hike", Decimal("8.50")),
    CBRMeeting(date(2023, 8, 15), "emergency", "hike", Decimal("12.00")),
    CBRMeeting(date(2023, 9, 15), "interim", "hike", Decimal("13.00")),
    CBRMeeting(date(2023, 10, 27), "core", "hike", Decimal("15.00")),
    CBRMeeting(date(2023, 12, 15), "interim", "hike", Decimal("16.00")),
    # 2024
    CBRMeeting(date(2024, 2, 16), "core", "hold", Decimal("16.00")),
    CBRMeeting(date(2024, 3, 22), "interim", "hold", Decimal("16.00")),
    CBRMeeting(date(2024, 4, 26), "core", "hold", Decimal("16.00")),
    CBRMeeting(date(2024, 6, 7), "interim", "hold", Decimal("16.00")),
    CBRMeeting(date(2024, 7, 26), "core", "hike", Decimal("18.00")),
    CBRMeeting(date(2024, 9, 13), "interim", "hike", Decimal("19.00")),
    CBRMeeting(date(2024, 10, 25), "core", "hike", Decimal("21.00")),
    CBRMeeting(date(2024, 12, 20), "interim", "hold", Decimal("21.00")),
    # 2025 — realized easing path verified against the official cbr.ru archive (R-C).
    CBRMeeting(date(2025, 2, 14), "core", "hold", Decimal("21.00")),
    CBRMeeting(date(2025, 3, 21), "interim", "hold", Decimal("21.00")),
    CBRMeeting(date(2025, 4, 25), "core", "hold", Decimal("21.00")),
    CBRMeeting(date(2025, 6, 6), "interim", "cut", Decimal("20.00")),  # FIRST cut (not 07-25)
    CBRMeeting(date(2025, 7, 25), "core", "cut", Decimal("18.00")),
    CBRMeeting(date(2025, 9, 12), "interim", "cut", Decimal("17.00")),
    CBRMeeting(date(2025, 10, 24), "core", "cut", Decimal("16.50")),
    CBRMeeting(date(2025, 12, 19), "interim", "cut", Decimal("16.00")),
    # 2026
    CBRMeeting(date(2026, 2, 13), "core", "cut", Decimal("15.50")),
    CBRMeeting(date(2026, 3, 20), "interim", "cut", Decimal("15.00")),  # filled
    CBRMeeting(date(2026, 4, 24), "core", "cut", Decimal("14.50")),  # filled — terminal 14.50%
    CBRMeeting(date(2026, 6, 19), "interim", None, None),  # future
    CBRMeeting(date(2026, 7, 24), "core", None, None),
    CBRMeeting(date(2026, 9, 11), "interim", None, None),
    CBRMeeting(date(2026, 10, 23), "core", None, None),
    CBRMeeting(date(2026, 12, 18), "interim", None, None),
)


# ── CPI Publication Dates (Rosstat) ─────────────────────────────────────────
# Rosstat publishes monthly CPI approximately 2 weeks after month end.
# Format: month_covered (YYYY-MM) -> publication_date.
# Only dates BEFORE publication_date can use that month's CPI in backtest
# (prevents look-ahead bias).

CPI_PUBLICATION_DATES: dict[str, date] = {
    # 2024
    "2024-01": date(2024, 2, 9),
    "2024-02": date(2024, 3, 13),
    "2024-03": date(2024, 4, 12),
    "2024-04": date(2024, 5, 17),
    "2024-05": date(2024, 6, 14),
    "2024-06": date(2024, 7, 12),
    "2024-07": date(2024, 8, 9),
    "2024-08": date(2024, 9, 13),
    "2024-09": date(2024, 10, 11),
    "2024-10": date(2024, 11, 13),
    "2024-11": date(2024, 12, 13),
    "2024-12": date(2025, 1, 15),
    # 2025
    "2025-01": date(2025, 2, 12),
    "2025-02": date(2025, 3, 12),
    "2025-03": date(2025, 4, 11),
    "2025-04": date(2025, 5, 16),
    "2025-05": date(2025, 6, 13),
    "2025-06": date(2025, 7, 11),
    "2025-07": date(2025, 8, 8),
    "2025-08": date(2025, 9, 12),
    "2025-09": date(2025, 10, 10),
    "2025-10": date(2025, 11, 14),
    "2025-11": date(2025, 12, 12),
    "2025-12": date(2026, 1, 16),
    # 2026 (Rosstat publishes ~mid of the following month)
    "2026-01": date(2026, 2, 13),
    "2026-02": date(2026, 3, 13),
    "2026-03": date(2026, 4, 10),
}


# ── Helper functions (no look-ahead, safe for backtesting) ──────────────────


def get_last_cbr_decision(as_of: date) -> CBRMeeting | None:
    """Return the most recent CBR meeting with a decision, on or before *as_of*.

    For backtesting: never returns meetings after *as_of* (no look-ahead).
    Meetings with ``decision is None`` (future/unknown) are skipped.
    """
    past = [m for m in CBR_MEETINGS if m.date <= as_of and m.decision is not None]
    return past[-1] if past else None


def get_next_cbr_meeting(as_of: date) -> CBRMeeting | None:
    """Return the next CBR meeting strictly after *as_of*.

    Returns both decided and future (undecided) meetings.
    """
    future = [m for m in CBR_MEETINGS if m.date > as_of]
    return future[0] if future else None


def days_to_next_cbr(as_of: date) -> int | None:
    """Days until the next CBR meeting. ``None`` if no future meetings in calendar."""
    nxt = get_next_cbr_meeting(as_of)
    return (nxt.date - as_of).days if nxt else None


def get_latest_published_cpi_month(as_of: date) -> str | None:
    """Return the latest CPI month whose publication date is on or before *as_of*.

    For backtesting: ensures no CPI look-ahead. Returns ``YYYY-MM`` string.
    """
    published = [month for month, pub_date in CPI_PUBLICATION_DATES.items() if pub_date <= as_of]
    return max(published) if published else None


# ── Macro Context (backtest-safe aggregation layer) ──────────────────────────


@dataclass(frozen=True)
class MacroSnapshot:
    """Point-in-time macro context for bond strategy decisions.

    All values use **percentage-point** convention (e.g. ``21.00`` means 21%).
    ``None`` indicates data unavailable for the given date.
    """

    key_rate: Decimal | None = None
    ruonia_7d_avg: Decimal | None = None
    cpi_yoy: Decimal | None = None
    last_cbr_decision: str | None = None  # "cut", "hold", "hike"
    yield_curve: dict[str, Decimal] | None = None  # maturity -> yield (%)
    breakeven_inflation: Decimal | None = None  # OFZ-IN vs OFZ-PD spread at 5Y
    usdrub: Decimal | None = None  # USD/RUB exchange rate
    ofzin_indexation_coefficient: Decimal | None = None  # cumulative CPI adjustment


# RUONIA proxy: key_rate minus 50bps (RUONIA typically tracks 30-80bps below).
_RUONIA_PROXY_OFFSET = Decimal("0.50")

# Rosstat YoY CPI data (approximate monthly values, percentage points).
# Coverage: 2022-01 through 2025-06.
_CPI_DATA: dict[str, Decimal] = {
    # 2022
    "2022-01": Decimal("8.7"),
    "2022-02": Decimal("9.2"),
    "2022-03": Decimal("16.7"),
    "2022-04": Decimal("17.8"),
    "2022-05": Decimal("17.1"),
    "2022-06": Decimal("15.9"),
    "2022-07": Decimal("15.1"),
    "2022-08": Decimal("14.3"),
    "2022-09": Decimal("13.7"),
    "2022-10": Decimal("12.6"),
    "2022-11": Decimal("12.0"),
    "2022-12": Decimal("11.9"),
    # 2023
    "2023-01": Decimal("11.8"),
    "2023-02": Decimal("11.0"),
    "2023-03": Decimal("3.5"),
    "2023-04": Decimal("2.3"),
    "2023-05": Decimal("2.5"),
    "2023-06": Decimal("3.2"),
    "2023-07": Decimal("4.3"),
    "2023-08": Decimal("5.1"),
    "2023-09": Decimal("6.0"),
    "2023-10": Decimal("6.7"),
    "2023-11": Decimal("7.5"),
    "2023-12": Decimal("7.4"),
    # 2024
    "2024-01": Decimal("7.4"),
    "2024-02": Decimal("7.7"),
    "2024-03": Decimal("7.7"),
    "2024-04": Decimal("7.8"),
    "2024-05": Decimal("8.3"),
    "2024-06": Decimal("8.6"),
    "2024-07": Decimal("9.1"),
    "2024-08": Decimal("9.1"),
    "2024-09": Decimal("8.6"),
    "2024-10": Decimal("8.5"),
    "2024-11": Decimal("8.9"),
    "2024-12": Decimal("9.5"),
    # 2025
    "2025-01": Decimal("10.0"),
    "2025-02": Decimal("10.0"),
    "2025-03": Decimal("9.9"),
    "2025-04": Decimal("9.7"),
    "2025-05": Decimal("9.4"),
    "2025-06": Decimal("9.1"),
    # 2025-H2 → 2026-Q1 extension (sourced rateinflation.com / CBR commentary,
    # cross-checked TradingEconomics; trailing-12m YoY %). Refreshed 2026-05-30.
    # NOTE: this dict is the single source of truth for CPI — ml/features/macro.py
    # reads it via get_cpi_yoy_fraction(). Do not reintroduce a parallel table.
    # TODO(data-epic): replace this static table with a live Rosstat/CBR feed.
    "2025-07": Decimal("8.8"),
    "2025-08": Decimal("8.1"),
    "2025-09": Decimal("8.0"),
    "2025-10": Decimal("7.8"),
    "2025-11": Decimal("6.6"),
    "2025-12": Decimal("5.6"),
    # 2026
    "2026-01": Decimal("6.0"),
    "2026-02": Decimal("5.1"),
    "2026-03": Decimal("5.9"),
}


# Rosstat publishes monthly CPI ~2 weeks after month end; allow that lag before
# considering the static table "stale".
_CPI_PUBLICATION_LAG_MONTHS = 2


def get_cpi_yoy_fraction(year: int, month: int) -> float | None:
    """Return trailing-12m CPI for *year*/*month* as a decimal fraction.

    Single source of truth for CPI across the codebase. ``_CPI_DATA`` stores
    percentages (``9.1`` = 9.1%); this returns the fraction (``0.091``) expected
    by the ML real-rate feature. Returns ``None`` if the month is not covered.
    """
    value = _CPI_DATA.get(f"{year:04d}-{month:02d}")
    return float(value) / 100.0 if value is not None else None


def latest_cpi_month() -> str:
    """Return the most recent month covered by ``_CPI_DATA`` as ``YYYY-MM``."""
    return max(_CPI_DATA)


def cpi_data_staleness_months(as_of: date) -> int:
    """Months by which the static CPI table lags *as_of*, net of publication lag.

    0 means the table is current enough (the latest covered month is within the
    normal Rosstat publication lag of *as_of*). A positive value is the number of
    months of genuinely missing data — callers should log/alert on it so the
    table never silently rots again (see ml/features/macro.py).
    """
    latest = latest_cpi_month()
    latest_year, latest_month = int(latest[:4]), int(latest[5:7])
    diff = (as_of.year - latest_year) * 12 + (as_of.month - latest_month)
    return max(0, diff - _CPI_PUBLICATION_LAG_MONTHS)


def _effective_cpi_publication_date(covered_month: str) -> date:
    """Effective publication date for a fetched CPI month with no recorded entry.

    Mirrors the lag ``cpi_data_staleness_months`` already uses: month-end of
    *covered_month* plus ``_CPI_PUBLICATION_LAG_MONTHS`` months. Used as the
    look-ahead boundary for newly-fetched months absent from ``CPI_PUBLICATION_DATES``.
    """
    year, month = int(covered_month[:4]), int(covered_month[5:7])
    # Publication lands in (covered_month + lag); use day 1 of that month as the
    # effective availability date (conservative within the month).
    total = (year * 12 + (month - 1)) + _CPI_PUBLICATION_LAG_MONTHS
    pub_year, pub_month = divmod(total, 12)
    return date(pub_year, pub_month + 1, 1)


def refresh_cpi_data(fetched: dict[str, Decimal], as_of: date) -> int:
    """Overlay publication-eligible fetched CPI months into the single ``_CPI_DATA`` source.

    IN-MEMORY overlay only (D-04 backtest-first: no persistence / live-loop scheduler
    this phase). ``_CPI_DATA`` stays the seeded FALLBACK for any month not overlaid —
    no fabrication. Feeding the single source means ``get_cpi_yoy_fraction`` /
    ``latest_cpi_month`` / ``cpi_data_staleness_months`` reflect live data with NO
    change to their bodies.

    Look-ahead safety (T-59-04): a fetched month is overlaid ONLY when its publication
    date is on or before *as_of*. If the month already has a ``CPI_PUBLICATION_DATES``
    entry, that date is used; otherwise an effective publication date is derived as
    month-end + ``_CPI_PUBLICATION_LAG_MONTHS`` and recorded into ``CPI_PUBLICATION_DATES``
    so the existing ``get_latest_published_cpi_month`` machinery stays consistent.

    Returns the count of months actually overlaid.
    """
    overlaid = 0
    for covered_month, yoy_pct in fetched.items():
        recorded = CPI_PUBLICATION_DATES.get(covered_month)
        if recorded is not None:
            pub_date = recorded
        else:
            pub_date = _effective_cpi_publication_date(covered_month)
        if pub_date > as_of:
            continue  # not yet publication-eligible — stays unavailable (look-ahead safe)
        if recorded is None:
            CPI_PUBLICATION_DATES[covered_month] = pub_date
        _CPI_DATA[covered_month] = yoy_pct
        overlaid += 1
    return overlaid


def refresh_cpi_from_cbr(fetcher: CBRFetcher, as_of: date) -> int:
    """Fetch live CPI via *fetcher* and overlay publication-eligible months into ``_CPI_DATA``.

    Thin convenience seam a future live-loop (deferred per D-04) would call. Returns the
    overlaid count, or 0 when the fetch returns ``None`` (the seeded ``_CPI_DATA`` is left
    intact — no fabrication on a fetch miss). No scheduler is wired this phase.
    """
    fetched = fetcher.fetch_cpi_yoy(as_of)
    if fetched is None:
        return 0
    return refresh_cpi_data(fetched, as_of)


_YIELD_CURVE_SLOPE_BPS: dict[str, float] = {
    "2022-03": -250.0,
    "2022-04": -180.0,
    "2022-06": -50.0,
    "2022-09": 20.0,
    "2022-12": 80.0,
    "2023-03": 120.0,
    "2023-06": 100.0,
    "2023-09": -30.0,
    "2023-12": -80.0,
    "2024-03": -120.0,
    "2024-06": -150.0,
    "2024-09": -200.0,
    "2024-12": -180.0,
    "2025-03": -150.0,
    "2025-06": -100.0,
    "2025-09": -20.0,
    "2025-12": 50.0,
}


def get_recent_cbr_decisions(as_of: date, count: int = 3) -> list[str]:
    """Return the last *count* CBR decisions (most recent first), on or before *as_of*.

    Meetings with ``decision is None`` are skipped.
    """
    past = [m for m in CBR_MEETINGS if m.date <= as_of and m.decision is not None]
    return [m.decision for m in reversed(past[-count:]) if m.decision is not None]


def is_cutting_cycle(as_of: date) -> bool:
    """Return True if the last 2 CBR decisions are both "cut"."""
    decisions = get_recent_cbr_decisions(as_of, count=2)
    return len(decisions) >= 2 and all(d == "cut" for d in decisions)  # noqa: PLR2004


def rate_regime_as_of(as_of: date) -> str:
    """Look-ahead-safe rate regime for the allocator tilt (Phase 76).

    Returns :data:`~finalayze.core.schemas.RATE_REGIME_EASING` once the most recent CBR
    decision on/before ``as_of`` is a ``"cut"`` (rates falling -> tilt toward OFZ
    duration + equity), else :data:`~finalayze.core.schemas.RATE_REGIME_HIGH_RATE` (tilt
    toward the deposit anchor). Reads ONLY meetings ``<= as_of`` via the existing meeting
    calendar (no look-ahead, no new fetcher), so the 2025-06-06 first cut flips the regime
    exactly at the gate's ``REGIME_SPLIT_BOUNDARY`` while 2025-06-05 stays ``high_rate``.
    Alternative ``is_cutting_cycle`` (two consecutive cuts) lags this boundary -- a future
    tuning lever, not used for the binding tilt.
    """
    last = get_last_cbr_decision(as_of)
    if last is not None and last.decision == "cut":
        return RATE_REGIME_EASING
    return RATE_REGIME_HIGH_RATE


def get_yield_slope_bps(as_of: date) -> float:
    """Return the yield curve slope (10Y-2Y) in bps for the month of *as_of*.

    Uses the latest key in ``_YIELD_CURVE_SLOPE_BPS`` that is <= the as_of month.
    Returns 0.0 if no data available.
    """
    target = as_of.strftime("%Y-%m")
    candidates = [k for k in _YIELD_CURVE_SLOPE_BPS if k <= target]
    if not candidates:
        return 0.0
    return _YIELD_CURVE_SLOPE_BPS[max(candidates)]


class MacroContextProvider:
    """Provides point-in-time macro data for bond strategy backtests.

    Uses static CBR meeting calendar and CPI data.  All lookups respect
    publication dates to avoid look-ahead bias.

    RUONIA is approximated as ``key_rate - 50bps`` (RUONIA typically tracks
    30-80bps below key rate). This is a backtest-only proxy.
    """

    def get_snapshot(self, as_of: date) -> MacroSnapshot:
        """Return macro data available as of *as_of* (no look-ahead).

        Args:
            as_of: The date to query. Only information published on or
                before this date is included.

        Returns:
            MacroSnapshot with available fields populated.
        """
        # key_rate from most recent CBR decision
        last_meeting = get_last_cbr_decision(as_of)
        key_rate = last_meeting.rate_after if last_meeting else None

        # RUONIA proxy: key_rate - 50bps
        ruonia = key_rate - _RUONIA_PROXY_OFFSET if key_rate is not None else None

        # CPI: latest published month (respects Rosstat publication lag)
        cpi_month = get_latest_published_cpi_month(as_of)
        cpi_yoy = _CPI_DATA.get(cpi_month) if cpi_month else None

        # Last CBR decision
        last_decision = last_meeting.decision if last_meeting else None

        return MacroSnapshot(
            key_rate=key_rate,
            ruonia_7d_avg=ruonia,
            cpi_yoy=cpi_yoy,
            last_cbr_decision=last_decision,
        )


# Default deposit spread below the key rate (D-04). Configurable per-tranche:
# longer-dated tranches may carry a different spread.
_DEFAULT_DEPOSIT_SPREAD_PP = Decimal("1.0")
_PCT_POINTS = Decimal(100)


def deposit_rate_as_of(as_of: date, spread_pp: Decimal = _DEFAULT_DEPOSIT_SPREAD_PP) -> Decimal:
    """Annual deposit rate as a decimal fraction = (key_rate - spread) / 100, as-of only.

    Reads ``key_rate`` via the look-ahead-safe meeting calendar (most recent CBR
    meeting on/before ``as_of``), reusing ``MacroContextProvider.get_snapshot`` --
    no new fetcher (anti-pattern 5). ``key_rate`` is in percentage points
    (``21.00`` means 21%), so the spread is subtracted in percentage points and
    the result is divided by 100 to a fraction (``21.00pp - 1.0pp -> 0.20``).

    Returns ``Decimal(0)`` before the first meeting in the calendar
    (``key_rate is None``). ``spread_pp`` is configurable per-tranche (D-04).
    """
    snap = MacroContextProvider().get_snapshot(as_of)
    if snap.key_rate is None:
        return Decimal(0)
    return (snap.key_rate - spread_pp) / _PCT_POINTS

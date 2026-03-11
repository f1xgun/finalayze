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

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import FXRate, KeyRateRecord

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
    CBRMeeting(date(2022, 9, 16), "interim", "hold", Decimal("7.50")),
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
    # 2025
    CBRMeeting(date(2025, 2, 14), "core", "hold", Decimal("21.00")),
    CBRMeeting(date(2025, 3, 21), "interim", "hold", Decimal("21.00")),
    CBRMeeting(date(2025, 4, 25), "core", "hold", Decimal("21.00")),
    CBRMeeting(date(2025, 6, 6), "interim", "hold", Decimal("21.00")),  # verify
    CBRMeeting(date(2025, 7, 25), "core", "cut", Decimal("20.00")),  # verify
    CBRMeeting(date(2025, 9, 12), "interim", "cut", Decimal("19.00")),  # verify
    CBRMeeting(date(2025, 10, 24), "core", "cut", Decimal("18.00")),  # verify
    CBRMeeting(date(2025, 12, 19), "interim", "cut", Decimal("17.00")),  # verify
    # 2026
    CBRMeeting(date(2026, 2, 13), "core", "cut", Decimal("16.00")),  # verify
    CBRMeeting(date(2026, 3, 20), "interim", None, None),  # scheduled
    CBRMeeting(date(2026, 4, 24), "core", None, None),
    CBRMeeting(date(2026, 6, 19), "interim", None, None),
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

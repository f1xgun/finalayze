"""CBR (Central Bank of Russia) API fetcher (Layer 2).

Fetches FX rates (REST XML) and key rate (SOAP XML) from cbr.ru.
Sync only — do NOT call from async code without asyncio.to_thread().
Uses httpx + lxml. No third-party CBR libraries.
Does NOT extend BaseFetcher — CBR API structure is fundamentally different.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
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

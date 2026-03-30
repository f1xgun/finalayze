"""Periodic FX rate updater (Layer 2).

Fetches USD/RUB rate from the CBR (Central Bank of Russia) daily XML feed
or falls back to a configurable static rate.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    from finalayze.markets.currency import CurrencyConverter

_log = structlog.get_logger()
_CBR_DAILY_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
_USD_CHAR_CODE = "USD"
_HTTP_TIMEOUT = 10.0


class FXRateService:
    """Fetches live FX rates and updates a CurrencyConverter."""

    def __init__(self, converter: CurrencyConverter) -> None:
        self._converter = converter
        self._client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        self._last_rate: Decimal | None = None
        self._last_rate_at: datetime | None = None

    async def update_usdrub(self) -> Decimal | None:
        """Fetch USD/RUB from CBR and update the converter. Returns the rate.

        On failure, returns the last successfully fetched rate (if any).
        """
        try:
            response = await self._client.get(_CBR_DAILY_URL)
            response.raise_for_status()
            rate = self._parse_cbr_xml(response.text)
            if rate is not None:
                self._converter.set_rate("USDRUB", rate)
                self._last_rate = rate
                self._last_rate_at = datetime.now(tz=UTC)
                _log.info("fx_rate_updated", pair="USDRUB", rate=float(rate))
                return rate
        except Exception:
            _log.warning("fx_rate_cbr_daily_failed", exc_info=True)

        # Fallback: return last known rate
        if self._last_rate is not None:
            age_s = (
                (datetime.now(tz=UTC) - self._last_rate_at).total_seconds()
                if self._last_rate_at
                else 0
            )
            _log.warning(
                "fx_rate_using_cached",
                rate=float(self._last_rate),
                cache_age_seconds=round(age_s),
            )
            return self._last_rate

        _log.error("fx_rate_no_fallback_available")
        return None

    @staticmethod
    def _parse_cbr_xml(xml_text: str) -> Decimal | None:
        """Parse CBR XML to extract USD rate.

        The CBR daily XML format has ``<Valute>`` elements with children:
        ``<CharCode>``, ``<Nominal>``, ``<VCurs>`` (the rate in Russian locale
        with a comma decimal separator).
        """
        root = ET.fromstring(xml_text)  # noqa: S314
        for valute in root.findall("Valute"):
            char_code = valute.findtext("CharCode")
            if char_code == _USD_CHAR_CODE:
                nominal = int(valute.findtext("Nominal", "1"))
                value_str = (valute.findtext("VCurs") or "0").replace(",", ".")
                return Decimal(value_str) / nominal
        return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

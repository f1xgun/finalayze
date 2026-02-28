"""Finnhub REST API data fetcher (Layer 2)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
from pydantic import ValidationError

from finalayze.core.exceptions import DataFetchError, RateLimitError
from finalayze.core.schemas import Candle
from finalayze.data.fetchers.base import BaseFetcher

if TYPE_CHECKING:
    from finalayze.data.rate_limiter import RateLimiter

# ── Timeframe mapping ───────────────────────────────────────────────────────
_TIMEFRAME_MAP: dict[str, str] = {
    "1d": "D",
    "1h": "60",
    "1m": "1",
}

# ── HTTP status codes ───────────────────────────────────────────────────────
_HTTP_OK = 200
_HTTP_RATE_LIMIT = 429

# ── Finnhub response status strings ────────────────────────────────────────
_STATUS_OK = "ok"
_STATUS_NO_DATA = "no_data"


class FinnhubFetcher(BaseFetcher):
    """Fetches OHLCV candles via the Finnhub free-tier REST API.

    Finnhub candle endpoint::

        GET https://finnhub.io/api/v1/stock/candle
            ?symbol=AAPL&resolution=D&from=<unix>&to=<unix>&token=<key>

    Response JSON keys: ``o`` (open), ``h`` (high), ``l`` (low), ``c`` (close),
    ``v`` (volume), ``t`` (timestamp array), ``s`` (status: ``"ok"`` or ``"no_data"``).
    """

    _BASE_URL = "https://finnhub.io/api/v1"
    _TIMEFRAME_MAP: dict[str, str] = _TIMEFRAME_MAP

    def __init__(
        self,
        api_key: str,
        market_id: str = "us",
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._api_key = api_key
        self._market_id = market_id
        self._rate_limiter = rate_limiter

    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Candle]:
        """Fetch OHLCV candles from Finnhub.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Start of the date range (inclusive).
            end: End of the date range (exclusive).
            timeframe: Bar size — one of "1d", "1h", "1m".

        Returns:
            List of Candle objects sorted by timestamp ascending.

        Raises:
            DataFetchError: On unknown timeframe, HTTP error, rate limit,
                API error, or when no data is available for the range.
        """
        resolution = self._resolve_timeframe(timeframe)
        from_ts = int(start.timestamp())
        to_ts = int(end.timestamp())

        url = f"{self._BASE_URL}/stock/candle"
        params: dict[str, str | int] = {
            "symbol": symbol,
            "resolution": resolution,
            "from": from_ts,
            "to": to_ts,
            "token": self._api_key,
        }

        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        with httpx.Client() as client:
            try:
                response = client.get(url, params=params)
            except httpx.HTTPError as exc:
                msg = f"HTTP request failed: {exc}"
                raise DataFetchError(msg) from exc

        if response.status_code == _HTTP_RATE_LIMIT:
            msg = f"Finnhub rate limit exceeded (HTTP {_HTTP_RATE_LIMIT})"
            raise RateLimitError(msg)

        if response.status_code != _HTTP_OK:
            msg = f"Finnhub API error: HTTP {response.status_code}"
            raise DataFetchError(msg)

        data = response.json()
        status = data.get("s")

        if status == _STATUS_NO_DATA:
            msg = "no data available"
            raise DataFetchError(msg)

        if status != _STATUS_OK:
            msg = f"Finnhub returned unexpected status: {status!r}"
            raise DataFetchError(msg)

        if not data.get("t"):
            msg = "no data available"
            raise DataFetchError(msg)

        return self._parse_candles(data, symbol=symbol, timeframe=timeframe)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _resolve_timeframe(self, timeframe: str) -> str:
        """Map a human-readable timeframe to a Finnhub resolution string.

        Raises:
            DataFetchError: When the timeframe is not supported.
        """
        try:
            return self._TIMEFRAME_MAP[timeframe]
        except KeyError:
            supported = ", ".join(sorted(self._TIMEFRAME_MAP))
            msg = f"Unsupported timeframe {timeframe!r}. Supported: {supported}"
            raise DataFetchError(msg) from None

    def _parse_candles(
        self,
        data: dict[str, object],
        *,
        symbol: str,
        timeframe: str,
    ) -> list[Candle]:
        """Build sorted Candle list from a successful Finnhub response dict."""
        try:
            opens = [float(v) for v in _require_list(data, "o")]  # type: ignore[arg-type]
            highs = [float(v) for v in _require_list(data, "h")]  # type: ignore[arg-type]
            lows = [float(v) for v in _require_list(data, "l")]  # type: ignore[arg-type]
            closes = [float(v) for v in _require_list(data, "c")]  # type: ignore[arg-type]
            volumes = [int(v) for v in _require_list(data, "v")]  # type: ignore[call-overload]
            timestamps = [int(v) for v in _require_list(data, "t")]  # type: ignore[call-overload]

            return sorted(
                [
                    Candle(
                        symbol=symbol,
                        market_id=self._market_id,
                        timeframe=timeframe,
                        timestamp=datetime.fromtimestamp(ts, tz=UTC),
                        open=Decimal(str(o)),
                        high=Decimal(str(h)),
                        low=Decimal(str(lo)),
                        close=Decimal(str(c)),
                        volume=int(v),
                        source="finnhub",
                    )
                    for ts, o, h, lo, c, v in zip(
                        timestamps, opens, highs, lows, closes, volumes, strict=True
                    )
                ],
                key=lambda candle: candle.timestamp,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            msg = f"Unexpected response format from Finnhub: {exc}"
            raise DataFetchError(msg) from exc


def _require_list(data: dict[str, object], key: str) -> list[object]:
    """Extract a list field from a Finnhub response dict.

    Raises:
        DataFetchError: When the key is missing or its value is not a list.
    """
    val = data.get(key)
    if not isinstance(val, list):
        msg = f"Missing or non-list field '{key}' in Finnhub response"
        raise DataFetchError(msg)
    return val

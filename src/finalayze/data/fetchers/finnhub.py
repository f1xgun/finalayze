"""Finnhub REST API data fetcher (Layer 2)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import httpx

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import Candle
from finalayze.data.fetchers.base import BaseFetcher

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

    def __init__(self, api_key: str, market_id: str = "us") -> None:
        self._api_key = api_key
        self._market_id = market_id

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

        with httpx.Client() as client:
            try:
                response = client.get(url, params=params)
            except httpx.HTTPError as exc:
                msg = f"HTTP request failed: {exc}"
                raise DataFetchError(msg) from exc

        if response.status_code == _HTTP_RATE_LIMIT:
            msg = f"Finnhub rate limit exceeded (HTTP {_HTTP_RATE_LIMIT})"
            raise DataFetchError(msg)

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
        opens: list[float] = data["o"]  # type: ignore[assignment]
        highs: list[float] = data["h"]  # type: ignore[assignment]
        lows: list[float] = data["l"]  # type: ignore[assignment]
        closes: list[float] = data["c"]  # type: ignore[assignment]
        volumes: list[int] = data["v"]  # type: ignore[assignment]
        timestamps: list[int] = data["t"]  # type: ignore[assignment]

        candles: list[Candle] = [
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
        ]

        return sorted(candles, key=lambda candle: candle.timestamp)

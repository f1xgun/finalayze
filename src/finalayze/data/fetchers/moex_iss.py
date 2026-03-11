"""MOEX ISS REST data fetcher (Layer 2).

Thin httpx client over the MOEX Information & Statistical Server (ISS) REST API.
Handles ISS pagination (100 rows/page), chunk-by-year for multi-year fetches
(with dedup on year boundaries), MSK→UTC timezone conversion (including
pre-2014 DST), retry with backoff, and rate limiting.

Sync only — do NOT call from async code without ``asyncio.to_thread()``.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

import httpx
import structlog

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import Candle, TurnoverRecord
from finalayze.data.fetchers.base import BaseFetcher

if TYPE_CHECKING:
    from finalayze.data.rate_limiter import RateLimiter

_BASE_URL = "https://iss.moex.com/iss"
_PAGE_SIZE = 100
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0  # seconds (doubles each attempt)

_MOEX_MARKET_ID = "moex"
_ISS_SOURCE = "moex_iss"

# Moscow timezone — ZoneInfo handles pre-2014 UTC+4 and current UTC+3 automatically
_MSK_TZ = ZoneInfo("Europe/Moscow")

# Interval code for daily candles in ISS
_INTERVAL_1D = 24

# Minimum weekday index considered a weekend (Saturday=5, Sunday=6)
_WEEKEND_WEEKDAY_MIN = 5

# ISS VALTODAY is reported in millions of RUB; multiply by this to get raw RUB
_VALTODAY_MULTIPLIER = Decimal(1000000)

_log = structlog.get_logger()


class MoexISSFetcher(BaseFetcher):
    """Fetch MOEX index candles and market turnover from MOEX ISS REST API.

    Args:
        rate_limiter: Optional rate limiter to throttle requests.
    """

    def __init__(self, rate_limiter: RateLimiter | None = None) -> None:
        self._rate_limiter = rate_limiter
        self._client = httpx.Client(timeout=30.0)

    # ── Context manager ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

    def __enter__(self) -> MoexISSFetcher:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── Public API ───────────────────────────────────────────────────────────

    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Candle]:
        """Fetch OHLCV candles for a MOEX index symbol via ISS.

        Args:
            symbol: MOEX index ticker (e.g. "IMOEX", "RTSI").
            start: Start of the date range (inclusive, UTC-aware).
            end: End of the date range (exclusive, UTC-aware).
            timeframe: Bar size — only "1d" supported currently.

        Returns:
            List of Candle objects in ISS-returned order (typically ascending).

        Raises:
            DataFetchError: On HTTP errors or timeouts.
        """
        interval = self._resolve_interval(timeframe)
        all_candles: list[Candle] = []

        chunks = self._year_chunks(start, end)
        for chunk_start, chunk_end in chunks:
            chunk_candles = self._fetch_candles_for_range(
                symbol, chunk_start, chunk_end, interval, timeframe
            )
            all_candles.extend(chunk_candles)

        # Dedup is only needed when multiple year chunks are fetched, because
        # the ISS `till` param is inclusive and a day on a year boundary could
        # appear in both adjacent chunks.
        if len(chunks) > 1:
            return self._dedup_candles(all_candles)
        return all_candles

    def fetch_market_turnover(
        self,
        start: datetime,
        end: datetime,
    ) -> list[TurnoverRecord]:
        """Fetch aggregate MOEX stock market turnover for a date range.

        Issues one request per weekday (skips Sat/Sun). Holidays/non-trading
        days return empty data and are silently skipped.

        VALTODAY from the API is in millions of RUB; multiplied by 1_000_000
        to normalize to raw RUB.

        Args:
            start: Start of the date range (inclusive, UTC-aware).
            end: End of the date range (exclusive, UTC-aware).

        Returns:
            List of TurnoverRecord objects, one per trading day in range.

        Raises:
            DataFetchError: On HTTP errors or timeouts.
        """
        url = f"{_BASE_URL}/engines/stock/turnovers.json"
        records: list[TurnoverRecord] = []

        current = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while current < end:
            # §17-M1: Skip weekends
            if current.weekday() >= _WEEKEND_WEEKDAY_MIN:
                current += timedelta(days=1)
                continue

            date_str = current.strftime("%Y-%m-%d")
            data = self._get_json(url, params={"date": date_str})

            block = data.get("turnovers", {})
            columns: list[str] = block.get("columns", [])
            rows: list[list[Any]] = block.get("data", [])

            if not columns or not rows:
                current += timedelta(days=1)
                continue

            # Locate column indices
            try:
                name_idx = columns.index("NAME")
                val_idx = columns.index("VALTODAY")
            except ValueError:
                current += timedelta(days=1)
                continue

            # §16-C1: Filter for aggregate row NAME == "TOTALS"
            totals_row: list[Any] | None = None
            for row in rows:
                if row[name_idx] == "TOTALS":
                    totals_row = row
                    break

            if totals_row is None:
                current += timedelta(days=1)
                continue

            raw_val = totals_row[val_idx]
            if raw_val is None:
                current += timedelta(days=1)
                continue

            # §17-C1: VALTODAY is in millions of RUB → multiply by _VALTODAY_MULTIPLIER
            volume_rub = Decimal(str(raw_val)) * _VALTODAY_MULTIPLIER

            # §18-M4: Timestamp from current date (already UTC midnight)
            ts = current.replace(hour=0, minute=0, second=0, microsecond=0)

            records.append(TurnoverRecord(timestamp=ts, volume_rub=volume_rub))
            current += timedelta(days=1)

        return records

    # ── Private helpers ──────────────────────────────────────────────────────

    def _fetch_candles_for_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: int,
        timeframe: str,
    ) -> list[Candle]:
        """Paginate through ISS candle endpoint for a single date range."""
        url = f"{_BASE_URL}/history/engines/stock/markets/index/securities/{symbol}/candles.json"
        # §19-M1: ISS `till` is INCLUSIVE — subtract 1 day from the exclusive `end`
        from_str = start.strftime("%Y-%m-%d")
        till_str = (end - timedelta(days=1)).strftime("%Y-%m-%d")

        candles: list[Candle] = []
        offset = 0

        while True:
            params: dict[str, Any] = {
                "from": from_str,
                "till": till_str,
                "interval": interval,
                "start": offset,
            }
            data = self._get_json(url, params=params)
            rows = self._fetch_candles_page(data)

            for row in rows:
                candle = self._parse_candle_row(row, symbol, timeframe)
                if candle is not None:
                    candles.append(candle)

            if len(rows) < _PAGE_SIZE:
                break
            offset += len(rows)

        return candles

    def _fetch_candles_page(self, data: dict[str, Any]) -> list[list[Any]]:
        """Extract candle rows from an ISS response dict (defensive lookup)."""
        # §18-M3: Defensive key lookup
        block = data.get("candles") or data.get("history", {})
        rows: list[list[Any]] = cast("list[list[Any]]", block.get("data", []))
        return rows

    def _parse_candle_row(self, row: list[Any], symbol: str, timeframe: str) -> Candle | None:
        """Parse a single ISS candle row into a Candle schema object."""
        # ISS column order: open, close, high, low, value, volume, begin, end
        if len(row) < 8:  # noqa: PLR2004
            return None

        open_price, close_price, high_price, low_price = row[0], row[1], row[2], row[3]
        volume_val = row[4]  # value (turnover in RUB), not share volume
        _volume = row[5]  # share volume (often 0 for indices)
        begin_str: str = row[6]

        # Parse begin timestamp as MSK-naive, then convert to UTC
        ts_msk = datetime.strptime(begin_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_MSK_TZ)
        ts_utc = ts_msk.astimezone(UTC)

        return Candle(
            symbol=symbol,
            market_id=_MOEX_MARKET_ID,
            timeframe=timeframe,
            timestamp=ts_utc,
            open=Decimal(str(open_price)),
            high=Decimal(str(high_price)),
            low=Decimal(str(low_price)),
            close=Decimal(str(close_price)),
            volume=int(volume_val) if volume_val else 0,
            source=_ISS_SOURCE,
        )

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET a URL with retry/backoff; wrap errors as DataFetchError."""
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        last_exc: Exception | None = None
        backoff = _RETRY_BACKOFF

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.get(url, params=params)
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    _log.warning("moex_iss_retry", attempt=attempt + 1, url=url)
                    time.sleep(backoff)
                    backoff *= 2
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                raise DataFetchError(f"http_error: {status} fetching {url}") from exc
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    _log.warning("moex_iss_retry", attempt=attempt + 1, url=url)
                    time.sleep(backoff)
                    backoff *= 2

        cause = "network_error"
        if isinstance(last_exc, httpx.TimeoutException):
            cause = "timeout"
        raise DataFetchError(f"{cause} after {_MAX_RETRIES} attempts fetching {url}") from last_exc

    @staticmethod
    def _resolve_interval(timeframe: str) -> int:
        """Map timeframe string to ISS interval code."""
        mapping = {
            "1d": _INTERVAL_1D,
        }
        if timeframe not in mapping:
            msg = f"Unsupported timeframe '{timeframe}' for MoexISSFetcher"
            raise DataFetchError(msg)
        return mapping[timeframe]

    @staticmethod
    def _year_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """Split a date range into calendar-year segments.

        Each segment is [year_start, year_end) aligned to Jan 1 boundaries,
        clipped to the requested [start, end) range.
        """
        chunks: list[tuple[datetime, datetime]] = []
        current_year = start.year
        end_year = end.year

        while current_year <= end_year:
            chunk_start = max(
                start,
                datetime(current_year, 1, 1, 0, 0, 0, tzinfo=UTC),
            )
            next_year_start = datetime(current_year + 1, 1, 1, 0, 0, 0, tzinfo=UTC)
            chunk_end = min(end, next_year_start)

            if chunk_start < chunk_end:
                chunks.append((chunk_start, chunk_end))

            current_year += 1

        return chunks

    @staticmethod
    def _dedup_candles(candles: list[Candle]) -> list[Candle]:
        """Deduplicate candles by timestamp, preserving order (first wins)."""
        seen: set[datetime] = set()
        result: list[Candle] = []
        for candle in candles:
            if candle.timestamp not in seen:
                seen.add(candle.timestamp)
                result.append(candle)
        return result

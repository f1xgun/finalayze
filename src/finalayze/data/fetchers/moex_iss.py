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
from datetime import UTC, date, datetime, timedelta
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

# Emit a turnover-backfill progress log every N weekdays (only for ranges large
# enough to be slow); keeps long cold-cache fetches observable.
_TURNOVER_PROGRESS_EVERY = 60

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

        # Progress accounting: one rate-limited request per weekday, so a cold
        # multi-year range is slow and otherwise silent. Log every N weekdays so
        # callers (e.g. backtest runner) can track a long backfill.
        total_weekdays = sum(
            1
            for d in range((end - start).days)
            if (start + timedelta(days=d)).weekday() < _WEEKEND_WEEKDAY_MIN
        )
        processed = 0

        current = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while current < end:
            # §17-M1: Skip weekends
            if current.weekday() >= _WEEKEND_WEEKDAY_MIN:
                current += timedelta(days=1)
                continue

            processed += 1
            if (
                total_weekdays >= _TURNOVER_PROGRESS_EVERY
                and processed % _TURNOVER_PROGRESS_EVERY == 0
            ):
                _log.info(
                    "moex_turnover_fetch_progress",
                    processed=processed,
                    total=total_weekdays,
                    date=current.strftime("%Y-%m-%d"),
                )

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

    def fetch_dividends(self, secid: str) -> list[tuple[date, Decimal, str]]:
        """Fetch the dividend history for a MOEX security via ISS.

        Reads ``securities/{SECID}/dividends.json`` and returns one tuple per
        declared dividend keyed on ``registryclosedate`` — the record (cut-off)
        date, which is the look-ahead-safe ``as_of`` for dividend-yield
        backfill (RESEARCH Pitfall 7: this is the record date, NOT the
        declaration date). Rows missing a record date are skipped; banks and
        instruments with no dividend block yield ``[]``.

        Args:
            secid: MOEX security id (e.g. "SBER").

        Returns:
            List of ``(registryclosedate, value, currencyid)`` tuples.

        Raises:
            DataFetchError: On HTTP errors or timeouts.
        """
        url = f"{_BASE_URL}/securities/{secid}/dividends.json"
        data = self._get_json(url, params={"iss.meta": "off"})

        block = data.get("dividends", {})
        columns: list[str] = block.get("columns", [])
        rows: list[list[Any]] = block.get("data", [])
        if not columns or not rows:
            return []

        col = {name: idx for idx, name in enumerate(columns)}
        try:
            date_idx = col["registryclosedate"]
            value_idx = col["value"]
            currency_idx = col["currencyid"]
        except KeyError:
            return []

        dividends: list[tuple[date, Decimal, str]] = []
        for row in rows:
            raw_date = row[date_idx]
            raw_value = row[value_idx]
            # Skip rows missing the look-ahead-safe record date or value
            # (T-63.1-07: only Decimal-ify non-None cells).
            if not raw_date or raw_value is None:
                continue
            try:
                as_of = date.fromisoformat(str(raw_date))
            except ValueError:
                continue
            value = Decimal(str(raw_value))
            currency = str(row[currency_idx]) if row[currency_idx] is not None else ""
            dividends.append((as_of, value, currency))

        return dividends

    def fetch_issuesize(self, secid: str) -> int | None:
        """Fetch the outstanding share count (ISSUESIZE) for a MOEX security.

        Reads the ``description`` block of ``securities/{SECID}.json``. Returns
        ``None`` when ISS reports no ISSUESIZE (e.g. CIAN — RESEARCH Pitfall 6),
        never fabricating a count.

        Args:
            secid: MOEX security id (e.g. "SBER").

        Returns:
            The current share count as ``int``, or ``None`` when unavailable.

        Raises:
            DataFetchError: On HTTP errors or timeouts.
        """
        url = f"{_BASE_URL}/securities/{secid}.json"
        data = self._get_json(url, params={"iss.meta": "off", "iss.only": "description"})

        block = data.get("description", {})
        columns: list[str] = block.get("columns", [])
        rows: list[list[Any]] = block.get("data", [])
        if not columns or not rows:
            return None

        col = {name: idx for idx, name in enumerate(columns)}
        try:
            name_idx = col["name"]
            value_idx = col["value"]
        except KeyError:
            return None

        for row in rows:
            if row[name_idx] == "ISSUESIZE":
                raw_value = row[value_idx]
                if not raw_value:
                    return None
                try:
                    return int(raw_value)
                except (TypeError, ValueError):
                    return None
        return None

    def fetch_close_history(
        self,
        secid: str,
        start: datetime,
        end: datetime,
        board: str = "TQBR",
    ) -> list[tuple[date, Decimal]]:
        """Fetch daily CLOSE prices for a MOEX share/ETF from a shares board.

        Reads ``history/.../boards/{board}/securities/{SECID}.json`` (the shares
        engine), which carries a ``CLOSE`` column but NO capitalization column —
        so market_cap must be reconstructed via :meth:`reconstruct_market_cap`.
        Used as the SmartLab market_cap cross-check / gap-fill (BACKFILL-H-03), and
        for ETFs on other boards (e.g. ``board="TQTF"`` for the LQDT money-market
        fund — the instrument-integration battery).

        Args:
            secid: MOEX security id (e.g. "SBER", "LQDT").
            start: Start of the date range (inclusive, UTC-aware).
            end: End of the date range (exclusive, UTC-aware).
            board: Shares-engine board id (default "TQBR" shares; "TQTF" for ETFs).

        Returns:
            List of ``(TRADEDATE, CLOSE)`` tuples; rows with a None CLOSE are
            skipped (T-63.1-07).

        Raises:
            DataFetchError: On HTTP errors or timeouts.
        """
        url = (
            f"{_BASE_URL}/history/engines/stock/markets/shares/boards/{board}"
            f"/securities/{secid}.json"
        )
        # §19-M1: ISS `till` is INCLUSIVE — subtract 1 day from the exclusive `end`.
        from_str = start.strftime("%Y-%m-%d")
        till_str = (end - timedelta(days=1)).strftime("%Y-%m-%d")

        closes: list[tuple[date, Decimal]] = []
        offset = 0

        while True:
            params: dict[str, Any] = {
                "from": from_str,
                "till": till_str,
                "start": offset,
                "iss.meta": "off",
            }
            data = self._get_json(url, params=params)
            block = data.get("history", {})
            columns: list[str] = block.get("columns", [])
            rows: list[list[Any]] = block.get("data", [])

            if not columns or not rows:
                break

            col = {name: idx for idx, name in enumerate(columns)}
            try:
                date_idx = col["TRADEDATE"]
                close_idx = col["CLOSE"]
            except KeyError:
                break

            for row in rows:
                raw_date = row[date_idx]
                raw_close = row[close_idx]
                if not raw_date or raw_close is None:
                    continue
                try:
                    trade_date = date.fromisoformat(str(raw_date))
                except ValueError:
                    continue
                closes.append((trade_date, Decimal(str(raw_close))))

            if len(rows) < _PAGE_SIZE:
                break
            offset += len(rows)

        return closes

    def fetch_currency_close_history(
        self,
        secid: str,
        start: datetime,
        end: datetime,
        board: str = "CETS",
    ) -> list[tuple[date, Decimal]]:
        """Fetch daily CLOSE for a MOEX currency-market instrument (e.g. GLDRUB_TOM).

        Reads ``history/engines/currency/markets/selt/boards/{board}/securities/{SECID}.json``
        — the currency/selt engine (gold spot, FX pairs), a DIFFERENT engine from the
        index path :meth:`fetch_candles` uses (``.../markets/index/...``), so gold and
        FX series need this dedicated method. ``CETS`` is the live system board; the other
        currency boards (CNGD/LICU/SPEC) report all-zero rows and are not queried.

        Like the index series this is an unauthenticated ISS REST read (NO token/cert —
        the "MOEX data = Tinkoff gRPC only" invariant governs INSTRUMENT candles, not the
        public ISS index/currency statistics). Rows with a None or non-positive CLOSE
        (holiday / no-trade sessions on the currency board) are skipped, never fabricated.

        Args:
            secid: MOEX currency-market id (e.g. "GLDRUB_TOM", "CNYRUB_TOM").
            start: Start of the date range (inclusive, UTC-aware).
            end: End of the date range (exclusive, UTC-aware).
            board: Currency board id (default "CETS", the live system board).

        Returns:
            List of ``(TRADEDATE, CLOSE)`` tuples in ISS order (ascending).

        Raises:
            DataFetchError: On HTTP errors or timeouts.
        """
        url = (
            f"{_BASE_URL}/history/engines/currency/markets/selt/boards/"
            f"{board}/securities/{secid}.json"
        )
        # §19-M1: ISS `till` is INCLUSIVE — subtract 1 day from the exclusive `end`.
        from_str = start.strftime("%Y-%m-%d")
        till_str = (end - timedelta(days=1)).strftime("%Y-%m-%d")

        closes: list[tuple[date, Decimal]] = []
        offset = 0

        while True:
            params: dict[str, Any] = {
                "from": from_str,
                "till": till_str,
                "start": offset,
                "iss.meta": "off",
            }
            data = self._get_json(url, params=params)
            block = data.get("history", {})
            columns: list[str] = block.get("columns", [])
            rows: list[list[Any]] = block.get("data", [])

            if not columns or not rows:
                break

            col = {name: idx for idx, name in enumerate(columns)}
            try:
                date_idx = col["TRADEDATE"]
                close_idx = col["CLOSE"]
            except KeyError:
                break

            for row in rows:
                raw_date = row[date_idx]
                raw_close = row[close_idx]
                if not raw_date or raw_close is None:
                    continue
                try:
                    trade_date = date.fromisoformat(str(raw_date))
                except ValueError:
                    continue
                close = Decimal(str(raw_close))
                if close <= 0:  # holiday / no-trade row on the currency board — skip
                    continue
                closes.append((trade_date, close))

            if len(rows) < _PAGE_SIZE:
                break
            offset += len(rows)

        return closes

    @staticmethod
    def reconstruct_market_cap(close: Decimal, issuesize: int | None) -> Decimal | None:
        """Reconstruct an APPROXIMATE market_cap as ``CLOSE * ISSUESIZE``.

        Pure (no I/O). Returns ``None`` when ``issuesize`` is None so the
        gap-fill path can flag the value as unavailable rather than fabricate it.

        WARNING (RESEARCH Pitfall 5): ISS exposes only the *current* ISSUESIZE,
        so applying it to an older CLOSE over-/under-states historical market_cap
        across share-count changes. This reconstruction is the flagged-approximate
        gap-fill / cross-check; SmartLab's per-quarter market_cap is primary.

        Args:
            close: Daily CLOSE price.
            issuesize: Current outstanding share count, or None.

        Returns:
            ``close * issuesize`` as Decimal, or None when issuesize is None.
        """
        if issuesize is None:
            return None
        return close * issuesize

    # ── Private helpers ──────────────────────────────────────────────────────

    def _fetch_candles_for_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        _interval: int,
        timeframe: str,
    ) -> list[Candle]:
        """Paginate through ISS history endpoint for index securities.

        Uses ``/history/.../securities/{symbol}.json`` which returns daily
        OHLCV rows for index tickers like IMOEX and RTSI.  The older
        ``/candles.json`` sub-path returns empty data for these instruments.
        """
        url = f"{_BASE_URL}/history/engines/stock/markets/index/securities/{symbol}.json"
        # §19-M1: ISS `till` is INCLUSIVE — subtract 1 day from the exclusive `end`
        from_str = start.strftime("%Y-%m-%d")
        till_str = (end - timedelta(days=1)).strftime("%Y-%m-%d")

        candles: list[Candle] = []
        offset = 0

        while True:
            params: dict[str, Any] = {
                "from": from_str,
                "till": till_str,
                "start": offset,
            }
            data = self._get_json(url, params=params)
            block = data.get("history", {})
            columns: list[str] = block.get("columns", [])
            rows: list[list[Any]] = block.get("data", [])

            if not columns or not rows:
                break

            for row in rows:
                candle = self._parse_history_row(row, columns, symbol, timeframe)
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
        row[4]  # value (turnover in RUB) -- not used for Candle.volume
        share_volume = row[5]  # share volume (correct for volume-based indicators)
        begin_raw = row[6]

        # ISS may return non-string values (float, None) for the begin field
        if begin_raw is None:
            return None
        begin_str = str(begin_raw)

        # Parse begin timestamp as MSK-naive, then convert to UTC
        try:
            ts_msk = datetime.strptime(begin_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_MSK_TZ)
        except ValueError:
            return None
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
            volume=int(share_volume) if share_volume else 0,
            source=_ISS_SOURCE,
        )

    def _parse_history_row(
        self,
        row: list[Any],
        columns: list[str],
        symbol: str,
        timeframe: str,
    ) -> Candle | None:
        """Parse an ISS history row (column-keyed) into a Candle."""
        try:
            col = {name: idx for idx, name in enumerate(columns)}
            trade_date = row[col["TRADEDATE"]]
            open_price = row[col["OPEN"]]
            high_price = row[col["HIGH"]]
            low_price = row[col["LOW"]]
            close_price = row[col["CLOSE"]]
            volume = row[col.get("VOLUME", -1)] if "VOLUME" in col else 0
        except (KeyError, IndexError):
            return None

        if any(v is None for v in (trade_date, open_price, high_price, low_price, close_price)):
            return None

        ts_msk = datetime.strptime(str(trade_date), "%Y-%m-%d").replace(tzinfo=_MSK_TZ)
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
            volume=int(volume) if volume else 0,
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

"""Unit tests for FinnhubFetcher (Layer 2 data fetcher)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import httpx
import pytest

from finalayze.core.exceptions import DataFetchError, RateLimitError
from finalayze.core.schemas import Candle
from finalayze.data.fetchers.finnhub import FinnhubFetcher

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────

FAKE_API_KEY = "test_key_abc123"
SYMBOL = "AAPL"
MARKET_ID = "us"

# Timestamps (Unix epoch seconds)
TS_1 = 1_679_000_000
TS_2 = 1_679_086_400  # TS_1 + 86 400 s (one day later)

# OHLCV bar 1
OPEN_1 = "150.00"
HIGH_1 = "155.50"
LOW_1 = "149.00"
CLOSE_1 = "153.25"
VOLUME_1 = 1_000_000

# OHLCV bar 2
OPEN_2 = "154.00"
HIGH_2 = "158.00"
LOW_2 = "152.50"
CLOSE_2 = "157.75"
VOLUME_2 = 1_200_000

# HTTP status codes
HTTP_OK = 200
HTTP_RATE_LIMIT = 429
HTTP_FORBIDDEN = 403

# Expected candle count for the two-bar fixture
EXPECTED_CANDLE_COUNT = 2


def _ok_response(
    timestamps: list[int] | None = None,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    closes: list[float] | None = None,
    volumes: list[int] | None = None,
) -> dict[str, object]:
    """Build a minimal successful Finnhub candle response."""
    if timestamps is None:
        timestamps = [TS_1, TS_2]
    if opens is None:
        opens = [float(OPEN_1), float(OPEN_2)]
    if highs is None:
        highs = [float(HIGH_1), float(HIGH_2)]
    if lows is None:
        lows = [float(LOW_1), float(LOW_2)]
    if closes is None:
        closes = [float(CLOSE_1), float(CLOSE_2)]
    if volumes is None:
        volumes = [VOLUME_1, VOLUME_2]
    return {
        "s": "ok",
        "o": opens,
        "h": highs,
        "l": lows,
        "c": closes,
        "v": volumes,
        "t": timestamps,
    }


def _mock_httpx_response(status_code: int, json_body: dict[str, object]) -> MagicMock:
    """Return a mock that mimics an httpx.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_body
    return mock_resp


@pytest.fixture
def fetcher() -> FinnhubFetcher:
    return FinnhubFetcher(api_key=FAKE_API_KEY, market_id=MARKET_ID)


@pytest.fixture
def date_range() -> tuple[datetime, datetime]:
    start = datetime(2023, 3, 16, tzinfo=UTC)
    end = datetime(2023, 3, 18, tzinfo=UTC)
    return start, end


# ── Tests ───────────────────────────────────────────────────────────────────


class TestFinnhubFetcherSuccess:
    @patch("finalayze.data.fetchers.finnhub.httpx.Client")
    def test_fetch_candles_returns_candles(
        self,
        mock_client_cls: MagicMock,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """Successful response yields Candle objects with correct field values."""
        mock_client_cls.return_value.__enter__.return_value.get.return_value = _mock_httpx_response(
            HTTP_OK, _ok_response()
        )

        start, end = date_range
        candles = fetcher.fetch_candles(SYMBOL, start, end, timeframe="1d")

        assert len(candles) == EXPECTED_CANDLE_COUNT

        c0 = candles[0]
        assert isinstance(c0, Candle)
        assert c0.symbol == SYMBOL
        assert c0.market_id == MARKET_ID
        assert c0.timeframe == "1d"
        assert c0.source == "finnhub"
        assert c0.open == Decimal(OPEN_1)
        assert c0.high == Decimal(HIGH_1)
        assert c0.low == Decimal(LOW_1)
        assert c0.close == Decimal(CLOSE_1)
        assert c0.volume == VOLUME_1

        c1 = candles[1]
        assert c1.open == Decimal(OPEN_2)
        assert c1.high == Decimal(HIGH_2)
        assert c1.low == Decimal(LOW_2)
        assert c1.close == Decimal(CLOSE_2)
        assert c1.volume == VOLUME_2

    @patch("finalayze.data.fetchers.finnhub.httpx.Client")
    def test_fetch_candles_timestamps_are_utc(
        self,
        mock_client_cls: MagicMock,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """All returned Candle timestamps must be UTC-aware."""
        mock_client_cls.return_value.__enter__.return_value.get.return_value = _mock_httpx_response(
            HTTP_OK, _ok_response()
        )

        start, end = date_range
        candles = fetcher.fetch_candles(SYMBOL, start, end)

        for candle in candles:
            assert candle.timestamp.tzinfo is not None, f"timestamp {candle.timestamp} is naive"
            assert candle.timestamp.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    @patch("finalayze.data.fetchers.finnhub.httpx.Client")
    def test_fetch_candles_prices_are_decimal(
        self,
        mock_client_cls: MagicMock,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """All price fields on returned Candles must be Decimal, not float."""
        mock_client_cls.return_value.__enter__.return_value.get.return_value = _mock_httpx_response(
            HTTP_OK, _ok_response()
        )

        start, end = date_range
        candles = fetcher.fetch_candles(SYMBOL, start, end)

        for candle in candles:
            assert isinstance(candle.open, Decimal), "open is not Decimal"
            assert isinstance(candle.high, Decimal), "high is not Decimal"
            assert isinstance(candle.low, Decimal), "low is not Decimal"
            assert isinstance(candle.close, Decimal), "close is not Decimal"

    @patch("finalayze.data.fetchers.finnhub.httpx.Client")
    def test_fetch_candles_sorted_ascending(
        self,
        mock_client_cls: MagicMock,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """Candles are returned sorted by timestamp even if the API gives them out of order."""
        # Provide timestamps in descending order intentionally
        reversed_body = _ok_response(
            timestamps=[TS_2, TS_1],
            opens=[float(OPEN_2), float(OPEN_1)],
            highs=[float(HIGH_2), float(HIGH_1)],
            lows=[float(LOW_2), float(LOW_1)],
            closes=[float(CLOSE_2), float(CLOSE_1)],
            volumes=[VOLUME_2, VOLUME_1],
        )
        mock_client_cls.return_value.__enter__.return_value.get.return_value = _mock_httpx_response(
            HTTP_OK, reversed_body
        )

        start, end = date_range
        candles = fetcher.fetch_candles(SYMBOL, start, end)

        assert len(candles) == EXPECTED_CANDLE_COUNT
        assert candles[0].timestamp < candles[1].timestamp
        # The earlier timestamp should correspond to OPEN_1
        assert candles[0].open == Decimal(OPEN_1)


class TestFinnhubFetcherErrors:
    @patch("finalayze.data.fetchers.finnhub.httpx.Client")
    def test_fetch_candles_no_data(
        self,
        mock_client_cls: MagicMock,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """s='no_data' response raises DataFetchError with 'no data available'."""
        no_data_body: dict[str, object] = {"s": "no_data"}
        mock_client_cls.return_value.__enter__.return_value.get.return_value = _mock_httpx_response(
            HTTP_OK, no_data_body
        )

        start, end = date_range
        with pytest.raises(DataFetchError, match="no data available"):
            fetcher.fetch_candles(SYMBOL, start, end)

    @patch("finalayze.data.fetchers.finnhub.httpx.Client")
    def test_fetch_candles_http_error(
        self,
        mock_client_cls: MagicMock,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """HTTP 429 (rate limit) raises DataFetchError."""
        mock_client_cls.return_value.__enter__.return_value.get.return_value = _mock_httpx_response(
            HTTP_RATE_LIMIT, {}
        )

        start, end = date_range
        with pytest.raises(DataFetchError):
            fetcher.fetch_candles(SYMBOL, start, end)

    def test_fetch_candles_unknown_timeframe(
        self,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """Unsupported timeframe raises DataFetchError before making any HTTP call."""
        start, end = date_range
        with pytest.raises(DataFetchError, match="Unsupported timeframe"):
            fetcher.fetch_candles(SYMBOL, start, end, timeframe="5m")

    @patch("finalayze.data.fetchers.finnhub.httpx.Client")
    def test_fetch_candles_empty_arrays_raises(
        self,
        mock_client_cls: MagicMock,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """s='ok' with empty arrays raises DataFetchError with 'no data available'."""
        empty_body: dict[str, object] = {
            "s": "ok",
            "c": [],
            "t": [],
            "o": [],
            "h": [],
            "l": [],
            "v": [],
        }
        mock_client_cls.return_value.__enter__.return_value.get.return_value = _mock_httpx_response(
            HTTP_OK, empty_body
        )

        start, end = date_range
        with pytest.raises(DataFetchError, match="no data available"):
            fetcher.fetch_candles(SYMBOL, start, end)

    @patch("finalayze.data.fetchers.finnhub.httpx.Client")
    def test_fetch_candles_http_timeout_raises(
        self,
        mock_client_cls: MagicMock,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """TimeoutException from httpx is wrapped in DataFetchError."""
        mock_get = mock_client_cls.return_value.__enter__.return_value.get
        mock_get.side_effect = httpx.TimeoutException("request timed out")

        start, end = date_range
        with pytest.raises(DataFetchError, match="HTTP request failed"):
            fetcher.fetch_candles(SYMBOL, start, end)

    @patch("finalayze.data.fetchers.finnhub.httpx.Client")
    def test_fetch_candles_http_generic_error_raises(
        self,
        mock_client_cls: MagicMock,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """HTTP 500 response raises DataFetchError."""
        http_500 = 500
        mock_client_cls.return_value.__enter__.return_value.get.return_value = _mock_httpx_response(
            http_500, {}
        )

        start, end = date_range
        with pytest.raises(DataFetchError, match="Finnhub API error"):
            fetcher.fetch_candles(SYMBOL, start, end)

    @patch("finalayze.data.fetchers.finnhub.httpx.Client")
    def test_fetch_candles_rate_limit_raises_rate_limit_error(
        self,
        mock_client_cls: MagicMock,
        fetcher: FinnhubFetcher,
        date_range: tuple[datetime, datetime],
    ) -> None:
        """HTTP 429 raises RateLimitError specifically (not just DataFetchError)."""
        mock_client_cls.return_value.__enter__.return_value.get.return_value = _mock_httpx_response(
            HTTP_RATE_LIMIT, {}
        )

        start, end = date_range
        with pytest.raises(RateLimitError):
            fetcher.fetch_candles(SYMBOL, start, end)

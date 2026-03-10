"""Tests for MoexISSFetcher — MOEX ISS REST client."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import httpx
import pytest

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import TurnoverRecord
from finalayze.data.fetchers.moex_iss import MoexISSFetcher


@pytest.fixture
def fetcher() -> MoexISSFetcher:
    return MoexISSFetcher()


# Sample ISS candle response (single page)
_ISS_CANDLES_RESPONSE = {
    "candles": {
        "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
        "data": [
            [3200.0, 3210.5, 3215.0, 3195.0, 1e8, 0, "2024-01-15 10:00:00", "2024-01-15 23:49:59"],
            [
                3210.5,
                3220.0,
                3225.0,
                3200.0,
                1.1e8,
                0,
                "2024-01-16 10:00:00",
                "2024-01-16 23:49:59",
            ],
        ],
    },
}

_ISS_CANDLES_EMPTY = {
    "candles": {
        "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
        "data": [],
    },
}

# Turnover response from /iss/engines/stock/turnovers.json?date=YYYY-MM-DD
# Aggregate row identified by NAME == "TOTALS", value in VALTODAY (millions RUB)
_ISS_TURNOVER_RESPONSE = {
    "turnovers": {
        "columns": ["NAME", "ID", "VALTODAY", "VALTODAY_USD", "NUMTRADES", "UPDATETIME", "TITLE"],
        "data": [
            ["stock", 1, 850000.0, 9500000000.0, 320000, "19:04:59", "Фондовый рынок"],
            ["TOTALS", 0, 1500000.0, 16800000000.0, 500000, "19:04:59", "Итого"],
        ],
    },
}

_ISS_TURNOVER_EMPTY = {"turnovers": {"columns": [], "data": []}}


class TestMoexISSFetcherCandles:
    @patch("time.sleep")
    def test_fetch_candles_basic(self, _sleep: MagicMock, fetcher: MoexISSFetcher) -> None:
        """Fetches IMOEX candles, converts MSK timestamps to UTC."""
        mock_ok = MagicMock(status_code=200)
        mock_ok.json.return_value = _ISS_CANDLES_RESPONSE
        mock_ok.raise_for_status = MagicMock()
        mock_empty = MagicMock(status_code=200)
        mock_empty.json.return_value = _ISS_CANDLES_EMPTY
        mock_empty.raise_for_status = MagicMock()

        with patch.object(fetcher, "_client") as mock_client:
            mock_client.get.side_effect = [mock_ok, mock_empty]
            candles = fetcher.fetch_candles(
                "IMOEX",
                datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 17, 0, 0, tzinfo=UTC),
            )

        assert len(candles) == 2
        assert candles[0].symbol == "IMOEX"
        assert candles[0].market_id == "moex"
        assert candles[0].source == "moex_iss"
        assert candles[0].timestamp.tzinfo is not None
        # MSK 10:00 (UTC+3) = UTC 07:00
        assert candles[0].timestamp.hour == 7

    @patch("time.sleep")
    def test_fetch_candles_pre2014_timezone(
        self, _sleep: MagicMock, fetcher: MoexISSFetcher
    ) -> None:
        """Pre-2014 Russia was UTC+4 (no DST since 2011). MSK 10:00 = UTC 06:00."""
        response_2013 = {
            "candles": {
                "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
                "data": [
                    [
                        1500.0,
                        1510.0,
                        1515.0,
                        1495.0,
                        5e7,
                        0,
                        "2013-06-15 10:00:00",
                        "2013-06-15 23:49:59",
                    ],
                ],
            },
        }
        mock_ok = MagicMock(status_code=200)
        mock_ok.json.return_value = response_2013
        mock_ok.raise_for_status = MagicMock()
        mock_empty = MagicMock(status_code=200)
        mock_empty.json.return_value = _ISS_CANDLES_EMPTY
        mock_empty.raise_for_status = MagicMock()

        with patch.object(fetcher, "_client") as mock_client:
            mock_client.get.side_effect = [mock_ok, mock_empty]
            candles = fetcher.fetch_candles(
                "IMOEX",
                datetime(2013, 6, 15, 0, 0, tzinfo=UTC),
                datetime(2013, 6, 16, 0, 0, tzinfo=UTC),
            )

        assert len(candles) == 1
        # 2013: Russia UTC+4. MSK 10:00 = UTC 06:00
        assert candles[0].timestamp.hour == 6

    @patch("time.sleep")
    def test_fetch_candles_empty(self, _sleep: MagicMock, fetcher: MoexISSFetcher) -> None:
        mock_empty = MagicMock(status_code=200)
        mock_empty.json.return_value = _ISS_CANDLES_EMPTY
        mock_empty.raise_for_status = MagicMock()

        with patch.object(fetcher, "_client") as mock_client:
            mock_client.get.return_value = mock_empty
            candles = fetcher.fetch_candles(
                "IMOEX",
                datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 17, 0, 0, tzinfo=UTC),
            )
        assert candles == []

    @patch("time.sleep")
    def test_fetch_candles_http_error(self, _sleep: MagicMock, fetcher: MoexISSFetcher) -> None:
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.get.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock(status_code=500)
            )
            with pytest.raises(DataFetchError, match="http_error"):
                fetcher.fetch_candles(
                    "IMOEX",
                    datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 17, 0, 0, tzinfo=UTC),
                )

    @patch("time.sleep")
    def test_fetch_candles_timeout(self, _sleep: MagicMock, fetcher: MoexISSFetcher) -> None:
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.get.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(DataFetchError, match="timeout"):
                fetcher.fetch_candles(
                    "IMOEX",
                    datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 17, 0, 0, tzinfo=UTC),
                )

    @patch("time.sleep")
    def test_fetch_candles_network_error(self, _sleep: MagicMock, fetcher: MoexISSFetcher) -> None:
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.get.side_effect = httpx.ConnectError("connection refused")
            with pytest.raises(DataFetchError, match="network_error"):
                fetcher.fetch_candles(
                    "IMOEX",
                    datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 17, 0, 0, tzinfo=UTC),
                )


class TestMoexISSFetcherTurnover:
    @patch("time.sleep")
    def test_fetch_turnover_filters_totals_row(
        self, _sleep: MagicMock, fetcher: MoexISSFetcher
    ) -> None:
        """Only NAME=TOTALS row extracted; per-market rows filtered out.
        VALTODAY in millions RUB → multiply by 1_000_000. One request per day."""
        mock_day1 = MagicMock(status_code=200)
        mock_day1.json.return_value = _ISS_TURNOVER_RESPONSE
        mock_day1.raise_for_status = MagicMock()
        mock_day2 = MagicMock(status_code=200)
        mock_day2.json.return_value = {
            "turnovers": {
                "columns": [
                    "NAME",
                    "ID",
                    "VALTODAY",
                    "VALTODAY_USD",
                    "NUMTRADES",
                    "UPDATETIME",
                    "TITLE",
                ],
                "data": [
                    ["TOTALS", 0, 1600000.0, 17000000000.0, 520000, "19:04:59", "Итого"],
                ],
            },
        }
        mock_day2.raise_for_status = MagicMock()

        with patch.object(fetcher, "_client") as mock_client:
            mock_client.get.side_effect = [mock_day1, mock_day2]
            records = fetcher.fetch_market_turnover(
                datetime(2024, 1, 15, 0, 0, tzinfo=UTC),  # Monday
                datetime(2024, 1, 17, 0, 0, tzinfo=UTC),  # Wednesday (exclusive)
            )

        # 2 trading days → 2 records (one per day, TOTALS only)
        # VALTODAY 1500000 (millions) * 1_000_000 = 1_500_000_000_000
        assert len(records) == 2
        assert isinstance(records[0], TurnoverRecord)
        assert records[0].volume_rub == Decimal(1500000000000)
        assert records[1].volume_rub == Decimal(1600000000000)

    @patch("time.sleep")
    def test_fetch_turnover_skips_weekends(
        self, _sleep: MagicMock, fetcher: MoexISSFetcher
    ) -> None:
        """Weekend dates are skipped without HTTP requests."""
        with patch.object(fetcher, "_client") as mock_client:
            records = fetcher.fetch_market_turnover(
                datetime(2024, 1, 13, 0, 0, tzinfo=UTC),  # Saturday
                datetime(2024, 1, 15, 0, 0, tzinfo=UTC),  # Monday (exclusive)
            )
        # No HTTP calls should be made for weekend days
        mock_client.get.assert_not_called()
        assert records == []

    @patch("time.sleep")
    def test_fetch_turnover_skips_empty_days(
        self, _sleep: MagicMock, fetcher: MoexISSFetcher
    ) -> None:
        """Holiday/non-trading dates return empty — silently skipped."""
        mock_empty = MagicMock(status_code=200)
        mock_empty.json.return_value = _ISS_TURNOVER_EMPTY
        mock_empty.raise_for_status = MagicMock()

        with patch.object(fetcher, "_client") as mock_client:
            mock_client.get.return_value = mock_empty
            records = fetcher.fetch_market_turnover(
                datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 16, 0, 0, tzinfo=UTC),
            )
        assert records == []


class TestMoexISSFetcherPagination:
    @patch("time.sleep")
    def test_candle_pagination_multiple_pages(
        self, _sleep: MagicMock, fetcher: MoexISSFetcher
    ) -> None:
        """Verify pagination loop: 100 rows → fetch next page; <100 → stop."""
        # Page 1: exactly 100 rows (triggers pagination continue)
        page1_data = [
            [
                3200.0 + i,
                3210.0 + i,
                3215.0 + i,
                3195.0 + i,
                1e8,
                0,
                f"2024-01-{15 + i // 10:02d} 10:00:00",
                f"2024-01-{15 + i // 10:02d} 23:49:59",
            ]
            for i in range(100)
        ]
        page1 = {
            "candles": {
                "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
                "data": page1_data,
            }
        }
        # Page 2: fewer than 100 rows (stops pagination)
        page2_data = [
            [3300.0, 3310.0, 3315.0, 3295.0, 1e8, 0, "2024-01-25 10:00:00", "2024-01-25 23:49:59"]
        ]
        page2 = {
            "candles": {
                "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
                "data": page2_data,
            }
        }

        mock_p1 = MagicMock(status_code=200)
        mock_p1.json.return_value = page1
        mock_p1.raise_for_status = MagicMock()
        mock_p2 = MagicMock(status_code=200)
        mock_p2.json.return_value = page2
        mock_p2.raise_for_status = MagicMock()

        with patch.object(fetcher, "_client") as mock_client:
            mock_client.get.side_effect = [mock_p1, mock_p2]
            candles = fetcher.fetch_candles(
                "IMOEX",
                datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 26, 0, 0, tzinfo=UTC),
            )

        assert len(candles) == 101
        assert mock_client.get.call_count == 2


class TestMoexISSFetcherLifecycle:
    def test_context_manager(self) -> None:
        with MoexISSFetcher() as fetcher:
            assert fetcher._client is not None

    @patch("time.sleep")
    def test_rate_limiter_called(self, _sleep: MagicMock) -> None:
        limiter = MagicMock()
        fetcher = MoexISSFetcher(rate_limiter=limiter)

        mock_empty = MagicMock(status_code=200)
        mock_empty.json.return_value = _ISS_CANDLES_EMPTY
        mock_empty.raise_for_status = MagicMock()

        with patch.object(fetcher, "_client") as mock_client:
            mock_client.get.return_value = mock_empty
            fetcher.fetch_candles(
                "IMOEX",
                datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 17, 0, 0, tzinfo=UTC),
            )

        limiter.acquire.assert_called()

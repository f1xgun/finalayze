"""Tests verifying fetchers call rate_limiter.acquire() before HTTP/gRPC."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from finalayze.data.rate_limiter import RateLimiter


class TestFinnhubRateLimiter:
    """Verify FinnhubFetcher calls limiter.acquire() before HTTP."""

    def test_acquire_called_before_http(self) -> None:
        from finalayze.data.fetchers.finnhub import FinnhubFetcher

        limiter = MagicMock(spec=RateLimiter)
        fetcher = FinnhubFetcher(api_key="test", rate_limiter=limiter)

        call_order: list[str] = []
        limiter.acquire.side_effect = lambda: call_order.append("acquire")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "s": "ok",
            "t": [1700000000],
            "o": [150.0],
            "h": [151.0],
            "l": [149.0],
            "c": [150.5],
            "v": [1000],
        }

        with patch("finalayze.data.fetchers.finnhub.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = lambda *a, **kw: (
                call_order.append("http"),
                mock_response,
            )[1]
            mock_client_cls.return_value = mock_client

            fetcher.fetch_candles(
                symbol="AAPL",
                start=datetime(2026, 1, 1, tzinfo=UTC),
                end=datetime(2026, 1, 2, tzinfo=UTC),
            )

        assert call_order == ["acquire", "http"]

    def test_no_limiter_works(self) -> None:
        """FinnhubFetcher works without rate_limiter."""
        from finalayze.data.fetchers.finnhub import FinnhubFetcher

        fetcher = FinnhubFetcher(api_key="test")
        assert fetcher._rate_limiter is None


class TestNewsApiRateLimiter:
    """Verify NewsApiFetcher calls limiter.acquire() before HTTP."""

    def test_acquire_called_before_http(self) -> None:
        from finalayze.data.fetchers.newsapi import NewsApiFetcher

        limiter = MagicMock(spec=RateLimiter)
        fetcher = NewsApiFetcher(api_key="test", rate_limiter=limiter)

        call_order: list[str] = []
        limiter.acquire.side_effect = lambda: call_order.append("acquire")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "articles": []}

        with patch("finalayze.data.fetchers.newsapi.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = lambda *a, **kw: (
                call_order.append("http"),
                mock_response,
            )[1]
            mock_client_cls.return_value = mock_client

            fetcher.fetch_news(
                query="test",
                from_date=datetime(2026, 1, 1, tzinfo=UTC),
                to_date=datetime(2026, 1, 2, tzinfo=UTC),
            )

        assert call_order == ["acquire", "http"]


class TestTinkoffRateLimiter:
    """Verify TinkoffFetcher calls limiter.acquire() before asyncio.run()."""

    def test_acquire_called_before_grpc(self) -> None:
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

        limiter = MagicMock(spec=RateLimiter)
        registry = MagicMock()
        instrument = MagicMock()
        instrument.figi = "BBG000B9XRY4"
        registry.get.return_value = instrument

        fetcher = TinkoffFetcher(token="test", registry=registry, rate_limiter=limiter)  # noqa: S106

        call_order: list[str] = []
        limiter.acquire.side_effect = lambda: call_order.append("acquire")

        with patch("finalayze.data.fetchers.tinkoff_data.asyncio.run") as mock_run:
            mock_run.side_effect = lambda coro: (
                call_order.append("grpc"),
                coro.close(),  # close the coroutine to avoid warnings
                [],
            )[2]

            fetcher.fetch_candles(
                symbol="SBER",
                start=datetime(2026, 1, 1, tzinfo=UTC),
                end=datetime(2026, 1, 2, tzinfo=UTC),
            )

        assert call_order == ["acquire", "grpc"]

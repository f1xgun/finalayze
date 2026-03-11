"""Tests for CBRFetcher — CBR XML API client."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import httpx
import pytest

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import FXRate, KeyRateRecord
from finalayze.data.fetchers.cbr import CBRFetcher


@pytest.fixture
def fetcher() -> CBRFetcher:
    return CBRFetcher()


_FX_XML = b"""<?xml version="1.0" encoding="windows-1251"?>
<ValCurs ID="R01235" DateRange1="15.01.2024" DateRange2="16.01.2024"
         name="Foreign Currency Market Dynamic">
    <Record Date="15.01.2024" Id="R01235">
        <Nominal>1</Nominal>
        <Value>89,5000</Value>
    </Record>
    <Record Date="16.01.2024" Id="R01235">
        <Nominal>1</Nominal>
        <Value>89,7500</Value>
    </Record>
</ValCurs>"""

_KEY_RATE_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
<soap:Body>
<KeyRateXMLResponse xmlns="http://web.cbr.ru/">
<KeyRateXMLResult>
<KeyRate>
<KR>
<DT>2024-01-01T00:00:00</DT>
<Rate>16.00</Rate>
</KR>
<KR>
<DT>2024-02-20T00:00:00</DT>
<Rate>16.00</Rate>
</KR>
</KeyRate>
</KeyRateXMLResult>
</KeyRateXMLResponse>
</soap:Body>
</soap:Envelope>"""


class TestCBRFetcherFX:
    @patch("time.sleep")
    def test_fetch_fx_rates_basic(self, _sleep: MagicMock, fetcher: CBRFetcher) -> None:  # noqa: PT019
        mock_response = MagicMock(status_code=200, content=_FX_XML)
        mock_response.raise_for_status = MagicMock()
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.request.return_value = mock_response
            rates = fetcher.fetch_fx_rates(
                "USD",
                datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 17, 0, 0, tzinfo=UTC),
            )
        assert len(rates) == 2
        assert isinstance(rates[0], FXRate)
        assert rates[0].pair == "USDRUB"
        assert rates[0].rate == Decimal("89.5000")  # comma parsed correctly
        assert rates[0].timestamp.tzinfo is not None

    @patch("time.sleep")
    def test_fetch_fx_end_date_exclusive(
        self,
        _sleep: MagicMock,  # noqa: PT019
        fetcher: CBRFetcher,
    ) -> None:
        """end date is exclusive in our API — CBR query subtracts 1 day."""
        mock_response = MagicMock(status_code=200, content=_FX_XML)
        mock_response.raise_for_status = MagicMock()
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.request.return_value = mock_response
            fetcher.fetch_fx_rates(
                "USD",
                datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 17, 0, 0, tzinfo=UTC),
            )
            # Verify CBR was called with end-1day (inclusive)
            call_kwargs = mock_client.request.call_args
            params = call_kwargs.kwargs.get("params", {})
            assert params["date_req2"] == "16/01/2024"  # 17th - 1 = 16th

    @patch("time.sleep")
    def test_fetch_fx_http_error(self, _sleep: MagicMock, fetcher: CBRFetcher) -> None:  # noqa: PT019
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.request.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock(status_code=500)
            )
            with pytest.raises(DataFetchError):
                fetcher.fetch_fx_rates(
                    "USD",
                    datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 17, 0, 0, tzinfo=UTC),
                )


class TestCBRFetcherKeyRate:
    @patch("time.sleep")
    def test_fetch_key_rate_normalized_to_decimal(
        self,
        _sleep: MagicMock,  # noqa: PT019
        fetcher: CBRFetcher,
    ) -> None:
        """CBR returns 16.00 (percentage) — we store as 0.16 (decimal fraction)."""
        mock_response = MagicMock(status_code=200, content=_KEY_RATE_XML)
        mock_response.raise_for_status = MagicMock()
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.request.return_value = mock_response
            rates = fetcher.fetch_key_rate(
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 3, 1, 0, 0, tzinfo=UTC),
            )
        assert len(rates) == 2
        assert isinstance(rates[0], KeyRateRecord)
        # 16.00% → 0.16
        assert rates[0].rate == Decimal("0.16")
        assert rates[0].timestamp.tzinfo is not None

    @patch("time.sleep")
    def test_fetch_key_rate_http_error(self, _sleep: MagicMock, fetcher: CBRFetcher) -> None:  # noqa: PT019
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.request.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(DataFetchError):
                fetcher.fetch_key_rate(
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 3, 1, 0, 0, tzinfo=UTC),
                )


class TestCBRFetcherLifecycle:
    def test_context_manager(self) -> None:
        with CBRFetcher() as fetcher:
            assert fetcher._client is not None

    @patch("time.sleep")
    def test_rate_limiter_called(self, _sleep: MagicMock) -> None:  # noqa: PT019
        limiter = MagicMock()
        fetcher = CBRFetcher(rate_limiter=limiter)
        mock_response = MagicMock(status_code=200, content=_FX_XML)
        mock_response.raise_for_status = MagicMock()
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.request.return_value = mock_response
            fetcher.fetch_fx_rates(
                "USD",
                datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 17, 0, 0, tzinfo=UTC),
            )
        limiter.acquire.assert_called()

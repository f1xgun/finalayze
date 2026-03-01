"""Tests for 6D.15: FX rate tracking service."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from finalayze.markets.currency import CurrencyConverter
from finalayze.markets.fx_service import FXRateService

_SAMPLE_CBR_XML = """\
<?xml version="1.0" encoding="windows-1251"?>
<ValCurs Date="01.03.2026" name="Foreign Currency Market">
    <Valute ID="R01235">
        <NumCode>840</NumCode>
        <CharCode>USD</CharCode>
        <Nominal>1</Nominal>
        <Name>US Dollar</Name>
        <VCurs>92,5410</VCurs>
    </Valute>
    <Valute ID="R01239">
        <NumCode>978</NumCode>
        <CharCode>EUR</CharCode>
        <Nominal>1</Nominal>
        <Name>Euro</Name>
        <VCurs>100,1234</VCurs>
    </Valute>
</ValCurs>"""


class TestParseCbrXml:
    """Test CBR XML parsing."""

    def test_parse_extracts_usd_rate(self) -> None:
        """Should parse the USD rate from CBR XML."""
        rate = FXRateService._parse_cbr_xml(_SAMPLE_CBR_XML)
        assert rate is not None
        assert rate == Decimal("92.5410")

    def test_parse_returns_none_for_missing_currency(self) -> None:
        """Should return None when USD is not in the XML."""
        xml = (
            "<ValCurs>"
            "<Valute><CharCode>GBP</CharCode><Nominal>1</Nominal><VCurs>115,00</VCurs></Valute>"
            "</ValCurs>"
        )
        rate = FXRateService._parse_cbr_xml(xml)
        assert rate is None

    def test_parse_handles_nominal_greater_than_one(self) -> None:
        """Nominal > 1 should divide the rate."""
        xml = (
            "<ValCurs>"
            "<Valute><CharCode>USD</CharCode><Nominal>100</Nominal>"
            "<VCurs>9254,10</VCurs></Valute>"
            "</ValCurs>"
        )
        rate = FXRateService._parse_cbr_xml(xml)
        assert rate is not None
        assert rate == Decimal("92.5410")


class TestUpdateUsdrub:
    """Test the async update_usdrub method."""

    @pytest.mark.asyncio
    async def test_update_sets_rate_on_converter(self) -> None:
        """Successful fetch should update the converter rate."""
        converter = CurrencyConverter(base_currency="USD")
        service = FXRateService(converter)

        mock_response = AsyncMock()
        mock_response.text = _SAMPLE_CBR_XML
        mock_response.raise_for_status = lambda: None

        with patch.object(service._client, "get", return_value=mock_response):
            rate = await service.update_usdrub()

        assert rate == Decimal("92.5410")
        # Verify the converter was updated
        converted = converter.convert(Decimal(1), "USD", "RUB")
        assert converted == Decimal("92.5410")

        await service.close()

    @pytest.mark.asyncio
    async def test_update_returns_none_on_error(self) -> None:
        """Network error should return None without crashing."""
        converter = CurrencyConverter(base_currency="USD")
        original_rate = converter._rates["USDRUB"]
        service = FXRateService(converter)

        with patch.object(service._client, "get", side_effect=httpx.ConnectError("timeout")):
            rate = await service.update_usdrub()

        assert rate is None
        # Converter should retain old rate
        assert converter._rates["USDRUB"] == original_rate

        await service.close()

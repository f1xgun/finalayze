"""RED TDD suite for the MoexISSFetcher dividend / issuesize / market-cap extension.

Wave 0: RED. The ISS fetcher class exists, but the methods asserted here
(``fetch_dividends``, ``fetch_issuesize``, ``reconstruct_market_cap``) are added
in Plan 03 — so these tests fail on ``AttributeError`` until then.

Mirrors ``test_moex_iss_fetcher.py``: patch ``_get_json`` with fixture JSON;
no live network (T-63.1-01).
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from finalayze.data.fetchers.moex_iss import MoexISSFetcher

_FIXTURES = Path(__file__).parent / "fixtures"

# --- Named constants (ruff PLR2004) ------------------------------------------
_SBER = "SBER"
_CIAN = "CIAN"
_SBER_ISSUESIZE = 21_586_948_000
_DIV_2025_DATE = date(2025, 7, 18)
_DIV_2025_VALUE = Decimal("34.84")
_DIV_2024_DATE = date(2024, 7, 10)
_DIV_2024_VALUE = Decimal("25.0")
_DIV_CURRENCY = "RUB"
_EXPECTED_DIVIDEND_COUNT = 2
_CLOSE_PRICE = Decimal("285.5")
_EXPECTED_MARKET_CAP = _CLOSE_PRICE * _SBER_ISSUESIZE


@pytest.fixture
def fetcher() -> MoexISSFetcher:
    return MoexISSFetcher()


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


class TestFetchDividends:
    def test_fetch_dividends(self, fetcher: MoexISSFetcher) -> None:
        """dividends.json -> [(registryclosedate, Decimal value, currency), ...]."""
        with patch.object(fetcher, "_get_json", return_value=_load("iss_sber_dividends.json")):
            divs = fetcher.fetch_dividends(_SBER)

        assert len(divs) == _EXPECTED_DIVIDEND_COUNT
        assert (_DIV_2025_DATE, _DIV_2025_VALUE, _DIV_CURRENCY) in divs
        assert (_DIV_2024_DATE, _DIV_2024_VALUE, _DIV_CURRENCY) in divs
        # registryclosedate is the look-ahead-safe as_of (RESEARCH Pitfall 7).
        for as_of, value, currency in divs:
            assert isinstance(as_of, date)
            assert isinstance(value, Decimal)
            assert currency == _DIV_CURRENCY


class TestFetchIssuesize:
    def test_fetch_issuesize(self, fetcher: MoexISSFetcher) -> None:
        """SBER description ISSUESIZE -> int."""
        with patch.object(fetcher, "_get_json", return_value=_load("iss_sber_description.json")):
            size = fetcher.fetch_issuesize(_SBER)
        assert size == _SBER_ISSUESIZE

    def test_fetch_issuesize_missing_returns_none(self, fetcher: MoexISSFetcher) -> None:
        """CIAN has empty ISSUESIZE -> None (RESEARCH Pitfall 6)."""
        with patch.object(fetcher, "_get_json", return_value=_load("iss_cian_description.json")):
            size = fetcher.fetch_issuesize(_CIAN)
        assert size is None


class TestReconstructMarketCap:
    def test_reconstruct_market_cap(self, fetcher: MoexISSFetcher) -> None:
        """market_cap = CLOSE * ISSUESIZE."""
        cap = fetcher.reconstruct_market_cap(_CLOSE_PRICE, _SBER_ISSUESIZE)
        assert cap == _EXPECTED_MARKET_CAP

    def test_reconstruct_market_cap_none_issuesize(self, fetcher: MoexISSFetcher) -> None:
        """issuesize None -> market_cap None (no fabrication)."""
        cap = fetcher.reconstruct_market_cap(_CLOSE_PRICE, None)
        assert cap is None

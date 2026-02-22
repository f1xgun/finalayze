"""Unit tests for data fetchers (Layer 2)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finalayze.core.schemas import Candle
from finalayze.data.fetchers.base import BaseFetcher
from finalayze.data.fetchers.yfinance import YFinanceFetcher

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────

EXPECTED_CANDLE_COUNT = 2
AAPL_OPEN_1 = Decimal("150.00")
AAPL_HIGH_1 = Decimal("155.50")
AAPL_LOW_1 = Decimal("149.00")
AAPL_CLOSE_1 = Decimal("153.25")
AAPL_VOLUME_1 = 1_000_000

AAPL_OPEN_2 = Decimal("154.00")
AAPL_HIGH_2 = Decimal("158.00")
AAPL_LOW_2 = Decimal("152.50")
AAPL_CLOSE_2 = Decimal("157.75")
AAPL_VOLUME_2 = 1_200_000


# ── BaseFetcher ─────────────────────────────────────────────────────────


class TestBaseFetcher:
    def test_is_abstract(self) -> None:
        """BaseFetcher cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFetcher()  # type: ignore[abstract]


# ── YFinanceFetcher ─────────────────────────────────────────────────────


class TestYFinanceFetcher:
    @pytest.fixture
    def fetcher(self) -> YFinanceFetcher:
        return YFinanceFetcher(market_id="us")

    def _make_mock_dataframe(self) -> pd.DataFrame:
        """Create a mock DataFrame mimicking yfinance output with 2 rows."""
        index = pd.DatetimeIndex(
            [
                datetime(2024, 1, 15, tzinfo=UTC),
                datetime(2024, 1, 16, tzinfo=UTC),
            ],
            name="Date",
        )
        return pd.DataFrame(
            {
                "Open": [150.00, 154.00],
                "High": [155.50, 158.00],
                "Low": [149.00, 152.50],
                "Close": [153.25, 157.75],
                "Volume": [1_000_000, 1_200_000],
            },
            index=index,
        )

    @patch("finalayze.data.fetchers.yfinance.yf")
    def test_fetch_returns_candles(self, mock_yf: MagicMock, fetcher: YFinanceFetcher) -> None:
        """fetch_candles returns correct Candle objects from yfinance data."""
        mock_yf.download.return_value = self._make_mock_dataframe()

        start = datetime(2024, 1, 15, tzinfo=UTC)
        end = datetime(2024, 1, 17, tzinfo=UTC)
        candles = fetcher.fetch_candles("AAPL", start, end, timeframe="1d")

        assert len(candles) == EXPECTED_CANDLE_COUNT

        # Verify first candle
        c0 = candles[0]
        assert isinstance(c0, Candle)
        assert c0.symbol == "AAPL"
        assert c0.market_id == "us"
        assert c0.timeframe == "1d"
        assert c0.open == AAPL_OPEN_1
        assert c0.high == AAPL_HIGH_1
        assert c0.low == AAPL_LOW_1
        assert c0.close == AAPL_CLOSE_1
        assert c0.volume == AAPL_VOLUME_1

        # Verify second candle
        c1 = candles[1]
        assert isinstance(c1, Candle)
        assert c1.symbol == "AAPL"
        assert c1.open == AAPL_OPEN_2
        assert c1.high == AAPL_HIGH_2
        assert c1.low == AAPL_LOW_2
        assert c1.close == AAPL_CLOSE_2
        assert c1.volume == AAPL_VOLUME_2

        # Verify yf.download was called correctly
        mock_yf.download.assert_called_once_with(
            "AAPL", start=start, end=end, interval="1d", progress=False
        )

    @patch("finalayze.data.fetchers.yfinance.yf")
    def test_fetch_empty_returns_empty_list(
        self, mock_yf: MagicMock, fetcher: YFinanceFetcher
    ) -> None:
        """fetch_candles returns empty list when yfinance returns empty DataFrame."""
        mock_yf.download.return_value = pd.DataFrame()

        start = datetime(2024, 1, 15, tzinfo=UTC)
        end = datetime(2024, 1, 17, tzinfo=UTC)
        candles = fetcher.fetch_candles("AAPL", start, end)

        assert candles == []

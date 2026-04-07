"""Unit tests for the MOEX dividend data pipeline.

Tests cover:
- TinkoffFetcher.fetch_dividends() method
- _setup_dividend_strategy() helper in run_iteration.py
- Static moex_dividends.yaml loading
- CachingFetcher.fetch_dividends() passthrough
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finalayze.data.fetchers.caching import CachingFetcher  # noqa: E402
from finalayze.strategies.dividend_gap import DividendEntry, DividendGapStrategy  # noqa: E402

with patch("sys.argv", ["run_iteration.py", "--name", "test", "--description", "test"]):
    import scripts.run_iteration as ri  # noqa: E402

_PRESETS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "src" / "finalayze" / "strategies" / "presets"
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_dividend(
    last_buy_date: datetime,
    dividend_net_units: int,
    dividend_net_nano: int,
) -> MagicMock:
    """Create a mock Tinkoff Dividend object."""
    div = MagicMock()
    div.last_buy_date = last_buy_date
    div.dividend_net = MagicMock()
    div.dividend_net.units = dividend_net_units
    div.dividend_net.nano = dividend_net_nano
    return div


def _make_mock_fetcher_with_dividends(
    dividends_by_symbol: dict[str, list[dict]],
) -> MagicMock:
    """Create a mock fetcher that has fetch_dividends returning given data."""
    fetcher = MagicMock()

    def _fetch_dividends(symbol: str, start: datetime, end: datetime) -> list[dict]:
        return dividends_by_symbol.get(symbol, [])

    fetcher.fetch_dividends = MagicMock(side_effect=_fetch_dividends)
    return fetcher


# ── TinkoffFetcher.fetch_dividends tests ─────────────────────────────────────


class TestTinkoffFetchDividends:
    """Tests for TinkoffFetcher.fetch_dividends() method."""

    def test_returns_list_of_dicts(self) -> None:
        """fetch_dividends returns a list of dicts with ex_date and amount."""
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

        registry = MagicMock()
        instrument = MagicMock()
        instrument.figi = "BBG004730N88"
        registry.get.return_value = instrument

        div = _make_mock_dividend(
            last_buy_date=datetime(2024, 7, 10, tzinfo=UTC),
            dividend_net_units=33,
            dividend_net_nano=300_000_000,
        )
        mock_response = MagicMock()
        mock_response.dividends = [div]

        fetcher = TinkoffFetcher(token="fake", registry=registry)  # noqa: S106
        fetcher._run_async = MagicMock(return_value=[div])
        with patch.object(fetcher, "_run_async", return_value=[div]):
            result = fetcher.fetch_dividends(
                "SBER",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 1, tzinfo=UTC),
            )

        assert isinstance(result, list)
        assert len(result) == 1
        assert "ex_date" in result[0]
        assert "amount" in result[0]
        # last_buy_date=2024-07-10 (Wed) → ex_date should be 2024-07-11 (Thu)
        assert result[0]["ex_date"].date() == datetime(2024, 7, 11, tzinfo=UTC).date()

    def test_maps_amount_correctly(self) -> None:
        """MoneyValue(units=25, nano=0) should map to 25.0."""
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

        registry = MagicMock()
        instrument = MagicMock()
        instrument.figi = "BBG004730N88"
        registry.get.return_value = instrument

        div = _make_mock_dividend(
            last_buy_date=datetime(2023, 7, 10, tzinfo=UTC),
            dividend_net_units=25,
            dividend_net_nano=0,
        )

        fetcher = TinkoffFetcher(token="fake", registry=registry)  # noqa: S106
        fetcher._run_async = MagicMock(return_value=[div])
        with patch.object(fetcher, "_run_async", return_value=[div]):
            result = fetcher.fetch_dividends(
                "SBER",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
            )

        assert result[0]["amount"] == pytest.approx(25.0)

    def test_maps_fractional_amount(self) -> None:
        """MoneyValue(units=6, nano=10_000_000) should map to ~6.01."""
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

        registry = MagicMock()
        instrument = MagicMock()
        instrument.figi = "BBG004730N88"
        registry.get.return_value = instrument

        div = _make_mock_dividend(
            last_buy_date=datetime(2022, 10, 11, tzinfo=UTC),
            dividend_net_units=6,
            dividend_net_nano=10_000_000,
        )

        fetcher = TinkoffFetcher(token="fake", registry=registry)  # noqa: S106
        fetcher._run_async = MagicMock(return_value=[div])
        with patch.object(fetcher, "_run_async", return_value=[div]):
            result = fetcher.fetch_dividends(
                "SBER",
                datetime(2022, 1, 1, tzinfo=UTC),
                datetime(2023, 1, 1, tzinfo=UTC),
            )

        assert result[0]["amount"] == pytest.approx(6.01)

    def test_handles_empty_response(self) -> None:
        """No dividends in range returns empty list."""
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

        registry = MagicMock()
        instrument = MagicMock()
        instrument.figi = "BBG004730N88"
        registry.get.return_value = instrument

        fetcher = TinkoffFetcher(token="fake", registry=registry)  # noqa: S106
        with patch.object(fetcher, "_run_async", return_value=[]):
            result = fetcher.fetch_dividends(
                "SBER",
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 6, 1, tzinfo=UTC),
            )

        assert result == []

    def test_shifts_friday_last_buy_to_monday(self) -> None:
        """last_buy_date on Friday should shift ex_date to Monday."""
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

        registry = MagicMock()
        instrument = MagicMock()
        instrument.figi = "BBG004730N88"
        registry.get.return_value = instrument

        # 2024-07-05 is a Friday
        div = _make_mock_dividend(
            last_buy_date=datetime(2024, 7, 5, tzinfo=UTC),
            dividend_net_units=10,
            dividend_net_nano=0,
        )

        fetcher = TinkoffFetcher(token="fake", registry=registry)  # noqa: S106
        fetcher._run_async = MagicMock(return_value=[div])
        with patch.object(fetcher, "_run_async", return_value=[div]):
            result = fetcher.fetch_dividends(
                "SBER",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 1, tzinfo=UTC),
            )

        # Friday + 1 = Saturday → skip to Monday 2024-07-08
        assert result[0]["ex_date"].date() == datetime(2024, 7, 8, tzinfo=UTC).date()

    def test_handles_api_error_gracefully(self) -> None:
        """API errors bubble up as DataFetchError."""
        from finalayze.core.exceptions import DataFetchError
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

        registry = MagicMock()
        instrument = MagicMock()
        instrument.figi = "BBG004730N88"
        registry.get.return_value = instrument

        fetcher = TinkoffFetcher(token="fake", registry=registry)  # noqa: S106
        with (
            patch.object(fetcher, "_run_async", side_effect=RuntimeError("gRPC unavailable")),
            pytest.raises(DataFetchError, match=r"dividend.*SBER"),
        ):
            fetcher.fetch_dividends(
                "SBER",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 1, tzinfo=UTC),
            )

    def test_respects_rate_limiter(self) -> None:
        """Rate limiter should be acquired before the API call."""
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

        registry = MagicMock()
        instrument = MagicMock()
        instrument.figi = "BBG004730N88"
        registry.get.return_value = instrument

        rate_limiter = MagicMock()

        fetcher = TinkoffFetcher(
            token="fake",  # noqa: S106
            registry=registry,
            rate_limiter=rate_limiter,
        )
        with patch.object(fetcher, "_run_async", return_value=[]):
            fetcher.fetch_dividends(
                "SBER",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 1, tzinfo=UTC),
            )

        rate_limiter.acquire.assert_called_once()


# ── _setup_dividend_strategy tests ───────────────────────────────────────────


class TestSetupDividendStrategy:
    """Tests for _setup_dividend_strategy() in run_iteration.py."""

    def test_populates_calendar_from_fetcher(self) -> None:
        """Strategy calendar should have entries from fetcher.fetch_dividends."""
        divs = {
            "SBER": [
                {"ex_date": datetime(2024, 7, 11, tzinfo=UTC), "amount": 33.3},
            ],
            "GAZP": [
                {"ex_date": datetime(2024, 7, 19, tzinfo=UTC), "amount": 19.49},
            ],
        }
        fetcher = _make_mock_fetcher_with_dividends(divs)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2025, 1, 1, tzinfo=UTC)

        strategy = ri._setup_dividend_gap_strategy(
            "ru_blue_chips", ["SBER", "GAZP"], fetcher, start, end
        )

        assert strategy is not None
        assert isinstance(strategy, DividendGapStrategy)
        # Check calendar was populated
        assert len(strategy._calendar.get("SBER", [])) == 1
        assert len(strategy._calendar.get("GAZP", [])) == 1
        assert strategy._calendar["SBER"][0].amount == pytest.approx(33.3)

    def test_returns_none_for_us_segment(self) -> None:
        """US segments should not get a dividend_gap strategy."""
        fetcher = MagicMock()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2025, 1, 1, tzinfo=UTC)

        result = ri._setup_dividend_gap_strategy("us_tech", ["AAPL", "MSFT"], fetcher, start, end)
        assert result is None

    def test_graceful_without_fetch_dividends(self) -> None:
        """When fetcher has no fetch_dividends, load from static YAML."""
        fetcher = MagicMock(spec=["fetch_candles"])  # No fetch_dividends
        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2025, 1, 1, tzinfo=UTC)

        strategy = ri._setup_dividend_gap_strategy(
            "ru_blue_chips", ["SBER", "GAZP", "LKOH", "GMKN"], fetcher, start, end
        )

        assert strategy is not None
        assert isinstance(strategy, DividendGapStrategy)
        # Should have loaded from static YAML (at least SBER has entries)
        assert len(strategy._calendar) > 0

    def test_skips_symbol_on_error(self) -> None:
        """If fetch_dividends raises for one symbol, others still work."""
        fetcher = MagicMock()
        call_count = 0

        def _fetch_divs(symbol: str, start: datetime, end: datetime) -> list[dict]:
            nonlocal call_count
            call_count += 1
            if symbol == "GAZP":
                raise RuntimeError("API error")
            return [{"ex_date": datetime(2024, 7, 11, tzinfo=UTC), "amount": 33.3}]

        fetcher.fetch_dividends = MagicMock(side_effect=_fetch_divs)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2025, 1, 1, tzinfo=UTC)

        strategy = ri._setup_dividend_gap_strategy(
            "ru_blue_chips", ["SBER", "GAZP", "LKOH"], fetcher, start, end
        )

        assert strategy is not None
        # SBER and LKOH should have entries; GAZP skipped
        assert "SBER" in strategy._calendar
        assert "GAZP" not in strategy._calendar
        assert "LKOH" in strategy._calendar


# ── Static moex_dividends.yaml tests ────────────────────────────────────────


class TestStaticDividendYaml:
    """Tests for the static moex_dividends.yaml file."""

    def test_yaml_file_exists(self) -> None:
        yaml_path = _PRESETS_DIR / "moex_dividends.yaml"
        assert yaml_path.exists(), f"Expected {yaml_path} to exist"

    def test_yaml_loads_correctly(self) -> None:
        yaml_path = _PRESETS_DIR / "moex_dividends.yaml"
        with yaml_path.open() as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)
        assert "SBER" in data
        assert "GAZP" in data
        assert "LKOH" in data

    def test_yaml_entries_have_required_fields(self) -> None:
        yaml_path = _PRESETS_DIR / "moex_dividends.yaml"
        with yaml_path.open() as f:
            data = yaml.safe_load(f)

        for symbol, entries in data.items():
            assert isinstance(entries, list), f"{symbol} should have a list of entries"
            for entry in entries:
                assert "ex_date" in entry, f"{symbol} entry missing ex_date"
                assert "amount" in entry, f"{symbol} entry missing amount"
                assert isinstance(entry["amount"], (int, float)), (
                    f"{symbol} amount should be numeric"
                )


# ── CachingFetcher.fetch_dividends tests ────────────────────────────────────


class TestCachingFetcherDividends:
    """Tests for CachingFetcher.fetch_dividends() passthrough."""

    def test_delegates_to_inner_fetcher(self, tmp_path: Path) -> None:
        """CachingFetcher should delegate fetch_dividends to inner fetcher."""
        inner = MagicMock()
        inner.fetch_dividends.return_value = [
            {"ex_date": datetime(2024, 7, 11, tzinfo=UTC), "amount": 33.3},
        ]

        caching = CachingFetcher(delegate=inner, cache_dir=tmp_path)
        result = caching.fetch_dividends(
            "SBER",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert len(result) == 1
        assert result[0]["amount"] == pytest.approx(33.3)
        inner.fetch_dividends.assert_called_once()

    def test_caches_result(self, tmp_path: Path) -> None:
        """Second call should return cached result, not call inner fetcher again."""
        inner = MagicMock()
        inner.fetch_dividends.return_value = [
            {"ex_date": datetime(2024, 7, 11, tzinfo=UTC), "amount": 33.3},
        ]

        caching = CachingFetcher(delegate=inner, cache_dir=tmp_path)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2025, 1, 1, tzinfo=UTC)

        # First call — cache miss (populates cache)
        caching.fetch_dividends("SBER", start, end)
        # Second call — cache hit (deserialises ex_date from string back to datetime)
        result2 = caching.fetch_dividends("SBER", start, end)

        assert len(result2) == 1
        assert result2[0]["amount"] == pytest.approx(33.3)
        assert isinstance(result2[0]["ex_date"], datetime)
        # Inner fetcher called only once
        assert inner.fetch_dividends.call_count == 1

    def test_returns_empty_when_inner_has_no_method(self, tmp_path: Path) -> None:
        """If inner fetcher lacks fetch_dividends, return empty list."""
        inner = MagicMock(spec=["fetch_candles"])  # No fetch_dividends

        caching = CachingFetcher(delegate=inner, cache_dir=tmp_path)
        result = caching.fetch_dividends(
            "SBER",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert result == []

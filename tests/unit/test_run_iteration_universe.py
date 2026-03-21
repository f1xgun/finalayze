"""Tests for run_iteration.py UNIVERSE dict and DividendEntry.status wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path so scripts/ can import config.settings
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import UNIVERSE dict
from scripts.run_iteration import UNIVERSE  # type: ignore[import-untyped]

# Toxic symbols that must be excluded from all ru_* segments
TOXIC_SYMBOLS = {"GAZP", "VTBR", "ALRS", "SNGS", "SNGSP", "IRAO"}


class TestUniverseRuBlueChips:
    """ru_blue_chips must not contain toxic symbols."""

    def test_no_toxic_symbols(self) -> None:
        ru_bc = set(UNIVERSE["ru_blue_chips"])
        overlap = ru_bc & TOXIC_SYMBOLS
        assert overlap == set(), f"ru_blue_chips contains toxic symbols: {overlap}"


class TestUniverseRuEnergy:
    """ru_energy must not contain GAZP or SNGS."""

    def test_no_toxic_symbols(self) -> None:
        ru_en = set(UNIVERSE["ru_energy"])
        overlap = ru_en & TOXIC_SYMBOLS
        assert overlap == set(), f"ru_energy contains toxic symbols: {overlap}"


class TestUniverseRuFinance:
    """ru_finance must not contain VTBR."""

    def test_no_toxic_symbols(self) -> None:
        ru_fin = set(UNIVERSE["ru_finance"])
        overlap = ru_fin & TOXIC_SYMBOLS
        assert overlap == set(), f"ru_finance contains toxic symbols: {overlap}"


# ------ DividendEntry.status wiring tests ------

UTC = timezone.utc


class TestDividendEntryStatusWiring:
    """All 3 data paths in _setup_dividend_gap_strategy must pass status= to DividendEntry."""

    def test_path1_tinkoff_api_passes_status(self) -> None:
        """Path 1: Tinkoff API should create DividendEntry with status='paid'."""
        from scripts.run_iteration import _setup_dividend_gap_strategy  # type: ignore[import-untyped]

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_dividends.return_value = [
            {"ex_date": datetime(2024, 6, 15, tzinfo=UTC), "amount": 10.0},
        ]

        strategy = _setup_dividend_gap_strategy(
            segment="ru_blue_chips",
            symbols=["SBER"],
            fetcher=mock_fetcher,
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 12, 31, tzinfo=UTC),
        )

        assert strategy is not None
        divs = strategy._calendar["SBER"]
        assert len(divs) == 1
        assert divs[0].status == "paid"

    def test_path2_event_data_passes_status(self) -> None:
        """Path 2: event_data JSON should pass status from entry or default 'paid'."""
        from scripts.run_iteration import _setup_dividend_gap_strategy  # type: ignore[import-untyped]

        mock_fetcher = MagicMock(spec=[])  # no fetch_dividends attribute

        event_data = {
            "dividends": {
                "SBER": [
                    {"ex_date": "2024-06-15", "amount": "10.0", "status": "cancelled"},
                    {"ex_date": "2024-09-15", "amount": "5.0"},  # no status -> default "paid"
                ],
            },
        }

        strategy = _setup_dividend_gap_strategy(
            segment="ru_blue_chips",
            symbols=["SBER"],
            fetcher=mock_fetcher,
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 12, 31, tzinfo=UTC),
            event_data=event_data,
        )

        assert strategy is not None
        divs = strategy._calendar["SBER"]
        assert len(divs) == 2
        assert divs[0].status == "cancelled"
        assert divs[1].status == "paid"

    def test_path3_static_yaml_passes_status(self) -> None:
        """Path 3: static YAML should pass status from entry or default 'paid'."""
        from scripts.run_iteration import _setup_dividend_gap_strategy  # type: ignore[import-untyped]

        mock_fetcher = MagicMock(spec=[])  # no fetch_dividends

        yaml_content = {
            "SBER": [
                {"ex_date": "2024-06-14", "amount": 10.0, "status": "reduced"},
                {"ex_date": "2024-09-14", "amount": 5.0},  # no status -> default "paid"
            ],
        }

        with patch("scripts.run_iteration.yaml") as mock_yaml, \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "open", MagicMock()):
            mock_yaml.safe_load.return_value = yaml_content

            strategy = _setup_dividend_gap_strategy(
                segment="ru_blue_chips",
                symbols=["SBER"],
                fetcher=mock_fetcher,
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 12, 31, tzinfo=UTC),
            )

        assert strategy is not None
        divs = strategy._calendar["SBER"]
        assert len(divs) == 2
        assert divs[0].status == "reduced"
        assert divs[1].status == "paid"

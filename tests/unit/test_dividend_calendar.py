"""Tests for DividendEntry status field and MOEX dividend calendar validation.

Validates:
- DividendEntry accepts status field (paid/cancelled/reduced)
- DividendGapStrategy skips cancelled dividends in signal generation
- Calendar YAML structure and content (150+ events, 20+ symbols, GAZP cancelled)
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from pathlib import Path

import pytest
import yaml

from finalayze.core.schemas import Candle
from finalayze.strategies.dividend_gap import DividendEntry, DividendGapStrategy

# ── Constants ──────────────────────────────────────────────────────────────

_CALENDAR_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "finalayze"
    / "strategies"
    / "presets"
    / "moex_dividends.yaml"
)

_MIN_SYMBOLS = 20
_MIN_EVENTS = 150
_GAZP_CANCELLED_AMOUNT = 52.53


# ── DividendEntry status field tests ──────────────────────────────────────


class TestDividendEntryStatus:
    """Test DividendEntry dataclass status field."""

    def test_dividend_entry_accepts_status(self) -> None:
        """DividendEntry with status='cancelled' stores status field."""
        entry = DividendEntry(
            ex_date=datetime(2022, 6, 30, tzinfo=UTC),
            amount=52.53,
            status="cancelled",
        )
        assert entry.status == "cancelled"

    def test_dividend_entry_default_status_is_paid(self) -> None:
        """DividendEntry without explicit status defaults to 'paid'."""
        entry = DividendEntry(
            ex_date=datetime(2023, 5, 10, tzinfo=UTC),
            amount=25.0,
        )
        assert entry.status == "paid"

    def test_dividend_entry_status_reduced(self) -> None:
        """DividendEntry accepts status='reduced'."""
        entry = DividendEntry(
            ex_date=datetime(2023, 1, 1, tzinfo=UTC),
            amount=10.0,
            status="reduced",
        )
        assert entry.status == "reduced"


# ── Signal generation with status ─────────────────────────────────────────


def _make_candle(
    date: datetime,
    close: float,
    *,
    symbol: str = "GAZP",
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: int = 1000,
) -> Candle:
    """Helper to create a Candle for testing."""
    return Candle(
        symbol=symbol,
        timestamp=date,
        open=open_ or close,
        high=high or close,
        low=low or close,
        close=close,
        volume=volume,
        market_id="moex",
        timeframe="1d",
    )


class TestDividendGapSkipsCancelled:
    """Test that DividendGapStrategy skips cancelled dividends."""

    def test_dividend_gap_skips_cancelled_dividends(self) -> None:
        """generate_signal returns None for cancelled dividend on ex_date."""
        strategy = DividendGapStrategy(min_gap_pct=3.0)

        ex_date = datetime(2022, 6, 30, tzinfo=UTC)
        entry = DividendEntry(ex_date=ex_date, amount=52.53, status="cancelled")
        strategy.add_dividend("GAZP", entry)

        # Pre-exdiv candle (close=500) and ex-div candle (close=450, gap ~10.5%)
        candles = [
            _make_candle(datetime(2022, 6, 29, tzinfo=UTC), 500.0, symbol="GAZP"),
            _make_candle(ex_date, 450.0, symbol="GAZP"),
        ]

        signal = strategy.generate_signal(
            symbol="GAZP",
            candles=candles,
            segment_id="ru_blue_chips",
        )
        assert signal is None, "Cancelled dividend should not generate a BUY signal"

    def test_dividend_gap_trades_paid_dividends(self) -> None:
        """generate_signal returns BUY for paid dividend with sufficient gap."""
        strategy = DividendGapStrategy(min_gap_pct=3.0)

        ex_date = datetime(2023, 5, 10, tzinfo=UTC)
        entry = DividendEntry(ex_date=ex_date, amount=25.0, status="paid")
        strategy.add_dividend("SBER", entry)

        # Pre-exdiv close=300, dividend=25 -> gap=8.33% > 3% threshold
        candles = [
            _make_candle(datetime(2023, 5, 9, tzinfo=UTC), 300.0, symbol="SBER"),
            _make_candle(ex_date, 275.0, symbol="SBER"),
        ]

        signal = strategy.generate_signal(
            symbol="SBER",
            candles=candles,
            segment_id="ru_blue_chips",
        )
        assert signal is not None, "Paid dividend with sufficient gap should generate a signal"
        assert signal.direction.value == "BUY"

    def test_dividend_gap_skips_reduced_dividends(self) -> None:
        """generate_signal returns None for reduced dividend on ex_date."""
        strategy = DividendGapStrategy(min_gap_pct=3.0)

        ex_date = datetime(2023, 1, 15, tzinfo=UTC)
        entry = DividendEntry(ex_date=ex_date, amount=30.0, status="reduced")
        strategy.add_dividend("TEST", entry)

        candles = [
            _make_candle(datetime(2023, 1, 14, tzinfo=UTC), 200.0, symbol="TEST"),
            _make_candle(ex_date, 170.0, symbol="TEST"),
        ]

        signal = strategy.generate_signal(
            symbol="TEST",
            candles=candles,
            segment_id="ru_blue_chips",
        )
        assert signal is None, "Reduced dividend should not generate a BUY signal"


# ── Calendar YAML validation tests ────────────────────────────────────────


class TestCalendarYAML:
    """Validate moex_dividends.yaml structure and content."""

    @pytest.fixture
    def calendar(self) -> dict[str, list[dict[str, object]]]:
        """Load the dividend calendar YAML."""
        assert _CALENDAR_PATH.exists(), f"Calendar file not found: {_CALENDAR_PATH}"
        data = yaml.safe_load(_CALENDAR_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, dict), "Calendar should be a dict"
        return data

    def test_calendar_yaml_structure(self, calendar: dict[str, list[dict[str, object]]]) -> None:
        """Calendar has 20+ symbols, 150+ events, all entries have required fields."""
        symbol_count = len(calendar)
        total_events = sum(len(entries) for entries in calendar.values())

        assert symbol_count >= _MIN_SYMBOLS, f"Need {_MIN_SYMBOLS}+ symbols, got {symbol_count}"
        assert total_events >= _MIN_EVENTS, f"Need {_MIN_EVENTS}+ events, got {total_events}"

        # Every entry must have ex_date, amount, status
        for symbol, entries in calendar.items():
            for i, entry in enumerate(entries):
                assert "ex_date" in entry, f"{symbol}[{i}] missing ex_date"
                assert "amount" in entry, f"{symbol}[{i}] missing amount"
                assert "status" in entry, f"{symbol}[{i}] missing status"
                assert entry["status"] in {"paid", "cancelled", "reduced"}, (
                    f"{symbol}[{i}] invalid status: {entry['status']}"
                )

    def test_calendar_contains_gazp_cancelled(
        self, calendar: dict[str, list[dict[str, object]]]
    ) -> None:
        """GAZP cancelled dividend (2022, ~52.53 RUB) is present."""
        gazp_entries = calendar.get("GAZP", [])
        assert len(gazp_entries) > 0, "GAZP should have dividend entries"

        cancelled = [
            e
            for e in gazp_entries
            if e.get("status") == "cancelled"
            and abs(float(e["amount"]) - _GAZP_CANCELLED_AMOUNT) < 0.1
        ]
        assert len(cancelled) == 1, (
            f"Expected exactly 1 GAZP cancelled dividend (~{_GAZP_CANCELLED_AMOUNT} RUB), "
            f"found {len(cancelled)}: {cancelled}"
        )

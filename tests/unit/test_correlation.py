"""Tests for correlation-aware position sizing (C.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.risk.correlation import (
    CorrelationStep,
    compute_avg_correlation,
    compute_correlation_matrix,
    count_correlated_positions,
)
from finalayze.risk.position_sizing_pipeline import SizingContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_FOUR_DP = Decimal("0.0001")


def _candle(symbol: str, close: Decimal, day_offset: int = 0) -> Candle:
    """Create a candle with given close price."""
    from datetime import timedelta

    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=_TS + timedelta(days=day_offset),
        open=close,
        high=close + Decimal(1),
        low=close - Decimal(1),
        close=close,
        volume=1000,
    )


def _make_candles(symbol: str, closes: list[Decimal]) -> list[Candle]:
    """Create a list of candles with sequential dates."""
    return [_candle(symbol, c, i) for i, c in enumerate(closes)]


def _sizing_context(
    base: Decimal = Decimal(10000),
    correlation_scale: Decimal = Decimal("0.5"),
) -> SizingContext:
    return SizingContext(
        equity=Decimal(100000),
        base_position=base,
        max_position_pct=Decimal("0.20"),
        min_position_size=Decimal(500),
        asset_vol=Decimal("0.20"),
        target_vol=Decimal("0.15"),
        regime_scale=Decimal("1.0"),
        correlation_scale=correlation_scale,
    )


# ---------------------------------------------------------------------------
# compute_correlation_matrix
# ---------------------------------------------------------------------------


class TestComputeCorrelationMatrix:
    def test_perfectly_correlated(self) -> None:
        """Two symbols with identical returns should have correlation ~1.0."""
        closes = [Decimal(100 + i) for i in range(62)]
        candle_sets = {
            "AAPL": _make_candles("AAPL", closes),
            "MSFT": _make_candles("MSFT", closes),
        }
        corr = compute_correlation_matrix(candle_sets, window=60)
        assert ("AAPL", "MSFT") in corr or ("MSFT", "AAPL") in corr
        key = ("AAPL", "MSFT") if ("AAPL", "MSFT") in corr else ("MSFT", "AAPL")
        assert corr[key] == pytest.approx(1.0, abs=0.01)

    def test_insufficient_data_excluded(self) -> None:
        """Pairs with fewer candles than window should be excluded."""
        candle_sets = {
            "AAPL": _make_candles("AAPL", [Decimal(100 + i) for i in range(62)]),
            "SHORT": _make_candles("SHORT", [Decimal(50 + i) for i in range(10)]),
        }
        corr = compute_correlation_matrix(candle_sets, window=60)
        assert ("AAPL", "SHORT") not in corr
        assert ("SHORT", "AAPL") not in corr

    def test_negatively_correlated(self) -> None:
        """Two symbols with opposite returns should have negative correlation."""
        # Build prices where returns are exactly opposite:
        # UP goes 100, 101, 102, ... (positive returns)
        # DOWN mirrors: 100, 99, 98, ... (negative returns of same magnitude)
        base = Decimal(100)
        up_closes = [base + Decimal(i) for i in range(62)]
        # To get truly opposite returns, DOWN = base^2 / UP (reciprocal returns)
        down_closes = [base * base / c for c in up_closes]
        candle_sets = {
            "DOWN": _make_candles("DOWN", down_closes),
            "UP": _make_candles("UP", up_closes),
        }
        corr = compute_correlation_matrix(candle_sets, window=60)
        key = ("DOWN", "UP") if ("DOWN", "UP") in corr else ("UP", "DOWN")
        assert corr[key] < -0.9  # strongly negative

    def test_empty_candle_sets(self) -> None:
        """Empty input should return empty dict."""
        assert compute_correlation_matrix({}) == {}

    def test_single_symbol(self) -> None:
        """Single symbol has no pairs."""
        candle_sets = {"AAPL": _make_candles("AAPL", [Decimal(100 + i) for i in range(62)])}
        assert compute_correlation_matrix(candle_sets, window=60) == {}


# ---------------------------------------------------------------------------
# compute_avg_correlation
# ---------------------------------------------------------------------------


class TestAvgCorrelation:
    def test_correct_average(self) -> None:
        """Average correlation with 2 open positions."""
        correlations = {
            ("AAPL", "MSFT"): 0.8,
            ("AAPL", "GOOG"): 0.4,
        }
        avg = compute_avg_correlation("AAPL", ["MSFT", "GOOG"], correlations)
        assert avg == pytest.approx(0.6, abs=0.001)

    def test_reverse_key_lookup(self) -> None:
        """Should find correlation even if key order is reversed."""
        correlations = {("MSFT", "AAPL"): 0.7}
        avg = compute_avg_correlation("AAPL", ["MSFT"], correlations)
        assert avg == pytest.approx(0.7, abs=0.001)

    def test_no_open_positions(self) -> None:
        """Zero open positions -> 0 average correlation."""
        avg = compute_avg_correlation("AAPL", [], {})
        assert avg == pytest.approx(0.0)

    def test_missing_pair(self) -> None:
        """Missing pair in correlations dict should be treated as 0."""
        correlations = {("AAPL", "MSFT"): 0.8}
        avg = compute_avg_correlation("AAPL", ["MSFT", "GOOG"], correlations)
        # (0.8 + 0.0) / 2 = 0.4
        assert avg == pytest.approx(0.4, abs=0.001)


# ---------------------------------------------------------------------------
# CorrelationStep
# ---------------------------------------------------------------------------


class TestCorrelationStep:
    def test_high_correlation_reduces_position(self) -> None:
        """High correlation_scale (0.8) -> scale = 1-0.8 = 0.2 -> clamped to 0.3."""
        step = CorrelationStep()
        ctx = _sizing_context(
            base=Decimal(10000),
            correlation_scale=Decimal("0.8"),
        )
        result = step.adjust(Decimal(10000), ctx)
        expected = Decimal(10000) * Decimal("0.30")
        assert result == expected.quantize(_FOUR_DP)

    def test_low_correlation_full_size(self) -> None:
        """Low correlation_scale (0.0) -> scale = 1.0."""
        step = CorrelationStep()
        ctx = _sizing_context(correlation_scale=Decimal("0.0"))
        result = step.adjust(Decimal(10000), ctx)
        assert result == Decimal(10000)

    def test_moderate_correlation(self) -> None:
        """Moderate correlation_scale (0.5) -> scale = 0.5."""
        step = CorrelationStep()
        ctx = _sizing_context(correlation_scale=Decimal("0.5"))
        result = step.adjust(Decimal(10000), ctx)
        expected = Decimal(10000) * Decimal("0.5")
        assert result == expected.quantize(_FOUR_DP)

    def test_bounds_lower(self) -> None:
        """Scale never goes below 0.30 even with correlation_scale > 0.7."""
        step = CorrelationStep()
        ctx = _sizing_context(correlation_scale=Decimal("0.95"))
        result = step.adjust(Decimal(10000), ctx)
        expected = Decimal(10000) * Decimal("0.30")
        assert result == expected.quantize(_FOUR_DP)

    def test_bounds_upper(self) -> None:
        """Scale never goes above 1.0 even with negative correlation_scale."""
        step = CorrelationStep()
        # correlation_scale of -0.5 -> 1 - (-0.5) = 1.5 -> clamped to 1.0
        ctx = _sizing_context(correlation_scale=Decimal("-0.5"))
        result = step.adjust(Decimal(10000), ctx)
        assert result == Decimal(10000)


# ---------------------------------------------------------------------------
# count_correlated_positions
# ---------------------------------------------------------------------------


class TestCountCorrelatedPositions:
    def test_counts_above_threshold(self) -> None:
        """Correctly counts positions with correlation > threshold."""
        correlations = {
            ("AAPL", "MSFT"): 0.8,
            ("AAPL", "GOOG"): 0.9,
            ("AAPL", "AMZN"): 0.5,
        }
        count = count_correlated_positions(
            "AAPL", ["MSFT", "GOOG", "AMZN"], correlations, threshold=0.7
        )
        assert count == 2

    def test_three_max(self) -> None:
        """Returns 3 when exactly 3 positions are above threshold."""
        correlations = {
            ("AAPL", "MSFT"): 0.8,
            ("AAPL", "GOOG"): 0.9,
            ("AAPL", "AMZN"): 0.75,
        }
        count = count_correlated_positions(
            "AAPL", ["MSFT", "GOOG", "AMZN"], correlations, threshold=0.7
        )
        assert count == 3

    def test_none_correlated(self) -> None:
        """Returns 0 when no positions are above threshold."""
        correlations = {
            ("AAPL", "MSFT"): 0.3,
            ("AAPL", "GOOG"): 0.1,
        }
        count = count_correlated_positions("AAPL", ["MSFT", "GOOG"], correlations, threshold=0.7)
        assert count == 0

    def test_empty_positions(self) -> None:
        """Returns 0 when no open positions."""
        count = count_correlated_positions("AAPL", [], {}, threshold=0.7)
        assert count == 0

    def test_reverse_key_lookup(self) -> None:
        """Finds correlation even if key order is reversed."""
        correlations = {("MSFT", "AAPL"): 0.85}
        count = count_correlated_positions("AAPL", ["MSFT"], correlations, threshold=0.7)
        assert count == 1

    def test_missing_pair(self) -> None:
        """Missing pair treated as 0 correlation (not counted)."""
        correlations: dict[tuple[str, str], float] = {}
        count = count_correlated_positions("AAPL", ["MSFT"], correlations, threshold=0.7)
        assert count == 0

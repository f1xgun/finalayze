"""Unit tests for risk management (Layer 4): position sizing, stop-loss, pre-trade checks."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from finalayze.core.schemas import Candle
from finalayze.risk.position_sizer import compute_position_size
from finalayze.risk.pre_trade_check import PreTradeChecker
from finalayze.risk.stop_loss import compute_atr_stop_loss

# ── Constants (ruff PLR2004: no magic numbers) ───────────────────────────

# Position sizer constants
WIN_RATE = 0.6
AVG_WIN_RATIO = Decimal("2.0")
EQUITY = Decimal(100_000)
KELLY_FRACTION = 0.5
MAX_POSITION_PCT = 0.20
ZERO = Decimal(0)

# Expected Kelly computation:
# f* = (0.6 * 2.0 - 0.4) / 2.0 = (1.2 - 0.4) / 2.0 = 0.4
# half_kelly = 0.4 * 0.5 = 0.2
# position = 100000 * 0.2 = 20000 (rounded to 2dp)
EXPECTED_HALF_KELLY_POSITION = Decimal("20000.00")
POSITION_TOLERANCE = Decimal("0.01")

# Capping test constants
HIGH_WIN_RATE = 0.9
HIGH_AVG_WIN_RATIO = Decimal("5.0")
# f* = (0.9 * 5.0 - 0.1) / 5.0 = (4.5 - 0.1) / 5.0 = 0.88
# half_kelly = 0.88 * 0.5 = 0.44
# position = 100000 * 0.44 = 44000, but max = 100000 * 0.20 = 20000
EXPECTED_CAPPED_POSITION = Decimal("20000.0")

# Negative Kelly constants (edge with zero or below)
LOW_WIN_RATE = 0.2
LOW_AVG_WIN_RATIO = Decimal("1.0")
# f* = (0.2 * 1.0 - 0.8) / 1.0 = -0.6 -> negative -> return 0

# Stop-loss constants
ENTRY_PRICE = Decimal("100.00")
ATR_PERIOD = 14
ATR_MULTIPLIER = Decimal("2.0")
CANDLE_RANGE = Decimal("2.0")  # high - low per candle
NUM_CANDLES_SUFFICIENT = 15  # > ATR_PERIOD + 1 (just enough with period=14)
NUM_CANDLES_INSUFFICIENT = 10  # < ATR_PERIOD + 1

# Pre-trade check constants
PORTFOLIO_EQUITY = Decimal(100_000)
AVAILABLE_CASH = Decimal(50_000)
SMALL_ORDER = Decimal(10_000)
LARGE_ORDER = Decimal(25_000)
HUGE_ORDER = Decimal(60_000)
MAX_POSITIONS = 10
POSITION_COUNT_OK = 5
POSITION_COUNT_AT_MAX = 10


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_candles_with_range(
    count: int,
    *,
    base_close: Decimal = Decimal("100.00"),
    candle_range: Decimal = CANDLE_RANGE,
) -> list[Candle]:
    """Create *count* candles with a fixed high-low range for predictable ATR."""
    half_range = candle_range / 2
    return [
        Candle(
            symbol="AAPL",
            market_id="us",
            timeframe="1d",
            timestamp=datetime(2024, 1, 1 + i, 14, 30, tzinfo=UTC),
            open=base_close,
            high=base_close + half_range,
            low=base_close - half_range,
            close=base_close,
            volume=1_000_000,
        )
        for i in range(count)
    ]


# ── PositionSizer ────────────────────────────────────────────────────────


class TestComputePositionSize:
    def test_basic_half_kelly(self) -> None:
        result = compute_position_size(
            win_rate=WIN_RATE,
            avg_win_ratio=AVG_WIN_RATIO,
            equity=EQUITY,
            kelly_fraction=KELLY_FRACTION,
            max_position_pct=MAX_POSITION_PCT,
        )
        assert abs(result - EXPECTED_HALF_KELLY_POSITION) < POSITION_TOLERANCE

    def test_capped_at_max_position(self) -> None:
        result = compute_position_size(
            win_rate=HIGH_WIN_RATE,
            avg_win_ratio=HIGH_AVG_WIN_RATIO,
            equity=EQUITY,
            kelly_fraction=KELLY_FRACTION,
            max_position_pct=MAX_POSITION_PCT,
        )
        assert result == EXPECTED_CAPPED_POSITION

    def test_negative_kelly_returns_zero(self) -> None:
        result = compute_position_size(
            win_rate=LOW_WIN_RATE,
            avg_win_ratio=LOW_AVG_WIN_RATIO,
            equity=EQUITY,
        )
        assert result == ZERO

    def test_zero_equity_returns_zero(self) -> None:
        result = compute_position_size(
            win_rate=WIN_RATE,
            avg_win_ratio=AVG_WIN_RATIO,
            equity=ZERO,
        )
        assert result == ZERO

    def test_zero_avg_win_ratio_returns_zero(self) -> None:
        result = compute_position_size(
            win_rate=WIN_RATE,
            avg_win_ratio=ZERO,
            equity=EQUITY,
        )
        assert result == ZERO


# ── StopLoss ─────────────────────────────────────────────────────────────


class TestComputeAtrStopLoss:
    def test_basic_stop_loss_below_entry(self) -> None:
        candles = _make_candles_with_range(NUM_CANDLES_SUFFICIENT)
        stop = compute_atr_stop_loss(
            entry_price=ENTRY_PRICE,
            candles=candles,
            atr_period=ATR_PERIOD,
            atr_multiplier=ATR_MULTIPLIER,
        )
        assert stop is not None
        assert stop < ENTRY_PRICE

    def test_insufficient_candles_returns_none(self) -> None:
        candles = _make_candles_with_range(NUM_CANDLES_INSUFFICIENT)
        stop = compute_atr_stop_loss(
            entry_price=ENTRY_PRICE,
            candles=candles,
            atr_period=ATR_PERIOD,
            atr_multiplier=ATR_MULTIPLIER,
        )
        assert stop is None

    def test_atr_value_matches_range(self) -> None:
        """With constant range candles, ATR == range, so stop = entry - range * multiplier."""
        candles = _make_candles_with_range(NUM_CANDLES_SUFFICIENT)
        stop = compute_atr_stop_loss(
            entry_price=ENTRY_PRICE,
            candles=candles,
            atr_period=ATR_PERIOD,
            atr_multiplier=ATR_MULTIPLIER,
        )
        # ATR should be 2.0 (constant high-low range), stop = 100 - 2*2 = 96
        expected_stop = ENTRY_PRICE - CANDLE_RANGE * ATR_MULTIPLIER
        assert stop is not None
        assert stop == expected_stop


# ── PreTradeChecker ──────────────────────────────────────────────────────


class TestPreTradeChecker:
    def test_passes_valid_order(self) -> None:
        checker = PreTradeChecker(
            max_position_pct=MAX_POSITION_PCT,
            max_positions_per_market=MAX_POSITIONS,
        )
        result = checker.check(
            order_value=SMALL_ORDER,
            portfolio_equity=PORTFOLIO_EQUITY,
            available_cash=AVAILABLE_CASH,
            open_position_count=POSITION_COUNT_OK,
        )
        assert result.passed is True
        assert result.violations == []

    def test_rejects_too_large_position(self) -> None:
        checker = PreTradeChecker(
            max_position_pct=MAX_POSITION_PCT,
            max_positions_per_market=MAX_POSITIONS,
        )
        result = checker.check(
            order_value=LARGE_ORDER,
            portfolio_equity=PORTFOLIO_EQUITY,
            available_cash=AVAILABLE_CASH,
            open_position_count=POSITION_COUNT_OK,
        )
        assert result.passed is False
        assert len(result.violations) == 1
        assert "Position size" in result.violations[0]

    def test_rejects_insufficient_cash(self) -> None:
        checker = PreTradeChecker(
            max_position_pct=MAX_POSITION_PCT,
            max_positions_per_market=MAX_POSITIONS,
        )
        result = checker.check(
            order_value=HUGE_ORDER,
            portfolio_equity=PORTFOLIO_EQUITY,
            available_cash=AVAILABLE_CASH,
            open_position_count=POSITION_COUNT_OK,
        )
        # HUGE_ORDER (60000) > AVAILABLE_CASH (50000) -> insufficient cash
        # HUGE_ORDER (60000) is 60% of equity -> exceeds max 20%
        assert result.passed is False
        assert len(result.violations) >= 1
        cash_violations = [v for v in result.violations if "cash" in v.lower()]
        assert len(cash_violations) == 1

    def test_rejects_too_many_positions(self) -> None:
        checker = PreTradeChecker(
            max_position_pct=MAX_POSITION_PCT,
            max_positions_per_market=MAX_POSITIONS,
        )
        result = checker.check(
            order_value=SMALL_ORDER,
            portfolio_equity=PORTFOLIO_EQUITY,
            available_cash=AVAILABLE_CASH,
            open_position_count=POSITION_COUNT_AT_MAX,
        )
        assert result.passed is False
        assert len(result.violations) == 1
        assert "Open positions" in result.violations[0]

    def test_multiple_violations(self) -> None:
        checker = PreTradeChecker(
            max_position_pct=MAX_POSITION_PCT,
            max_positions_per_market=MAX_POSITIONS,
        )
        result = checker.check(
            order_value=HUGE_ORDER,
            portfolio_equity=PORTFOLIO_EQUITY,
            available_cash=AVAILABLE_CASH,
            open_position_count=POSITION_COUNT_AT_MAX,
        )
        # Should fail on position size, cash, and position count
        expected_violation_count = 3
        assert result.passed is False
        assert len(result.violations) == expected_violation_count

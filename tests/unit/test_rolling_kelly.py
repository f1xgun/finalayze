"""Unit tests for RollingKelly position sizing estimator (Layer 4)."""

from __future__ import annotations

from decimal import Decimal

from finalayze.risk.kelly import (
    _DEFAULT_FRACTION,
    _DEFAULT_WINDOW,
    _FIXED_FRACTIONAL,
    _MIN_KELLY_BLEND_TRADES,
    _MIN_KELLY_FRACTION,
    _MIN_TRADES_FOR_KELLY,
    RollingKelly,
    TradeRecord,
)

# ── Constants (ruff PLR2004: no magic numbers) ───────────────────────────

GOOD_WIN_PNL = Decimal(150)
GOOD_WIN_PCT = Decimal("0.03")
SMALL_LOSS_PNL = Decimal(-50)
SMALL_LOSS_PCT = Decimal("-0.01")

POOR_WIN_PNL = Decimal(80)
POOR_WIN_PCT = Decimal("0.02")
LARGE_LOSS_PNL = Decimal(-200)
LARGE_LOSS_PCT = Decimal("-0.05")

EXPECTED_DEFAULT_WINDOW = 50
EXPECTED_DEFAULT_FRACTION = 0.25
EXPECTED_MIN_TRADES = 10
EXPECTED_FIXED_FRACTIONAL = Decimal("0.01")
EXPECTED_MIN_KELLY = Decimal("0.01")

CUSTOM_WINDOW = 30
CUSTOM_FRACTION = 0.5
SMALL_WINDOW = 5

TRADE_COUNT_FEW = 5
TRADE_COUNT_ENOUGH = 25
TRADE_COUNT_PURE_KELLY = 55


class TestRollingKellyConstants:
    """Verify module-level constants are sane."""

    def test_default_window(self) -> None:
        assert _DEFAULT_WINDOW == EXPECTED_DEFAULT_WINDOW

    def test_default_fraction(self) -> None:
        assert _DEFAULT_FRACTION == EXPECTED_DEFAULT_FRACTION

    def test_min_trades(self) -> None:
        assert _MIN_TRADES_FOR_KELLY == EXPECTED_MIN_TRADES

    def test_fixed_fractional(self) -> None:
        assert _FIXED_FRACTIONAL == EXPECTED_FIXED_FRACTIONAL

    def test_min_kelly_fraction(self) -> None:
        assert _MIN_KELLY_FRACTION == EXPECTED_MIN_KELLY


class TestTradeRecord:
    """TradeRecord dataclass tests."""

    def test_creation(self) -> None:
        record = TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT)
        assert record.pnl == GOOD_WIN_PNL
        assert record.pnl_pct == GOOD_WIN_PCT


class TestRollingKellyInit:
    """Constructor and defaults."""

    def test_default_parameters(self) -> None:
        kelly = RollingKelly()
        assert kelly.trade_count == 0

    def test_custom_parameters(self) -> None:
        kelly = RollingKelly(window=CUSTOM_WINDOW, fraction=CUSTOM_FRACTION)
        assert kelly.trade_count == 0


class TestFixedFractionalWithFewTrades:
    """Before _MIN_TRADES_FOR_KELLY trades, optimal_fraction returns _FIXED_FRACTIONAL."""

    def test_zero_trades(self) -> None:
        kelly = RollingKelly()
        assert kelly.optimal_fraction() == EXPECTED_FIXED_FRACTIONAL

    def test_fewer_than_threshold(self) -> None:
        kelly = RollingKelly()
        for _ in range(TRADE_COUNT_FEW):
            kelly.update(TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT))
        assert kelly.trade_count == TRADE_COUNT_FEW
        assert kelly.optimal_fraction() == EXPECTED_FIXED_FRACTIONAL

    def test_exactly_at_threshold_minus_one(self) -> None:
        kelly = RollingKelly()
        threshold_minus_one = _MIN_TRADES_FOR_KELLY - 1
        for _ in range(threshold_minus_one):
            kelly.update(TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT))
        assert kelly.optimal_fraction() == EXPECTED_FIXED_FRACTIONAL


class TestKellyWithEnoughWinningTrades:
    """After 20+ trades with good win rate, Kelly fraction > fixed fractional."""

    def test_good_win_rate_returns_above_fixed(self) -> None:
        kelly = RollingKelly()
        # 70% win rate with 3:1 avg_win/avg_loss ratio
        wins = 18
        losses = TRADE_COUNT_ENOUGH - wins  # 7 losses
        for _ in range(wins):
            kelly.update(TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT))
        for _ in range(losses):
            kelly.update(TradeRecord(pnl=SMALL_LOSS_PNL, pnl_pct=SMALL_LOSS_PCT))

        result = kelly.optimal_fraction()
        assert result > EXPECTED_FIXED_FRACTIONAL
        assert isinstance(result, Decimal)


class TestKellyWithPoorWinRate:
    """38% win rate should yield a smaller fraction."""

    def test_poor_win_rate(self) -> None:
        kelly = RollingKelly()
        total = TRADE_COUNT_ENOUGH
        wins = 9  # ~36% win rate
        losses = total - wins
        for _ in range(wins):
            kelly.update(TradeRecord(pnl=POOR_WIN_PNL, pnl_pct=POOR_WIN_PCT))
        for _ in range(losses):
            kelly.update(TradeRecord(pnl=SMALL_LOSS_PNL, pnl_pct=SMALL_LOSS_PCT))

        result = kelly.optimal_fraction()
        # With poor win rate the Kelly fraction should be very small or minimum
        half_kelly_cap = Decimal("0.5")
        assert result < half_kelly_cap


class TestKellyNegativeEdgeReturnsReducedFixed:
    """When edge is negative, returns half of FIXED_FRACTIONAL to allow recovery."""

    def test_negative_edge(self) -> None:
        kelly = RollingKelly()
        total = TRADE_COUNT_ENOUGH
        wins = 5  # 20% win rate
        losses = total - wins
        for _ in range(wins):
            kelly.update(TradeRecord(pnl=POOR_WIN_PNL, pnl_pct=POOR_WIN_PCT))
        for _ in range(losses):
            kelly.update(TradeRecord(pnl=LARGE_LOSS_PNL, pnl_pct=LARGE_LOSS_PCT))

        result = kelly.optimal_fraction()
        assert result == Decimal("0.005")  # Half of FIXED_FRACTIONAL for recovery


class TestRollingWindowEvictsOldTrades:
    """Deque maxlen evicts oldest trades."""

    def test_window_eviction(self) -> None:
        kelly = RollingKelly(window=SMALL_WINDOW)
        # Fill window with losses
        for _ in range(SMALL_WINDOW):
            kelly.update(TradeRecord(pnl=SMALL_LOSS_PNL, pnl_pct=SMALL_LOSS_PCT))
        assert kelly.trade_count == SMALL_WINDOW

        # Add more trades — count stays at window size
        for _ in range(SMALL_WINDOW):
            kelly.update(TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT))
        assert kelly.trade_count == SMALL_WINDOW


class TestAllWinsNoLosses:
    """When all trades are wins, returns fixed fractional (no losses to compute ratio)."""

    def test_all_wins(self) -> None:
        kelly = RollingKelly()
        for _ in range(TRADE_COUNT_ENOUGH):
            kelly.update(TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT))

        assert kelly.optimal_fraction() == EXPECTED_FIXED_FRACTIONAL

    def test_all_losses(self) -> None:
        kelly = RollingKelly()
        for _ in range(TRADE_COUNT_ENOUGH):
            kelly.update(TradeRecord(pnl=SMALL_LOSS_PNL, pnl_pct=SMALL_LOSS_PCT))

        assert kelly.optimal_fraction() == EXPECTED_FIXED_FRACTIONAL


class TestTradeCountProperty:
    """trade_count property tracks correctly."""

    def test_starts_at_zero(self) -> None:
        kelly = RollingKelly()
        assert kelly.trade_count == 0

    def test_increments(self) -> None:
        kelly = RollingKelly()
        kelly.update(TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT))
        assert kelly.trade_count == 1
        kelly.update(TradeRecord(pnl=SMALL_LOSS_PNL, pnl_pct=SMALL_LOSS_PCT))
        expected_count = 2
        assert kelly.trade_count == expected_count

    def test_capped_at_window(self) -> None:
        kelly = RollingKelly(window=SMALL_WINDOW)
        for _ in range(SMALL_WINDOW + SMALL_WINDOW):
            kelly.update(TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT))
        assert kelly.trade_count == SMALL_WINDOW


class TestBreakEvenTradesExcluded:
    """Break-even trades (pnl == 0) are excluded from win/loss classification."""

    def test_break_even_excluded_from_losses(self) -> None:
        """Break-even trades don't inflate the loss count."""
        kelly = RollingKelly()
        # 15 wins, 5 losses, 5 break-even = 25 trades
        wins = 15
        losses = 5
        break_even = 5
        for _ in range(wins):
            kelly.update(TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT))
        for _ in range(losses):
            kelly.update(TradeRecord(pnl=SMALL_LOSS_PNL, pnl_pct=SMALL_LOSS_PCT))
        for _ in range(break_even):
            kelly.update(TradeRecord(pnl=Decimal(0), pnl_pct=Decimal(0)))

        result = kelly.optimal_fraction()
        # Win rate computed on decisive trades only: 15 / (15+5) = 75%
        # Should be > fixed fractional
        assert result > EXPECTED_FIXED_FRACTIONAL

    def test_all_break_even_returns_fixed(self) -> None:
        """If all trades are break-even, returns fixed fractional."""
        kelly = RollingKelly()
        for _ in range(TRADE_COUNT_ENOUGH):
            kelly.update(TradeRecord(pnl=Decimal(0), pnl_pct=Decimal(0)))

        # No wins and no losses → returns fixed fractional
        assert kelly.optimal_fraction() == EXPECTED_FIXED_FRACTIONAL


class TestQuarterKellyDampening:
    """Full Kelly is dampened by the fraction parameter."""

    def test_half_kelly_vs_quarter_kelly(self) -> None:
        """Half-Kelly should produce a larger fraction than quarter-Kelly."""
        # Use >50 trades to bypass blend zone and test pure Kelly
        half_fresh = RollingKelly(fraction=CUSTOM_FRACTION)  # 0.5
        quarter_fresh = RollingKelly()  # default 0.25
        wins = 40
        losses = TRADE_COUNT_PURE_KELLY - wins
        for _ in range(wins):
            trade = TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT)
            half_fresh.update(trade)
            quarter_fresh.update(trade)
        for _ in range(losses):
            trade = TradeRecord(pnl=SMALL_LOSS_PNL, pnl_pct=SMALL_LOSS_PCT)
            half_fresh.update(trade)
            quarter_fresh.update(trade)

        half_result = half_fresh.optimal_fraction()
        quarter_result = quarter_fresh.optimal_fraction()
        assert half_result > quarter_result

    def test_fraction_one_returns_full_kelly(self) -> None:
        """fraction=1.0 gives full Kelly (undampened) with >50 trades."""
        full_kelly = RollingKelly(fraction=1.0)
        quarter_kelly = RollingKelly()  # 0.25

        wins = 40
        losses = TRADE_COUNT_PURE_KELLY - wins
        for _ in range(wins):
            trade = TradeRecord(pnl=GOOD_WIN_PNL, pnl_pct=GOOD_WIN_PCT)
            full_kelly.update(trade)
            quarter_kelly.update(trade)
        for _ in range(losses):
            trade = TradeRecord(pnl=SMALL_LOSS_PNL, pnl_pct=SMALL_LOSS_PCT)
            full_kelly.update(trade)
            quarter_kelly.update(trade)

        full_result = full_kelly.optimal_fraction()
        quarter_result = quarter_kelly.optimal_fraction()
        # Full Kelly should be ~4x quarter Kelly
        ratio = float(full_result) / float(quarter_result)
        expected_ratio = 4.0
        tolerance = 0.1
        assert abs(ratio - expected_ratio) < tolerance

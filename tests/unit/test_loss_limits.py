"""Unit tests for daily/weekly loss limit tracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.risk.loss_limits import LossLimitTracker

# ── Constants ─────────────────────────────────────────────────────────────

INITIAL_EQUITY = Decimal(100_000)
DAILY_LIMIT_PCT = 3.0
WEEKLY_LIMIT_PCT = 5.0
COOLDOWN_DAYS = 1

# Equity levels for test scenarios
EQUITY_AFTER_2PCT_LOSS = Decimal(98_000)
EQUITY_AFTER_3PCT_LOSS = Decimal(97_000)
EQUITY_AFTER_4PCT_LOSS = Decimal(96_000)
EQUITY_AFTER_5PCT_LOSS = Decimal(95_000)
EQUITY_AFTER_6PCT_LOSS = Decimal(94_000)

MONDAY = datetime(2025, 1, 6, tzinfo=UTC)  # A Monday
TUESDAY = MONDAY + timedelta(days=1)
WEDNESDAY = MONDAY + timedelta(days=2)
THURSDAY = MONDAY + timedelta(days=3)
FRIDAY = MONDAY + timedelta(days=4)
NEXT_MONDAY = MONDAY + timedelta(days=7)


class TestLossLimitTracker:
    """Tests for LossLimitTracker."""

    def test_no_halt_within_limits(self) -> None:
        """Trading is allowed when losses are within limits."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # 2% loss is within 3% daily limit
        assert not tracker.is_halted(MONDAY, EQUITY_AFTER_2PCT_LOSS)

    def test_daily_limit_triggers_halt(self) -> None:
        """Trading halts when daily loss exceeds limit."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # 4% loss exceeds 3% daily limit
        assert tracker.is_halted(MONDAY, EQUITY_AFTER_4PCT_LOSS)

    def test_daily_limit_exact_boundary(self) -> None:
        """Trading halts at exactly the daily limit."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # Exactly 3% loss triggers halt
        assert tracker.is_halted(MONDAY, EQUITY_AFTER_3PCT_LOSS)

    def test_cooldown_resumes_next_day(self) -> None:
        """After daily halt, trading resumes after cooldown period."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # Trigger daily halt
        assert tracker.is_halted(MONDAY, EQUITY_AFTER_4PCT_LOSS)

        # Next day with reset equity, no longer halted
        tracker.reset_day(TUESDAY, EQUITY_AFTER_4PCT_LOSS)
        assert not tracker.is_halted(TUESDAY, EQUITY_AFTER_4PCT_LOSS)

    def test_weekly_limit_triggers_halt(self) -> None:
        """Trading halts when cumulative weekly loss exceeds limit."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        # Start of week
        tracker.reset_week(MONDAY, INITIAL_EQUITY)
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # 2% loss Monday (within daily limit)
        assert not tracker.is_halted(MONDAY, EQUITY_AFTER_2PCT_LOSS)

        # Tuesday: another day, equity starts at 98k
        tracker.reset_day(TUESDAY, EQUITY_AFTER_2PCT_LOSS)
        # Drop to 95k = 5% from week start (100k), triggers weekly limit
        assert tracker.is_halted(TUESDAY, EQUITY_AFTER_5PCT_LOSS)

    def test_weekly_reset_clears_weekly_baseline(self) -> None:
        """New week resets the weekly baseline."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_week(MONDAY, INITIAL_EQUITY)
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # 5% weekly loss triggers halt
        assert tracker.is_halted(MONDAY, EQUITY_AFTER_5PCT_LOSS)

        # New week resets
        tracker.reset_week(NEXT_MONDAY, EQUITY_AFTER_5PCT_LOSS)
        tracker.reset_day(NEXT_MONDAY, EQUITY_AFTER_5PCT_LOSS)

        # No longer halted (new baseline is 95k)
        assert not tracker.is_halted(NEXT_MONDAY, EQUITY_AFTER_5PCT_LOSS)

    def test_gains_do_not_trigger_halt(self) -> None:
        """Positive equity changes do not trigger halt."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_week(MONDAY, INITIAL_EQUITY)
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # 10% gain
        gain_equity = Decimal(110_000)
        assert not tracker.is_halted(MONDAY, gain_equity)

    def test_default_parameters(self) -> None:
        """Default parameters work without explicit values."""
        tracker = LossLimitTracker()
        tracker.reset_day(MONDAY, INITIAL_EQUITY)
        tracker.reset_week(MONDAY, INITIAL_EQUITY)

        # Small loss should not trigger halt with defaults
        assert not tracker.is_halted(MONDAY, EQUITY_AFTER_2PCT_LOSS)

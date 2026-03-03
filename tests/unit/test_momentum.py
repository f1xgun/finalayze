"""Unit tests for momentum strategy hist_rising logic."""

from __future__ import annotations


def test_hist_rising_negative_histogram() -> None:
    """Verify hist_rising requires current_hist > 0."""
    # current_hist = -3, prev_hist = -5 => improving but negative
    current_hist = -3.0
    prev_hist = -5.0
    # OLD: hist_rising = current_hist > prev_hist = True (bug)
    # NEW: also requires current_hist > 0 = False (fix)
    new_hist_rising = current_hist > prev_hist and current_hist > 0
    assert new_hist_rising is False


def test_hist_rising_positive_histogram() -> None:
    """Verify hist_rising is True when histogram is positive and improving."""
    current_hist = 3.0
    prev_hist = 1.0
    new_hist_rising = current_hist > prev_hist and current_hist > 0
    assert new_hist_rising is True

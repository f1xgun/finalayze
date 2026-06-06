"""Unit tests for scripts/diagnose_segment_liveness.py (UNIV-01 / D-01).

Covers the pure three-bucket (plus ``alive``) liveness classifier that names the
root-cause bucket for each segment: ``no_symbols`` / ``no_candles`` /
``no_signals`` / ``alive``. The classifier is the diagnose-first deliverable
Waves 2-3 key their fix-or-disable decisions on (no assumptions, measured data).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.diagnose_segment_liveness import (  # noqa: E402
    _BUCKET_ALIVE,
    _BUCKET_NO_CANDLES,
    _BUCKET_NO_SIGNALS,
    _BUCKET_NO_SYMBOLS,
    classify_segment_liveness,
)

from finalayze.markets.liquidity import _MIN_BARS_FOR_LIQUIDITY  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (no magic numbers -- ruff PLR2004)
# ---------------------------------------------------------------------------
_BARS_BELOW = _MIN_BARS_FOR_LIQUIDITY - 1
_BARS_AT = _MIN_BARS_FOR_LIQUIDITY
_BARS_ABOVE = _MIN_BARS_FOR_LIQUIDITY + 10
_SOME_SYMBOLS = 3
_SOME_SIGNALS = 5
_SOME_TRADES = 7


class TestClassifySegmentLiveness:
    def test_no_symbols_when_selector_empty(self) -> None:
        # ru_utilities: selector returns [] (its sole liquid name IRAO is sanctioned).
        bucket = classify_segment_liveness(
            selected_count=0,
            max_bar_count=0,
            trade_count=0,
            signal_count=0,
        )
        assert bucket == _BUCKET_NO_SYMBOLS

    def test_no_candles_when_selected_but_short_history(self) -> None:
        # Symbols selected but every one is below the 60-bar eligibility gate.
        bucket = classify_segment_liveness(
            selected_count=_SOME_SYMBOLS,
            max_bar_count=_BARS_BELOW,
            trade_count=0,
            signal_count=0,
        )
        assert bucket == _BUCKET_NO_CANDLES

    def test_no_signals_when_candles_present_but_zero_trades_and_signals(self) -> None:
        bucket = classify_segment_liveness(
            selected_count=_SOME_SYMBOLS,
            max_bar_count=_BARS_ABOVE,
            trade_count=0,
            signal_count=0,
        )
        assert bucket == _BUCKET_NO_SIGNALS

    def test_alive_when_candles_and_trades(self) -> None:
        bucket = classify_segment_liveness(
            selected_count=_SOME_SYMBOLS,
            max_bar_count=_BARS_ABOVE,
            trade_count=_SOME_TRADES,
            signal_count=_SOME_SIGNALS,
        )
        assert bucket == _BUCKET_ALIVE

    def test_boundary_at_min_bars_with_signals_is_no_signals(self) -> None:
        # max_bar_count exactly == 60 with trade_count 0 but signals fired -> no_signals
        # (signals cleared the gate but never became a trade -- a fixable threshold bucket,
        # NOT no_candles).
        bucket = classify_segment_liveness(
            selected_count=_SOME_SYMBOLS,
            max_bar_count=_BARS_AT,
            trade_count=0,
            signal_count=_SOME_SIGNALS,
        )
        assert bucket == _BUCKET_NO_SIGNALS

"""BaseStrategy wrapper around CBRCalendar for CBR rate-driven signals (Layer 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.cbr_calendar import (
    _DEFAULT_AFFECTED_SYMBOLS,
    generate_cbr_signal,
)

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle, Signal
    from finalayze.strategies.cbr_calendar import CBRCalendar, CBRRateEvent

# ── Constants ────────────────────────────────────────────────────────────────

_STRATEGY_NAME = "cbr_calendar"

_SUPPORTED_SEGMENTS: list[str] = [
    "ru_blue_chips",
    "ru_energy",
    "ru_finance",
    "ru_tech",
]

_MAX_BARS_SINCE_EVENT = 10


class CBRStrategyWrapper(BaseStrategy):
    """Adapt CBRCalendar + generate_cbr_signal into the BaseStrategy interface.

    Scans all registered CBR rate events in the calendar, picks the most recent
    one within ``_MAX_BARS_SINCE_EVENT`` candles, and delegates to
    :func:`generate_cbr_signal` for the actual signal logic.
    """

    def __init__(
        self,
        calendar: CBRCalendar,
        affected_symbols: list[str] | None = None,
    ) -> None:
        self._calendar = calendar
        self._affected_symbols: list[str] = (
            list(affected_symbols)
            if affected_symbols is not None
            else list(_DEFAULT_AFFECTED_SYMBOLS)
        )

    # ── BaseStrategy interface ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return _STRATEGY_NAME

    def supported_segments(self) -> list[str]:
        return list(_SUPPORTED_SEGMENTS)

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,  # noqa: ARG002
        sentiment_score: float = 0.0,  # noqa: ARG002
        has_open_position: bool = False,  # noqa: ARG002
    ) -> Signal | None:
        """Find the most recent CBR event within the lookback window and return its signal."""
        if not candles:
            return None

        best_event: CBRRateEvent | None = None
        best_bars_since: int = _MAX_BARS_SINCE_EVENT + 1

        for event in self._calendar._events.values():
            bars_since = sum(1 for c in candles if c.timestamp.date() > event.date)
            if bars_since <= _MAX_BARS_SINCE_EVENT and bars_since < best_bars_since:
                best_bars_since = bars_since
                best_event = event

        if best_event is None:
            return None

        signals = generate_cbr_signal(
            best_event,
            best_bars_since,
            self._affected_symbols,
        )
        return signals.get(symbol)

    def get_parameters(self, segment_id: str) -> dict[str, object]:  # noqa: ARG002
        return {
            "affected_symbols": list(self._affected_symbols),
            "max_bars_since_event": _MAX_BARS_SINCE_EVENT,
        }

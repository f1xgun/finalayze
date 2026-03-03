"""CBR (Central Bank of Russia) rate calendar and contrarian signal generation (Layer 4).

Russian rate changes are typically 100-200bp (vs Fed's 25bp), creating outsized
impact on bank stocks (SBER, VTBR, SBERP). Surprise hikes trigger an initial
sell-off followed by a contrarian buying opportunity 3-5 days later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from finalayze.core.schemas import Signal, SignalDirection

if TYPE_CHECKING:
    from datetime import date

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_MIN_SURPRISE_BPS = 50
_DEFAULT_AFFECTED_SYMBOLS: list[str] = ["SBER", "VTBR", "SBERP"]
_CONTRARIAN_DELAY_MIN = 3
_CONTRARIAN_DELAY_MAX = 5
_STRATEGY_NAME = "cbr_calendar"
_MARKET_ID = "moex"
_SEGMENT_ID = "ru_finance"

# Confidence scaling: abs(surprise_bps) / _BPS_FULL_CONFIDENCE → capped at 1.0
_BPS_FULL_CONFIDENCE = 200


@dataclass(frozen=True)
class CBRRateEvent:
    """A single CBR rate decision event.

    Attributes:
        date: The date of the rate decision.
        rate_decision: The announced key rate (e.g. 16.0 for 16%).
        expected_rate: The consensus-expected rate before announcement.
        surprise_bps: Actual minus expected rate in basis points
                      (e.g. +100 means 1% surprise hike).
    """

    date: date
    rate_decision: float
    expected_rate: float
    surprise_bps: int


@dataclass
class CBRCalendar:
    """Registry of CBR rate decision events with surprise detection."""

    _events: dict[date, CBRRateEvent] = field(default_factory=dict)

    def add_event(self, event: CBRRateEvent) -> None:
        """Register a CBR rate decision."""
        self._events[event.date] = event

    def get_event_for_date(self, lookup_date: date) -> CBRRateEvent | None:
        """Look up a CBR rate event by date."""
        return self._events.get(lookup_date)

    @staticmethod
    def is_surprise_hike(
        event: CBRRateEvent,
        min_surprise_bps: int = _DEFAULT_MIN_SURPRISE_BPS,
    ) -> bool:
        """Return True if event is a surprise rate hike above the threshold."""
        return event.surprise_bps > min_surprise_bps

    @staticmethod
    def is_surprise_cut(
        event: CBRRateEvent,
        min_surprise_bps: int = _DEFAULT_MIN_SURPRISE_BPS,
    ) -> bool:
        """Return True if event is a surprise rate cut below the negative threshold."""
        return event.surprise_bps < -min_surprise_bps


def generate_cbr_signal(
    event: CBRRateEvent,
    bars_since_event: int,
    affected_symbols: list[str] | None = None,
) -> dict[str, Signal | None]:
    """Generate CBR-driven signals for affected bank stocks.

    Signal logic:
    - Surprise hike, bar 0 → SELL (initial sell-off)
    - Surprise hike, bars 3-5 → BUY (contrarian rebound)
    - Surprise cut, bar 0 → BUY (immediate benefit to banks)
    - No surprise → no signal
    - All other bar counts → no signal

    Args:
        event: The CBR rate decision event.
        bars_since_event: Number of trading bars since the event date.
        affected_symbols: Symbols to generate signals for.
            Defaults to ["SBER", "VTBR", "SBERP"].

    Returns:
        Dict mapping each affected symbol to a Signal or None.
    """
    if affected_symbols is None:
        affected_symbols = list(_DEFAULT_AFFECTED_SYMBOLS)

    abs_surprise = abs(event.surprise_bps)
    confidence = min(1.0, abs_surprise / _BPS_FULL_CONFIDENCE)
    features: dict[str, float] = {
        "surprise_bps": float(event.surprise_bps),
        "rate_decision": event.rate_decision,
        "expected_rate": event.expected_rate,
        "bars_since_event": float(bars_since_event),
    }

    is_hike = CBRCalendar.is_surprise_hike(event)
    is_cut = CBRCalendar.is_surprise_cut(event)

    result: dict[str, Signal | None] = {}

    for sym in affected_symbols:
        signal: Signal | None = None

        if is_hike:
            if bars_since_event == 0:
                signal = _make_signal(
                    sym,
                    SignalDirection.SELL,
                    confidence,
                    features,
                    reasoning=f"CBR surprise hike +{event.surprise_bps}bp → immediate SELL",
                )
            elif _CONTRARIAN_DELAY_MIN <= bars_since_event <= _CONTRARIAN_DELAY_MAX:
                signal = _make_signal(
                    sym,
                    SignalDirection.BUY,
                    confidence,
                    features,
                    reasoning=(
                        f"CBR surprise hike +{event.surprise_bps}bp → "
                        f"contrarian BUY (bar {bars_since_event})"
                    ),
                )
        elif is_cut and bars_since_event == 0:
            signal = _make_signal(
                sym,
                SignalDirection.BUY,
                confidence,
                features,
                reasoning=f"CBR surprise cut {event.surprise_bps}bp → immediate BUY",
            )

        result[sym] = signal

    return result


def _make_signal(
    symbol: str,
    direction: SignalDirection,
    confidence: float,
    features: dict[str, float],
    *,
    reasoning: str,
) -> Signal:
    return Signal(
        strategy_name=_STRATEGY_NAME,
        symbol=symbol,
        market_id=_MARKET_ID,
        segment_id=_SEGMENT_ID,
        direction=direction,
        confidence=confidence,
        features=features,
        reasoning=reasoning,
    )

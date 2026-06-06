"""Chandelier Exit stop-loss computation (Layer 4).

The Chandelier Exit places a trailing stop below the highest high of a
look-back window, offset by a multiple of the Average True Range (ATR).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

# Segment-specific Chandelier multipliers
_CHANDELIER_MULTIPLIERS: dict[str, Decimal] = {
    "us_tech": Decimal("3.0"),
    "us_broad": Decimal("3.0"),
    "us_healthcare": Decimal("3.5"),
    "us_finance": Decimal("2.5"),
    "ru_blue_chips": Decimal("4.0"),
    # RUFIN-02 / D-02: tightened 4.0 -> 3.5. Two independent justifications:
    #   (a) 3.5 matches the existing ru_tech multiplier (see ru_tech entry below) --
    #       so the new value is not a fresh free parameter, it reuses a precedent.
    #   (b) The DIRECTION of the change is derived from ru_finance's own attribution
    #       (the phase67 diagnostic run, scripts/diagnose_ru_finance.py on
    #       results/iterations/phase67-...): avg-loss -1022.83 vs avg-win +566.52
    #       (payoff 0.554) and 23 stop exits summing -25,734.80 -- losers run too
    #       far, so a tighter chandelier cuts the big losers sooner.
    # Stays above the 3.0 us_tech/us_broad floor (anti-curve-fit single bounded
    # step, D-03).
    "ru_finance": Decimal("3.5"),
    "ru_energy": Decimal("4.5"),
    "ru_tech": Decimal("3.5"),
}

_DEFAULT_CHANDELIER_MULTIPLIER = Decimal("3.0")


def get_chandelier_multiplier(segment_id: str) -> Decimal:
    """Return the segment-specific Chandelier multiplier.

    Args:
        segment_id: Market segment identifier (e.g. "us_tech", "ru_energy").

    Returns:
        The Chandelier ATR multiplier for the given segment.
    """
    return _CHANDELIER_MULTIPLIERS.get(segment_id, _DEFAULT_CHANDELIER_MULTIPLIER)


def _compute_atr(candles: list[Candle]) -> Decimal:
    """Compute Average True Range over a list of candles.

    Requires at least 2 candles (first candle is used only for prev_close).
    """
    if len(candles) < 2:  # noqa: PLR2004
        return Decimal(0)

    true_ranges: list[Decimal] = []
    for i in range(1, len(candles)):
        prev_close = candles[i - 1].close
        high = candles[i].high
        low = candles[i].low
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    return sum(true_ranges, Decimal(0)) / Decimal(len(true_ranges))


def compute_chandelier_stop(
    candles: list[Candle],
    atr_period: int = 22,
    multiplier: Decimal = Decimal("3.0"),
) -> Decimal | None:
    """Compute Chandelier stop: highest_high(period) - multiplier * ATR(period).

    This returns a CANDIDATE stop price. The caller must enforce monotonic
    ratcheting: ``new_stop = max(current_stop, candidate_stop)``.

    Args:
        candles: Historical OHLCV candles (oldest first). Must have at least
            ``atr_period`` candles.
        atr_period: Look-back window for highest high and ATR.
        multiplier: ATR multiplier for stop distance.

    Returns:
        The Chandelier stop price, or ``None`` if insufficient data.
    """
    if len(candles) < atr_period:
        return None

    recent = candles[-atr_period:]
    highest_high = max(c.high for c in recent)
    atr = _compute_atr(recent)

    if atr <= 0:
        return None

    stop = highest_high - multiplier * atr
    return max(stop, Decimal(0))

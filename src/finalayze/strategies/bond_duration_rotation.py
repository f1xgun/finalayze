"""Bond duration rotation strategy based on CBR regime (Layer 4).

Rotates OFZ-PD portfolio duration toward regime-appropriate targets
using a rule-based CBR monetary policy regime classifier.

Regime inputs:
- RUONIA 7-day average vs. CBR key rate gap
- Last CBR meeting decision (cut / hold / hike)
- CPI YoY (stagflation override)
"""

from __future__ import annotations

from decimal import Decimal
from enum import IntEnum
from typing import TYPE_CHECKING

from finalayze.core.schemas import Candle, Signal, SignalDirection

if TYPE_CHECKING:
    from datetime import date

# ── Regime classifier ────────────────────────────────────────────────────────

# Must be wider than the RUONIA proxy offset (50bps) to avoid false
# dovish signals when RUONIA is approximated as key_rate - 50bps.
_GAP_HOLD_LEAN_THRESHOLD = Decimal("0.75")
_CPI_STAGFLATION_THRESHOLD = Decimal("8.0")
# When key rate is at or above this level, CBR is in restrictive territory.
# Unless CBR is actively cutting, OFZ-PD should not be bought (force HAWKISH).
_KEY_RATE_RESTRICTIVE = Decimal("15.0")

_STRATEGY_NAME = "bond_duration_rotation"
_MARKET_ID = "moex"
_SEGMENT_ID = "ru_ofz_pd"

# Base confidence values
_BASE_CONFIDENCE_BUY = 0.60
_BASE_CONFIDENCE_SELL = 0.65
_BASE_CONFIDENCE_HAWKISH_EXIT = 0.75


class CBRRegime(IntEnum):
    """CBR monetary policy regime. Higher = more hawkish."""

    DOVISH = 0  # target duration 4.0-5.0Y
    NEUTRAL = 1  # target duration 2.5-3.5Y
    HAWKISH = 2  # target duration 0-1.5Y (shift to floaters)


def classify_regime(
    key_rate: Decimal,
    ruonia_7d_avg: Decimal,
    cpi_yoy_latest_published: Decimal,
    last_cbr_decision: str,  # "cut", "hold", "hike"
) -> CBRRegime:
    """Rule-based CBR regime classifier.

    Primary signal: last CBR meeting decision.
    - last_cbr_decision == "hike" -> HAWKISH
    - last_cbr_decision == "cut"  -> DOVISH
    - last_cbr_decision == "hold" -> use RUONIA-key rate gap as tiebreaker:
        - gap < -0.30  -> DOVISH  (markets pricing in cut)
        - gap > +0.30  -> HAWKISH (markets pricing in hike)
        - otherwise    -> NEUTRAL

    CPI override: If CPI > 8% YoY, force at least NEUTRAL (never DOVISH).

    Key-rate restrictive override: If key_rate >= 15% and CBR is not cutting,
    force HAWKISH. At restrictive rates OFZ-PD lose value; only a confirmed
    easing cycle justifies buying fixed-coupon bonds.
    """
    if last_cbr_decision == "hike":
        regime = CBRRegime.HAWKISH
    elif last_cbr_decision == "cut":
        regime = CBRRegime.DOVISH
    else:
        # "hold" — use RUONIA gap as tiebreaker
        gap = ruonia_7d_avg - key_rate
        if gap < -_GAP_HOLD_LEAN_THRESHOLD:
            regime = CBRRegime.DOVISH
        elif gap > _GAP_HOLD_LEAN_THRESHOLD:
            regime = CBRRegime.HAWKISH
        else:
            regime = CBRRegime.NEUTRAL

    # CPI stagflation override
    if cpi_yoy_latest_published > _CPI_STAGFLATION_THRESHOLD:
        regime = max(regime, CBRRegime.NEUTRAL)

    # Key-rate restrictive override: at high rates, only an active easing
    # cycle (cut) justifies holding fixed-coupon OFZ-PD.
    if key_rate >= _KEY_RATE_RESTRICTIVE and last_cbr_decision != "cut":
        regime = max(regime, CBRRegime.HAWKISH)

    return regime


# ── Duration target mapping ──────────────────────────────────────────────────

# Which bonds are appropriate for each regime
_REGIME_BONDS: dict[CBRRegime, list[str]] = {
    CBRRegime.DOVISH: ["SU26246RMFS7", "SU26252RMFS5", "SU26244RMFS2", "SU26243RMFS4"],
    CBRRegime.NEUTRAL: ["SU26241RMFS8", "SU26239RMFS2", "SU26252RMFS5"],
    CBRRegime.HAWKISH: [],  # Exit all PD bonds
}

_REGIME_DURATION_TARGET: dict[CBRRegime, tuple[Decimal, Decimal]] = {
    CBRRegime.DOVISH: (Decimal("4.0"), Decimal("5.0")),
    CBRRegime.NEUTRAL: (Decimal("2.5"), Decimal("3.5")),
    CBRRegime.HAWKISH: (Decimal(0), Decimal("1.5")),
}

_REGIME_LABELS: dict[CBRRegime, str] = {
    CBRRegime.DOVISH: "DOVISH",
    CBRRegime.NEUTRAL: "NEUTRAL",
    CBRRegime.HAWKISH: "HAWKISH",
}


# ── Strategy ─────────────────────────────────────────────────────────────────


class BondDurationRotationStrategy:
    """Strategic layer: duration rotation based on CBR regime.

    Uses rule-based regime classifier (RUONIA gap + CBR decision + CPI).
    Rotates OFZ-PD portfolio duration toward regime-appropriate targets.
    """

    def __init__(
        self,
        bond_durations: dict[str, Decimal],  # symbol -> estimated mod duration
        bond_maturities: dict[str, date],
        coupon_rates: dict[str, Decimal],
        face_value: Decimal = Decimal(1000),
    ) -> None:
        self._bond_durations = bond_durations
        self._bond_maturities = bond_maturities
        self._coupon_rates = coupon_rates
        self._face_value = face_value

    @property
    def name(self) -> str:
        return _STRATEGY_NAME

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        open_positions: dict[str, Decimal],
        bar_idx: int,  # noqa: ARG002
        *,
        key_rate: Decimal | None = None,
        ruonia_7d_avg: Decimal | None = None,
        cpi_yoy: Decimal | None = None,
        last_cbr_decision: str | None = None,
    ) -> Signal | None:
        """Generate duration rotation signal.

        Extra kwargs provide regime data. If any is None, use NEUTRAL regime.
        """
        # Unknown bond => no signal
        if symbol not in self._bond_durations:
            return None

        # Classify regime (fall back to NEUTRAL if data missing)
        regime = self._classify(
            key_rate=key_rate,
            ruonia_7d_avg=ruonia_7d_avg,
            cpi_yoy=cpi_yoy,
            last_cbr_decision=last_cbr_decision,
        )

        is_held = symbol in open_positions and open_positions[symbol] > 0
        decision = self._decide(symbol, regime, is_held=is_held)

        if decision is None:
            return None

        direction, confidence, reasoning = decision
        return self._make_signal(
            symbol=symbol,
            candles=candles,
            direction=direction,
            confidence=confidence,
            regime=regime,
            reasoning=reasoning,
        )

    def _decide(
        self,
        symbol: str,
        regime: CBRRegime,
        *,
        is_held: bool,
    ) -> tuple[SignalDirection, float, str] | None:
        """Determine direction, confidence, and reasoning for a bond in a regime.

        Returns None when no action is needed.
        """
        is_in_regime_list = symbol in _REGIME_BONDS[regime]

        # HAWKISH => SELL everything held, no BUY
        if regime == CBRRegime.HAWKISH:
            if is_held:
                return (
                    SignalDirection.SELL,
                    _BASE_CONFIDENCE_HAWKISH_EXIT,
                    (
                        f"HAWKISH regime: exit all OFZ-PD. "
                        f"Sell {symbol} (duration={self._bond_durations[symbol]}Y)"
                    ),
                )
            return None

        duration = self._bond_durations[symbol]
        dur_low, dur_high = _REGIME_DURATION_TARGET[regime]
        label = _REGIME_LABELS[regime]

        if is_in_regime_list and not is_held:
            return (
                SignalDirection.BUY,
                _BASE_CONFIDENCE_BUY,
                (
                    f"{label} regime: BUY {symbol} "
                    f"(duration={duration}Y, target={dur_low}-{dur_high}Y)"
                ),
            )

        if not is_in_regime_list and is_held:
            return (
                SignalDirection.SELL,
                _BASE_CONFIDENCE_SELL,
                (
                    f"{label} regime: SELL {symbol} "
                    f"(duration={duration}Y, outside target={dur_low}-{dur_high}Y)"
                ),
            )

        # In regime list and held => no action; not in list and not held => no action
        return None

    def _classify(
        self,
        *,
        key_rate: Decimal | None,
        ruonia_7d_avg: Decimal | None,
        cpi_yoy: Decimal | None,
        last_cbr_decision: str | None,
    ) -> CBRRegime:
        """Classify regime, defaulting to NEUTRAL if data is missing."""
        if (
            key_rate is None
            or ruonia_7d_avg is None
            or cpi_yoy is None
            or last_cbr_decision is None
        ):
            return CBRRegime.NEUTRAL

        return classify_regime(
            key_rate=key_rate,
            ruonia_7d_avg=ruonia_7d_avg,
            cpi_yoy_latest_published=cpi_yoy,
            last_cbr_decision=last_cbr_decision,
        )

    def _make_signal(
        self,
        *,
        symbol: str,
        candles: list[Candle],
        direction: SignalDirection,
        confidence: float,
        regime: CBRRegime,
        reasoning: str,
    ) -> Signal:
        """Build a Signal with bond-specific fields."""
        market_id = candles[0].market_id if candles else _MARKET_ID
        duration = float(self._bond_durations.get(symbol, Decimal(0)))
        dur_low, dur_high = _REGIME_DURATION_TARGET[regime]

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=_SEGMENT_ID,
            direction=direction,
            confidence=confidence,
            instrument_type="bond",
            strategy_payload={
                "regime": float(regime),
                "duration": duration,
                "target_duration_low": float(dur_low),
                "target_duration_high": float(dur_high),
            },
            reasoning=reasoning,
        )

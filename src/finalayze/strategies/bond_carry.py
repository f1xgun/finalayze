"""Bond carry strategy for OFZ-PK floaters (Core layer).

Buy-and-hold with maturity ladder rebalancing. Expected return
tracks RUONIA + spread (~1.3-1.6% above RUONIA).

Respects macro regime context via ``last_cbr_decision`` kwarg:
- ``"hike"``: skip rebalancing BUYs (don't add fixed-coupon during hiking)
- ``"hold"``: normal rebalancing
- ``"cut"``: normal rebalancing with slightly higher confidence (0.85)
Maturity SELL signals always fire regardless of regime.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from finalayze.core.schemas import Candle, Signal, SignalDirection

if TYPE_CHECKING:
    from datetime import date

logger = logging.getLogger(__name__)

# Months before maturity to start rotating out
_MATURITY_ROTATION_MONTHS = 6
# Quarterly rebalance (every ~63 trading days)
_REBALANCE_INTERVAL_BARS = 63
# Average days per month used for months-to-maturity calculation
_DAYS_PER_MONTH = 30.44

# BUY confidence per CBR regime
_CONFIDENCE_BUY_DEFAULT = 0.8
_CONFIDENCE_BUY_CUT = 0.85


class BondCarryStrategy:
    """Core layer: OFZ-PK floater carry strategy.

    Simple maturity ladder with quarterly rebalancing and maturity rotation.
    Not a subclass of BaseStrategy -- bond strategies have a different interface.

    Used as a strategy_fn for BondBacktestEngine:
        strategy_fn = carry_strategy.generate_signal
    """

    def __init__(
        self,
        symbols: list[str],
        maturity_dates: dict[str, date],
        rebalance_interval: int = _REBALANCE_INTERVAL_BARS,
    ) -> None:
        self._symbols = symbols
        self._maturity_dates = maturity_dates
        self._rebalance_interval = rebalance_interval
        self._last_rebalance_bar = -rebalance_interval  # trigger on first bar

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        open_positions: dict[str, Any],
        bar_idx: int,
        **kwargs: Any,
    ) -> Signal | None:
        """Generate signal for a single OFZ-PK bond.

        Args:
            symbol: Bond ticker.
            candles: Candles up to current bar.
            open_positions: Currently open positions (all symbols).
            bar_idx: Current bar index.
            **kwargs: Macro context forwarded by BondCycleProcessor.
                Recognised keys: ``last_cbr_decision`` (``"hike"``/``"hold"``/``"cut"``),
                ``key_rate``, ``ruonia_7d_avg``, ``cpi_yoy``.

        Returns:
            Signal or None.
        """
        if not candles:
            return None

        last_candle = candles[-1]
        current_date = (
            last_candle.timestamp.date()
            if hasattr(last_candle.timestamp, "date")
            else last_candle.timestamp
        )

        maturity = self._maturity_dates.get(symbol)
        months_to_maturity = self._months_to_maturity(current_date, maturity)

        # Check if this bond is approaching maturity and we hold it -> SELL
        # Maturity SELL fires regardless of macro regime
        if (
            months_to_maturity is not None
            and months_to_maturity < _MATURITY_ROTATION_MONTHS
            and symbol in open_positions
        ):
            return Signal(
                strategy_name="bond_carry",
                symbol=symbol,
                market_id="moex",
                segment_id="ru_ofz_pk",
                direction=SignalDirection.SELL,
                confidence=0.9,
                strategy_payload={"months_to_maturity": float(months_to_maturity)},
                reasoning=(
                    f"Maturity rotation: {symbol} matures in {months_to_maturity:.1f} months"
                ),
                instrument_type="bond",
            )

        # Extract macro regime from kwargs
        last_cbr_decision: str | None = kwargs.get("last_cbr_decision")

        # Check if it is rebalance time
        is_rebalance = (bar_idx - self._last_rebalance_bar) >= self._rebalance_interval
        should_buy = is_rebalance or not open_positions

        # On rebalance (or empty portfolio): buy underweight bonds not near maturity
        if (
            should_buy
            and symbol not in open_positions
            and months_to_maturity is not None
            and months_to_maturity >= _MATURITY_ROTATION_MONTHS
        ):
            # Gate BUY on macro regime
            if last_cbr_decision == "hike":
                logger.info(
                    "Hiking regime: skip rebalance for %s (bar=%d)",
                    symbol,
                    bar_idx,
                )
                return None

            if is_rebalance:
                self._last_rebalance_bar = bar_idx

            # Determine confidence based on regime
            confidence = (
                _CONFIDENCE_BUY_CUT if last_cbr_decision == "cut" else _CONFIDENCE_BUY_DEFAULT
            )

            regime_label = last_cbr_decision or "unknown"
            return Signal(
                strategy_name="bond_carry",
                symbol=symbol,
                market_id="moex",
                segment_id="ru_ofz_pk",
                direction=SignalDirection.BUY,
                confidence=confidence,
                strategy_payload={"rebalance": 1.0, "bar_idx": float(bar_idx)},
                reasoning=f"Maturity ladder: adding {symbol} (regime: {regime_label})",
                instrument_type="bond",
            )

        return None

    @staticmethod
    def _months_to_maturity(
        current_date: date,
        maturity: date | None,
    ) -> float | None:
        """Calculate months remaining until maturity."""
        if maturity is None:
            return None
        return (maturity - current_date).days / _DAYS_PER_MONTH

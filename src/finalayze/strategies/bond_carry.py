"""Bond carry strategy for OFZ-PK floaters (Core layer).

Buy-and-hold with maturity ladder rebalancing. Expected return
tracks RUONIA + spread (~1.3-1.6% above RUONIA).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from finalayze.core.schemas import Candle, Signal, SignalDirection

if TYPE_CHECKING:
    from datetime import date

# Months before maturity to start rotating out
_MATURITY_ROTATION_MONTHS = 6
# Quarterly rebalance (every ~63 trading days)
_REBALANCE_INTERVAL_BARS = 63
# Average days per month used for months-to-maturity calculation
_DAYS_PER_MONTH = 30.44


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
        **kwargs: Any,  # noqa: ARG002 — macro kwargs forwarded by engine
    ) -> Signal | None:
        """Generate signal for a single OFZ-PK bond.

        Args:
            symbol: Bond ticker.
            candles: Candles up to current bar.
            open_positions: Currently open positions (all symbols).
            bar_idx: Current bar index.
            **kwargs: Accepted for protocol compatibility (macro data).
                Carry strategy does not use macro context.

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
                features={"months_to_maturity": float(months_to_maturity)},
                reasoning=(
                    f"Maturity rotation: {symbol} matures in {months_to_maturity:.1f} months"
                ),
                instrument_type="bond",
            )

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
            if is_rebalance:
                self._last_rebalance_bar = bar_idx
            return Signal(
                strategy_name="bond_carry",
                symbol=symbol,
                market_id="moex",
                segment_id="ru_ofz_pk",
                direction=SignalDirection.BUY,
                confidence=0.8,
                features={"rebalance": 1.0, "bar_idx": float(bar_idx)},
                reasoning=f"Maturity ladder: adding {symbol}",
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

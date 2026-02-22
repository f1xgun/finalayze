"""Currency conversion utilities (Layer 2).

Phase 1: USD-only stub with a hardcoded fallback rate.
Phase 2: Replace with real-time rates from Tinkoff/CBR API.
"""

from __future__ import annotations

from decimal import Decimal

# Supported currency pairs
type CurrencyPair = str  # e.g. "USDRUB", "RUBUSD"

# Hardcoded fallback rate for Phase 1 (updated manually)
_FALLBACK_USDRUB = Decimal("90.0")

# Currency pairs are exactly 6 characters (3-char base + 3-char quote, e.g. "USDRUB")
_CURRENCY_PAIR_LENGTH = 6


class CurrencyConverter:
    """Converts amounts between currencies.

    Phase 1 supports only USD (no conversion needed).
    Phase 2 will add real-time USD/RUB from Tinkoff API.
    """

    def __init__(self, base_currency: str = "USD") -> None:
        self._base = base_currency
        self._rates: dict[str, Decimal] = {
            "USDRUB": _FALLBACK_USDRUB,
            "RUBUSD": Decimal(1) / _FALLBACK_USDRUB,
            "USDUSD": Decimal(1),
            "RUBRUB": Decimal(1),
        }

    def convert(self, amount: Decimal, from_currency: str, to_currency: str) -> Decimal:
        """Convert amount from one currency to another.

        Raises ValueError for unsupported currency pairs.
        """
        if from_currency == to_currency:
            return amount
        pair = f"{from_currency}{to_currency}"
        if pair not in self._rates:
            msg = f"Unsupported currency pair: {pair}"
            raise ValueError(msg)
        return amount * self._rates[pair]

    def set_rate(self, pair: str, rate: Decimal) -> None:
        """Update a conversion rate (e.g. from a live feed).

        ``pair`` must be exactly 6 characters (e.g. ``"USDRUB"``).
        ``rate`` must be positive.

        Note: always call ``set_rate`` for the *canonical* direction of a pair
        (e.g. ``"USDRUB"``).  The inverse pair is always derived from it.
        If you independently call ``set_rate`` for *both* directions of the same
        pair the rates will become inconsistent, because the second call will
        overwrite the inverse that was computed by the first.
        """
        if len(pair) != _CURRENCY_PAIR_LENGTH:
            msg = (
                f"Currency pair must be {_CURRENCY_PAIR_LENGTH} characters "
                f"(e.g. 'USDRUB'), got '{pair}' ({len(pair)} chars)"
            )
            raise ValueError(msg)
        if rate <= Decimal(0):
            msg = f"Rate must be positive, got {rate}"
            raise ValueError(msg)
        self._rates[pair] = rate
        # Also update the inverse so both directions stay consistent.
        # Only set the inverse here; never call set_rate for both directions
        # independently (see docstring).
        reverse_pair = pair[3:] + pair[:3]
        self._rates[reverse_pair] = Decimal(1) / rate

    def to_base(self, amount: Decimal, from_currency: str) -> Decimal:
        """Convert amount to the base currency."""
        return self.convert(amount, from_currency, self._base)

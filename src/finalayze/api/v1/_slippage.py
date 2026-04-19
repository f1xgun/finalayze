"""Slippage computation helper (TCA convention, D-08 sign rule).

Single source of truth for converting a filled price + signal-time reference
price into basis-point slippage. Consumed by Plan 03 (TRAD-01 read path in
``list_trades`` / ``get_trade``) and by the analytics handler
(``avg_slippage_bps``).

Sign convention per D-08 of .planning/phases/55-signals-trades-analytics/55-CONTEXT.md:
positive bps = adverse fill for the trader. BUY filled above reference and
SELL filled below reference both produce positive bps.

Null-fallback per D-07: when the reference price is unavailable (legacy
signals, orders with no ``signal_id``) or non-positive, return ``None`` so
downstream aggregations can exclude the row rather than mislead.
"""

from __future__ import annotations

from decimal import Decimal

_BPS = Decimal(10000)


def compute_slippage_bps(
    fill_price: Decimal,
    reference_price: Decimal | None,
    side: str,
) -> float | None:
    """Compute execution slippage in basis points.

    Args:
        fill_price: Actual executed price (Decimal, from OrderModel.filled_avg_price).
        reference_price: SignalModel.signal_price at signal time (may be None).
        side: "BUY" or "SELL" (case-insensitive).

    Returns:
        bps where positive = adverse for trader (D-08). None when reference is
        None or <= 0 (D-07 null fallback / div-by-zero guard).
    """
    if reference_price is None or reference_price <= 0:
        return None
    diff = fill_price - reference_price
    bps = (diff / reference_price) * _BPS
    if side.upper() == "SELL":
        bps = -bps
    return float(bps)

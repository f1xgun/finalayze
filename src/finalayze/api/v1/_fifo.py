"""FIFO BUY/SELL pair accounting for trade analytics (D-01, D-04).

Shared by Plan 03 (TRAD-02 /trades/analytics) and Plan 04 (SIGP-01
/strategies/performance). Located in ``api/v1/`` rather than its own helpers
package to keep wave-3 dependency surface minimal — pure helper, no DB / ORM
imports. Orders are duck-typed on the fields named in the docstring so the
helper works equally for SQLAlchemy ``OrderModel`` rows and synthetic
``SimpleNamespace`` fixtures in unit tests.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Iterator
    from datetime import datetime

_log = structlog.get_logger()


@dataclass(frozen=True)
class PairedTrade:
    """A closed round-trip trade — one BUY matched against one SELL for some qty.

    Partial fills produce multiple PairedTrade instances. Shorts (SELL without
    prior BUY residual) are never emitted — logged and skipped per Pitfall 3.

    Fields:
        symbol: Market symbol (e.g., "SBER").
        entry_price: Buy fill price (Decimal, from OrderModel.filled_avg_price).
        exit_price: Sell fill price (Decimal).
        quantity: Matched share count (Decimal). Equals min(buy_remaining, sell_remaining).
        entry_ts: Timestamp of the BUY fill.
        exit_ts: Timestamp of the SELL fill.
        closing_signal_id: signal_id on the closing (SELL) order — D-04 strategy attribution.
        entry_signal_id: signal_id on the opening (BUY) order.
    """

    symbol: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    entry_ts: datetime
    exit_ts: datetime
    closing_signal_id: uuid.UUID | None
    entry_signal_id: uuid.UUID | None


def fifo_pair(orders: Iterable[Any]) -> Iterator[PairedTrade]:
    """Pair filled BUYs with subsequent SELLs per-symbol, FIFO.

    Orders must expose: ``status``, ``side``, ``symbol``, ``filled_quantity``
    (Decimal), ``filled_avg_price`` (Decimal), ``filled_at`` (datetime),
    ``signal_id`` (UUID | None). Caller is responsible for ordering — sort
    ascending by (symbol, filled_at) before passing in.

    Non-filled orders, zero/negative quantities, and null prices are skipped.
    Partial fills are split by quantity (D-01). SELL without matching BUY
    residual is logged (``fifo_sell_without_open_buy`` /
    ``fifo_sell_residual_unmatched``) and skipped — shorts are out of scope.
    Remaining-quantity bookkeeping is keyed on ``id(order)`` so we never
    mutate the caller's ORM rows.
    """
    remaining: dict[int, Decimal] = {}
    stacks: dict[str, deque[Any]] = {}

    for o in orders:
        if o.status != "filled":
            continue
        if o.filled_quantity is None or o.filled_quantity <= 0:
            continue
        if o.filled_avg_price is None:
            continue
        stack = stacks.setdefault(o.symbol, deque())
        if o.side.upper() == "BUY":
            remaining[id(o)] = Decimal(o.filled_quantity)
            stack.append(o)
            continue

        # SELL path: match against earliest open BUY(s) until quantity exhausted.
        to_close = Decimal(o.filled_quantity)
        if not stack:
            _log.warning(
                "fifo_sell_without_open_buy",
                symbol=o.symbol,
                sell_qty=float(to_close),
            )
            continue
        while to_close > 0 and stack:
            buy = stack[0]
            avail = remaining[id(buy)]
            matched = min(to_close, avail)
            yield PairedTrade(
                symbol=o.symbol,
                entry_price=Decimal(buy.filled_avg_price),
                exit_price=Decimal(o.filled_avg_price),
                quantity=matched,
                entry_ts=buy.filled_at,
                exit_ts=o.filled_at,
                closing_signal_id=o.signal_id,
                entry_signal_id=buy.signal_id,
            )
            to_close -= matched
            remaining[id(buy)] = avail - matched
            if remaining[id(buy)] <= 0:
                stack.popleft()
                del remaining[id(buy)]
        if to_close > 0:
            _log.warning(
                "fifo_sell_residual_unmatched",
                symbol=o.symbol,
                residual_qty=float(to_close),
            )

"""Tax lots + strict-FIFO realized-result matching (Layer L1, pure Decimal).

Reconstructs per-lot cost basis by replaying an operations history and matches
SELLs against the oldest open lots (strict FIFO, partial-lot close). All money
math is ``decimal.Decimal`` -- never float.

Realized result on a matched fragment:
    realized = proceeds - matched_cost - buy_commission_share - sell_commission

Imports only stdlib -- a true leaf module. See
docs/research/tax_optimization_engine_design.md sections 2.2 and 4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date


class TaxError(Exception):
    """Base error for the tax engine."""


class FifoMatchError(TaxError):
    """Raised when a SELL cannot be matched against known open lots (oversell)."""


class OperationType(StrEnum):
    """Executed operation kinds relevant to the first slice.

    Mirrors the readonly sidecar ``op.type`` tokens (design section 3.1) but
    transliterated to the subset the engine reconstructs.
    """

    BUY = "BUY"
    SELL = "SELL"
    DIVIDEND = "DIVIDEND"
    COUPON = "COUPON"
    TAX = "TAX"
    INPUT_SECURITIES = "INPUT_SECURITIES"


@dataclass(frozen=True)
class Operation:
    """A single executed operation reconstructed from history.

    ``payment`` is the signed cash flow (negative for a BUY outflow, positive for
    a SELL/DIVIDEND/COUPON inflow), matching the sidecar convention. For non-trade
    rows (dividend/coupon/tax) ``quantity`` and ``price_per_unit`` may be ``None``.

    ``cost_basis_known`` is ``False`` for INPUT_SECURITIES (transferred-in with no
    acquisition price/date) -- an honest-degradation flag, never a fabricated
    number (design section 3.2 gap 4).
    """

    op_type: OperationType
    op_date: date
    figi: str
    ticker: str
    payment: Decimal
    currency: str
    quantity: Decimal | None = None
    price_per_unit: Decimal | None = None
    commission: Decimal = Decimal(0)
    cost_basis_known: bool = True
    # ``payment`` for dividends/coupons from the sidecar is already NET of
    # broker withholding -- callers mark such figures as estimates (gap 5).
    payment_is_net_estimate: bool = False


@dataclass(frozen=True)
class TaxLot:
    """An open lot with reconstructed cost basis.

    ``cost_basis_known`` is ``False`` for INPUT_SECURITIES lots (no acquire price/
    date) -- the LDV clock and cost are UNKNOWN and must be flagged, not guessed.
    ``on_iis`` lots are excluded from LDV/harvest (design sections 1.1, 3.2 gap 7).
    """

    figi: str
    ticker: str
    acquire_date: date
    quantity: Decimal
    price_per_unit: Decimal
    commission_buy: Decimal
    currency: str
    russian_issuer: bool = True
    on_iis: bool = False
    cost_basis_known: bool = True

    @property
    def cost_basis(self) -> Decimal:
        """Total acquisition cost of the lot (price * qty + buy commission)."""
        return self.price_per_unit * self.quantity + self.commission_buy


@dataclass(frozen=True)
class RealizedResult:
    """One realized fragment from matching a SELL against one open lot.

    A single SELL may produce several fragments (one per matched lot). ``realized``
    is the signed capital result in RUB: positive = gain, negative = loss.
    """

    figi: str
    ticker: str
    acquire_date: date
    dispose_date: date
    quantity: Decimal
    proceeds: Decimal
    matched_cost: Decimal
    buy_commission_share: Decimal
    sell_commission_share: Decimal
    currency: str = "RUB"
    russian_issuer: bool = True
    on_iis: bool = False

    @property
    def realized(self) -> Decimal:
        """proceeds - matched_cost - buy_commission_share - sell_commission_share."""
        return (
            self.proceeds
            - self.matched_cost
            - self.buy_commission_share
            - self.sell_commission_share
        )


@dataclass
class _OpenLot:
    """Mutable working lot used during FIFO replay (internal)."""

    figi: str
    ticker: str
    acquire_date: date
    remaining_qty: Decimal
    price_per_unit: Decimal
    commission_buy_remaining: Decimal
    original_qty: Decimal
    currency: str
    russian_issuer: bool = True
    on_iis: bool = False
    cost_basis_known: bool = True


# event-stream ordering: on the same date a BUY settles BEFORE a SELL, so a
# same-day buy-then-sell is matchable, but a SELL dated strictly before a BUY
# never sees that future lot (look-ahead-free, design 2.6 / R-3).
_BUY_SETTLES_FIRST = 0
_SELL_SETTLES_AFTER = 1


def _buy_qty(op: Operation) -> Decimal:
    """BUY quantity, validated positive (a malformed row must never be swallowed)."""
    qty = op.quantity if op.quantity is not None else Decimal(0)
    if qty <= 0:
        msg = (
            f"malformed BUY quantity for {op.ticker} on {op.op_date}: {qty} "
            f"(a BUY must have a strictly positive quantity)"
        )
        raise FifoMatchError(msg)
    return qty


def _sell_qty(op: Operation) -> Decimal:
    """SELL quantity, validated positive (a malformed row must never be swallowed)."""
    qty = op.quantity if op.quantity is not None else Decimal(0)
    if qty <= 0:
        msg = (
            f"malformed SELL quantity for {op.ticker} on {op.op_date}: {qty} "
            f"(a SELL must have a strictly positive quantity)"
        )
        raise FifoMatchError(msg)
    return qty


def _merged_events(
    buys: list[Operation],
    sells: list[Operation],
) -> list[tuple[int, Operation]]:
    """A single chronologically-merged event stream (BUYs settle before SELLs same-day).

    Validates every quantity up front so a malformed BUY/SELL raises rather than
    being silently absorbed (WR-05). Ordering key is ``(date, buy_before_sell)`` so
    a SELL never matches a lot acquired strictly after it (CR-01 / look-ahead-free).
    """
    events: list[tuple[int, Operation]] = []
    for buy in buys:
        _buy_qty(buy)  # validate; raises on non-positive
        events.append((_BUY_SETTLES_FIRST, buy))
    for sell in sells:
        _sell_qty(sell)  # validate; raises on non-positive
        events.append((_SELL_SETTLES_AFTER, sell))
    events.sort(key=lambda ev: (ev[1].op_date, ev[0]))
    return events


def _op_lot(op: Operation) -> _OpenLot:
    qty = op.quantity if op.quantity is not None else Decimal(0)
    price = op.price_per_unit if op.price_per_unit is not None else Decimal(0)
    return _OpenLot(
        figi=op.figi,
        ticker=op.ticker,
        acquire_date=op.op_date,
        remaining_qty=qty,
        price_per_unit=price,
        commission_buy_remaining=op.commission,
        original_qty=qty,
        currency=op.currency,
        cost_basis_known=op.cost_basis_known,
    )


def fifo_match(
    buys: list[Operation],
    sells: list[Operation],
) -> list[RealizedResult]:
    """Match SELLs against the oldest open lots (strict FIFO, partial close).

    ``buys`` are BUY (or INPUT_SECURITIES) operations; ``sells`` are SELLs. Each
    SELL consumes the oldest open lot(s) first, splitting a lot on a partial
    close. The buy commission is apportioned pro-rata to the consumed quantity;
    the sell commission is apportioned pro-rata across the lot fragments a single
    SELL touches (so a SELL's commission is charged exactly once in total).

    Raises ``FifoMatchError`` on an oversell (a SELL with no matching open lot) or
    on a malformed quantity. Events replay in a single chronological stream so a
    SELL can only match lots acquired on/before its date (look-ahead-free, CR-01).
    """
    open_lots: list[_OpenLot] = []
    results: list[RealizedResult] = []

    for kind, op in _merged_events(buys, sells):
        if kind == _BUY_SETTLES_FIRST:
            open_lots.append(_op_lot(op))
            continue
        # kind == SELL: match ONLY against lots opened so far (all op_date <= sell)
        remaining_to_match = _sell_qty(op)
        sell_price = op.price_per_unit if op.price_per_unit is not None else Decimal(0)
        total_sell_qty = remaining_to_match
        while remaining_to_match > 0:
            lot = _oldest_lot_with_qty(open_lots)
            if lot is None:
                msg = (
                    f"oversell: SELL of {op.ticker} on {op.op_date} exceeds known "
                    f"open lots by {remaining_to_match} units"
                )
                raise FifoMatchError(msg)
            take = min(remaining_to_match, lot.remaining_qty)
            proceeds = take * sell_price
            matched_cost = take * lot.price_per_unit
            # buy commission apportioned to the consumed slice of THIS lot
            # (_oldest_lot_with_qty guarantees lot.remaining_qty > 0 here)
            buy_comm_share = lot.commission_buy_remaining * take / lot.remaining_qty
            # sell commission apportioned across the whole SELL by qty
            # (total_sell_qty > 0 guaranteed by _sell_qty validation)
            sell_comm_share = op.commission * take / total_sell_qty
            results.append(
                RealizedResult(
                    figi=lot.figi,
                    ticker=lot.ticker,
                    acquire_date=lot.acquire_date,
                    dispose_date=op.op_date,
                    quantity=take,
                    proceeds=proceeds,
                    matched_cost=matched_cost,
                    buy_commission_share=buy_comm_share,
                    sell_commission_share=sell_comm_share,
                    currency=lot.currency,
                    russian_issuer=lot.russian_issuer,
                    on_iis=lot.on_iis,
                )
            )
            lot.commission_buy_remaining -= buy_comm_share
            lot.remaining_qty -= take
            remaining_to_match -= take

    return results


def _oldest_lot_with_qty(open_lots: list[_OpenLot]) -> _OpenLot | None:
    for lot in open_lots:
        if lot.remaining_qty > 0:
            return lot
    return None


def open_lots_after_match(
    buys: list[Operation],
    sells: list[Operation],
) -> list[TaxLot]:
    """Return the STILL-OPEN lots (with remaining qty) after applying FIFO SELLs.

    Used by the LDV / harvest paths to reason about un-disposed positions. The
    returned ``TaxLot`` quantity/commission reflect the un-consumed remainder.
    Raises ``FifoMatchError`` on oversell or a malformed quantity (same rule as
    ``fifo_match``). Replays a single chronological stream so a SELL never
    consumes a lot acquired strictly after it (CR-01).
    """
    return _consume(buys, sells)


def _to_taxlot(lot: _OpenLot) -> TaxLot:
    return TaxLot(
        figi=lot.figi,
        ticker=lot.ticker,
        acquire_date=lot.acquire_date,
        quantity=lot.remaining_qty,
        price_per_unit=lot.price_per_unit,
        commission_buy=lot.commission_buy_remaining,
        currency=lot.currency,
        russian_issuer=lot.russian_issuer,
        on_iis=lot.on_iis,
        cost_basis_known=lot.cost_basis_known,
    )


def _consume(buys: list[Operation], sells: list[Operation]) -> list[TaxLot]:
    """Replay the merged event stream and return the still-open lots (CR-01).

    A SELL matches only lots opened on/before its date; an oversell or malformed
    quantity raises ``FifoMatchError``.
    """
    working: list[_OpenLot] = []
    for kind, op in _merged_events(buys, sells):
        if kind == _BUY_SETTLES_FIRST:
            working.append(_op_lot(op))
            continue
        remaining_to_match = _sell_qty(op)
        while remaining_to_match > 0:
            lot = _oldest_lot_with_qty(working)
            if lot is None:
                msg = (
                    f"oversell: SELL of {op.ticker} on {op.op_date} exceeds known "
                    f"open lots by {remaining_to_match} units"
                )
                raise FifoMatchError(msg)
            take = min(remaining_to_match, lot.remaining_qty)
            # (_oldest_lot_with_qty guarantees lot.remaining_qty > 0 here)
            buy_comm_share = lot.commission_buy_remaining * take / lot.remaining_qty
            lot.commission_buy_remaining -= buy_comm_share
            lot.remaining_qty -= take
            remaining_to_match -= take
    return [_to_taxlot(lot) for lot in working if lot.remaining_qty > 0]

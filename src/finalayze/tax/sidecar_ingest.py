"""Parse a STATIC readonly ``operations`` JSON fixture into Operation objects.

Layer L2 (stdlib + tax.lots). This module is the design's step-11 "later"
sidecar bridge, built here against a STATIC fixture only. It NEVER runs the CLI,
never uses a token, never touches the network -- it only parses JSON already on
disk / in memory.

Honest degradation (design section 3.2):
- INPUT_SECURITIES rows carry no acquisition price/date -> ``cost_basis_known``
  is False and ``price_per_unit`` stays None (LDV clock + cost UNKNOWN, flagged
  downstream -- never a fabricated number).
- DIVIDEND / COUPON ``payment`` is already NET of broker withholding -> marked
  ``payment_is_net_estimate`` so callers surface it as an estimate.

All money is parsed via ``Decimal(str(x))`` to avoid float representation error.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from finalayze.tax.lots import Operation, OperationType

if TYPE_CHECKING:
    from pathlib import Path

# maps raw sidecar ``operationType`` tokens to engine OperationType
_TYPE_MAP: dict[str, OperationType] = {
    "OPERATION_TYPE_BUY": OperationType.BUY,
    "OPERATION_TYPE_SELL": OperationType.SELL,
    "OPERATION_TYPE_DIVIDEND": OperationType.DIVIDEND,
    "OPERATION_TYPE_COUPON": OperationType.COUPON,
    "OPERATION_TYPE_TAX": OperationType.TAX,
    "OPERATION_TYPE_INPUT_SECURITIES": OperationType.INPUT_SECURITIES,
}

_NET_ESTIMATE_TYPES = frozenset({OperationType.DIVIDEND, OperationType.COUPON})


class SidecarIngestError(Exception):
    """Raised when a readonly operations payload cannot be parsed."""


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _parse_date(raw: str) -> date:
    # ISO-8601 with a trailing 'Z' (UTC); we only keep the calendar date.
    text = raw.replace("Z", "+00:00")
    return datetime.fromisoformat(text).date()


def _parse_op(row: dict[str, Any]) -> Operation:
    raw_type = row.get("operationType", "")
    op_type = _TYPE_MAP.get(raw_type)
    if op_type is None:
        msg = f"unknown operationType: {raw_type!r}"
        raise SidecarIngestError(msg)

    quantity = _to_decimal(row.get("quantity"))
    price = _to_decimal(row.get("price"))
    # IN-04: `... or Decimal(0)` collapses a legitimate parsed Decimal("0") (falsy)
    # to the same value -- no live bug for money, but the idiom would hide a
    # meaningful zero if semantics changed. Use an explicit None check instead.
    commission_parsed = _to_decimal(row.get("commission"))
    commission = commission_parsed if commission_parsed is not None else Decimal(0)
    payment_parsed = _to_decimal(row.get("payment"))
    payment = payment_parsed if payment_parsed is not None else Decimal(0)

    is_input = op_type is OperationType.INPUT_SECURITIES
    return Operation(
        op_type=op_type,
        op_date=_parse_date(str(row["date"])),
        figi=str(row.get("figi", "")),
        ticker=str(row.get("ticker", "")),
        payment=payment,
        currency=str(row.get("currency", "RUB")),
        quantity=quantity,
        # a transferred-in lot has NO known acquisition price -> keep None
        price_per_unit=None if is_input else price,
        commission=commission,
        cost_basis_known=not is_input,
        payment_is_net_estimate=op_type in _NET_ESTIMATE_TYPES,
    )


def parse_operations_json(raw: str) -> list[Operation]:
    """Parse a readonly ``operations`` JSON string into Operation objects.

    Accepts either ``{"operations": [...]}`` or a bare list ``[...]``. Only
    EXECUTED rows are expected (the sidecar already filters on state); no
    filtering is required here.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"invalid operations JSON: {exc}"
        raise SidecarIngestError(msg) from exc

    rows = payload["operations"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        msg = "operations payload must be a list or an object with an 'operations' list"
        raise SidecarIngestError(msg)
    return [_parse_op(row) for row in rows]


def parse_operations_file(path: Path) -> list[Operation]:
    """Parse a STATIC readonly ``operations`` JSON file into Operation objects.

    Reads bytes from disk only -- never runs the CLI, never hits the network.
    """
    return parse_operations_json(path.read_text(encoding="utf-8"))

"""Tests for static readonly-operations JSON ingestion (design section 4.3 step 11).

Parses a STATIC fixture (never runs the CLI, never touches the network):
- INPUT_SECURITIES -> cost_basis_known False (LDV clock + cost unknown flag).
- dividend/coupon net ``payment`` -> payment_is_net_estimate marker.
INVARIANT 4: INPUT_SECURITIES is flagged, never a fabricated number.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from finalayze.tax.lots import OperationType
from finalayze.tax.sidecar_ingest import parse_operations_file, parse_operations_json

FIXTURE = Path(__file__).parent / "fixtures" / "operations_sample.json"

EXPECTED_OP_COUNT = 5
SBER_BUY_QTY = Decimal(1000)
SBER_BUY_PRICE = Decimal(200)
SBER_BUY_COMMISSION = Decimal(100)


def test_parse_file_returns_all_executed_ops() -> None:
    ops = parse_operations_file(FIXTURE)
    assert len(ops) == EXPECTED_OP_COUNT


def test_buy_op_carries_price_qty_commission() -> None:
    ops = parse_operations_file(FIXTURE)
    buy = next(o for o in ops if o.op_type is OperationType.BUY)
    assert buy.quantity == SBER_BUY_QTY
    assert buy.price_per_unit == SBER_BUY_PRICE
    assert buy.commission == SBER_BUY_COMMISSION
    assert buy.cost_basis_known is True
    assert isinstance(buy.payment, Decimal)


def test_input_securities_flagged_cost_basis_unknown() -> None:
    ops = parse_operations_file(FIXTURE)
    transferred = next(o for o in ops if o.op_type is OperationType.INPUT_SECURITIES)
    assert transferred.cost_basis_known is False
    # price is unknown for a transferred-in lot -> not fabricated
    assert transferred.price_per_unit is None


def test_dividend_and_coupon_marked_net_estimate() -> None:
    ops = parse_operations_file(FIXTURE)
    div = next(o for o in ops if o.op_type is OperationType.DIVIDEND)
    coup = next(o for o in ops if o.op_type is OperationType.COUPON)
    assert div.payment_is_net_estimate is True
    assert coup.payment_is_net_estimate is True
    # gross gains (BUY/SELL) are not net-estimate
    buy = next(o for o in ops if o.op_type is OperationType.BUY)
    assert buy.payment_is_net_estimate is False


def test_commission_null_becomes_zero() -> None:
    ops = parse_operations_file(FIXTURE)
    coup = next(o for o in ops if o.op_type is OperationType.COUPON)
    assert coup.commission == Decimal(0)


def test_parse_json_string_matches_file() -> None:
    raw = FIXTURE.read_text(encoding="utf-8")
    ops = parse_operations_json(raw)
    assert len(ops) == EXPECTED_OP_COUNT

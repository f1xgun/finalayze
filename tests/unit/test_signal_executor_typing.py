"""S7.1 — SignalExecutor stage contexts must be typed against BrokerBase.

Audit #17 flagged ``broker: Any`` on ``_SignalContext`` and ``_OrderContext``
inside ``orchestration/signal_executor.py``. Those two dataclasses are the
hand-off carriers between the three pipeline stages (signal → validate →
submit), so an ``Any`` here defeats type-checking across the entire
critical execution path: a mistyped broker reference (e.g. passing the
``BrokerRouter`` instead of the routed leaf broker) compiles and runs.

Canonical type is ``finalayze.execution.broker_base.BrokerBase`` — the
ABC implemented by ``SimulatedBroker``, ``AlpacaBroker``, and
``TinkoffBroker`` and returned by ``BrokerRouter.route()``.

Contract:
  S7.1-01: ``_SignalContext.broker`` is annotated as ``BrokerBase``.
  S7.1-02: ``_OrderContext.broker`` is annotated as ``BrokerBase``.
  S7.1-03: neither dataclass uses ``Any`` for the ``broker`` field.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MODULE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "finalayze"
    / "orchestration"
    / "signal_executor.py"
)


def _broker_annotation(class_name: str) -> str:
    """Return the source-text annotation for the ``broker`` field of *class_name*."""
    tree = ast.parse(_MODULE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "broker"
                ):
                    return ast.unparse(stmt.annotation)
    msg = f"`broker` field not found on class {class_name}"
    raise AssertionError(msg)


def test_signal_context_broker_typed_as_broker_base() -> None:
    assert _broker_annotation("_SignalContext") == "BrokerBase"


def test_order_context_broker_typed_as_broker_base() -> None:
    assert _broker_annotation("_OrderContext") == "BrokerBase"


def test_no_any_on_broker_fields() -> None:
    """Guard against the type sliding back to ``Any`` later."""
    for cls in ("_SignalContext", "_OrderContext"):
        ann = _broker_annotation(cls)
        assert ann != "Any", (
            f"{cls}.broker reverted to Any — re-introduces audit #17. "
            "Use BrokerBase (the abstract broker interface) instead."
        )

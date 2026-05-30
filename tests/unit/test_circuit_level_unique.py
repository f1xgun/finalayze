"""S6.1 — guard against re-introducing a second CircuitLevel enum.

Risk model: two parallel ``CircuitLevel`` definitions silently disagree under
``==``, allowing a layered-portfolio breach to be reported as equity-level
``HALT`` while the equity-side alerter watches for the StrEnum ``HALTED``.

Canonical home is ``finalayze.risk.circuit_breaker`` (StrEnum, four values:
NORMAL/CAUTION/HALTED/LIQUIDATE). Any other ``class CircuitLevel`` definition
inside ``src/finalayze/risk/`` is a regression.
"""

from __future__ import annotations

import ast
from pathlib import Path

_RISK_DIR = Path(__file__).resolve().parents[2] / "src" / "finalayze" / "risk"
_CANONICAL = _RISK_DIR / "circuit_breaker.py"


def _files_defining_circuit_level() -> list[Path]:
    matches: list[Path] = []
    for path in sorted(_RISK_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CircuitLevel":
                matches.append(path)
                break
    return matches


def test_circuit_level_defined_exactly_once() -> None:
    """Only ``risk/circuit_breaker.py`` may declare ``class CircuitLevel``."""
    defs = _files_defining_circuit_level()
    assert defs == [_CANONICAL], (
        "CircuitLevel must be defined exactly once (in risk/circuit_breaker.py); "
        f"found: {[str(p.relative_to(_RISK_DIR.parents[2])) for p in defs]}"
    )


def test_layer_circuit_breaker_imports_canonical() -> None:
    """``layer_circuit_breaker`` must re-export the canonical enum, not redefine it."""
    from finalayze.risk.circuit_breaker import CircuitLevel as Canonical
    from finalayze.risk.layer_circuit_breaker import CircuitLevel as Layered

    assert Canonical is Layered, (
        "layer_circuit_breaker.CircuitLevel must be the same object as "
        "circuit_breaker.CircuitLevel (re-export, not redefinition)"
    )

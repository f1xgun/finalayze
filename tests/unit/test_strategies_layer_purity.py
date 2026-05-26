"""S4.2 — strategies/ must not reach into risk.position_sizer.

The audit flagged ``strategies/dual_momentum.py`` importing
``compute_realized_vol`` from ``risk.position_sizer`` as a layer-purity
violation (position_sizer mixes the pure vol calc with sizing concerns).
This test pins the contract so a future refactor cannot regress.

The pure vol helper lives in ``risk.regime`` (the canonical layer-4
sibling) — strategies that need annualised vol should import from there.
"""

from __future__ import annotations

import ast
from pathlib import Path

_STRATEGIES_DIR = Path(__file__).resolve().parents[2] / "src" / "finalayze" / "strategies"
_FORBIDDEN_MODULE = "finalayze.risk.position_sizer"


def _iter_strategy_python_files() -> list[Path]:
    return [p for p in _STRATEGIES_DIR.rglob("*.py") if "__pycache__" not in p.parts]


def _file_imports_forbidden(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and node.module == _FORBIDDEN_MODULE) or (
            isinstance(node, ast.Import)
            and any(alias.name == _FORBIDDEN_MODULE for alias in node.names)
        ):
            return True
    return False


def test_strategies_do_not_import_risk_position_sizer() -> None:
    """No file under strategies/ may import from risk.position_sizer."""
    root = _STRATEGIES_DIR.parent.parent.parent
    offenders = [
        str(path.relative_to(root))
        for path in _iter_strategy_python_files()
        if _file_imports_forbidden(path)
    ]

    assert not offenders, (
        f"strategies/ files must not import from {_FORBIDDEN_MODULE}; "
        f"use risk.regime.compute_realized_vol or move the helper to a "
        f"shared math module. Offenders: {offenders}"
    )

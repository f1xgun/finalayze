"""Backward-compatibility shim -- bond_cycle moved to finalayze.orchestration.bond_cycle.

Makes ``finalayze.core.bond_cycle`` an alias for the canonical module so that
both ``from finalayze.core.bond_cycle import X`` and
``patch("finalayze.core.bond_cycle.X")`` continue to work transparently.
"""

from __future__ import annotations

import sys

import finalayze.orchestration.bond_cycle as _canonical

# Register the canonical module under the old name so that all attribute
# lookups (including unittest.mock.patch targets) resolve correctly.
sys.modules[__name__] = _canonical

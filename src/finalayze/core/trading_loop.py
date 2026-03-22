"""Backward-compatibility shim -- trading_loop moved to finalayze.orchestration.trading_loop.

Makes ``finalayze.core.trading_loop`` an alias for the canonical module so that
both ``from finalayze.core.trading_loop import X`` and
``patch("finalayze.core.trading_loop.X")`` continue to work transparently.
"""

from __future__ import annotations

import sys

import finalayze.orchestration.trading_loop as _canonical

# Register the canonical module under the old name so that all attribute
# lookups (including unittest.mock.patch targets) resolve correctly.
sys.modules[__name__] = _canonical

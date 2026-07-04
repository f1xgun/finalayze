"""Local conftest for tax-engine unit tests.

The parent ``tests/unit/conftest.py`` registers an autouse fixture that imports
``finalayze.orchestration.trading_loop`` (a heavy L5+ import chain). The tax
engine is a pure L1/L2 Decimal package with zero dependency on that chain, so we
override the autouse fixture with a no-op here to keep these tests isolated and
fast (and to avoid importing higher layers into a layered-package test).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _patch_trading_loop_init() -> None:
    """No-op override of the parent autouse TradingLoop patch (not needed here)."""
    return

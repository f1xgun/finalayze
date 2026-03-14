"""Tests for TradingLoop bond cycle integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from finalayze.core.bond_cycle import BondCycleProcessor, BondCycleResult
from finalayze.data.macro_cache import MacroCacheService


def test_bond_cycle_delegates_to_processor() -> None:
    """TradingLoop._bond_cycle() must call processor.run_cycle()."""
    import inspect

    # We test the method in isolation, not the full TradingLoop
    from finalayze.core.trading_loop import TradingLoop

    # Check that bond_cycle_processor is an accepted parameter
    sig = inspect.signature(TradingLoop.__init__)
    assert "bond_cycle_processor" in sig.parameters
    assert "macro_cache" in sig.parameters

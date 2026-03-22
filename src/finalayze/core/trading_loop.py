"""Backward-compatibility shim -- trading_loop moved to finalayze.orchestration.trading_loop.

This module re-exports all public names so that existing ``from finalayze.core.trading_loop import ...``
statements continue to work. New code should import from ``finalayze.orchestration.trading_loop``.
"""

from finalayze.orchestration.trading_loop import *  # noqa: F401, F403
from finalayze.orchestration.trading_loop import TradingLoop as TradingLoop  # explicit re-export

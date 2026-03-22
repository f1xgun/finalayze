"""Tests for TradingLoop preflight checks (05-02)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest


def _make_loop(
    *,
    bond_processor: MagicMock | None = None,
    macro_cache: MagicMock | None = None,
    alerter: MagicMock | None = None,
    broker_router: MagicMock | None = None,
) -> MagicMock:
    """Create a minimal TradingLoop-like mock for testing preflight."""
    from finalayze.core.trading_loop import TradingLoop

    loop = MagicMock(spec=TradingLoop)
    loop._bond_processor = bond_processor
    loop._macro_cache = macro_cache
    loop._alerter = alerter or MagicMock()
    loop._broker_router = broker_router or MagicMock()
    loop._bond_enabled = True
    loop._settings = MagicMock()
    loop._settings.mode = "sandbox"
    loop._settings.tinkoff_token = "test-token"
    loop._registry = MagicMock()
    loop._registry.list_by_market.return_value = [MagicMock()]
    return loop


class TestPreflightChecks:
    """Preflight check tests -- verify gRPC, macro, ledger on startup."""

    def test_preflight_returns_true_all_ok(self) -> None:
        """Preflight returns True when gRPC, macro, ledger all pass."""
        from finalayze.core.trading_loop import TradingLoop

        broker_router = MagicMock()
        broker_router.route.return_value.get_portfolio.return_value = MagicMock(equity=Decimal(100))

        macro_cache = MagicMock()
        macro_cache.get.return_value = MagicMock(
            timestamp=datetime.now(UTC),
        )

        bond_processor = MagicMock()
        bond_processor.reconcile_with_broker.return_value = None

        loop = _make_loop(
            bond_processor=bond_processor,
            macro_cache=macro_cache,
            broker_router=broker_router,
        )
        result = TradingLoop._preflight_check(loop)
        assert result is True

    def test_preflight_returns_false_grpc_fails(self) -> None:
        """Preflight returns False when gRPC connectivity fails."""
        from finalayze.core.trading_loop import TradingLoop

        broker_router = MagicMock()
        broker_router.route.side_effect = Exception("gRPC timeout")

        macro_cache = MagicMock()
        macro_cache.get.return_value = MagicMock(timestamp=datetime.now(UTC))

        bond_processor = MagicMock()

        loop = _make_loop(
            bond_processor=bond_processor,
            macro_cache=macro_cache,
            broker_router=broker_router,
        )
        result = TradingLoop._preflight_check(loop)
        assert result is False
        assert loop._bond_enabled is False

    def test_preflight_grpc_fail_equity_continues(self) -> None:
        """When bond preflight fails, equity trading continues (independent degradation)."""
        from finalayze.core.trading_loop import TradingLoop

        broker_router = MagicMock()
        broker_router.route.side_effect = Exception("gRPC timeout")

        macro_cache = MagicMock()
        bond_processor = MagicMock()

        loop = _make_loop(
            bond_processor=bond_processor,
            macro_cache=macro_cache,
            broker_router=broker_router,
        )
        TradingLoop._preflight_check(loop)
        # bond_enabled is False but TradingLoop itself is not stopped
        assert loop._bond_enabled is False

    def test_preflight_sends_no_error_on_success(self) -> None:
        """Preflight does not send error alert on success."""
        from finalayze.core.trading_loop import TradingLoop

        broker_router = MagicMock()
        broker_router.route.return_value.get_portfolio.return_value = MagicMock(equity=Decimal(100))

        macro_cache = MagicMock()
        macro_cache.get.return_value = MagicMock(timestamp=datetime.now(UTC))

        bond_processor = MagicMock()
        alerter = MagicMock()

        loop = _make_loop(
            bond_processor=bond_processor,
            macro_cache=macro_cache,
            broker_router=broker_router,
            alerter=alerter,
        )
        result = TradingLoop._preflight_check(loop)
        assert result is True
        alerter.on_error.assert_not_called()

    def test_preflight_sends_degraded_alert_on_failure(self) -> None:
        """Preflight sends degraded-state alert when bond disabled."""
        from finalayze.core.trading_loop import TradingLoop

        broker_router = MagicMock()
        broker_router.route.side_effect = Exception("gRPC timeout")

        macro_cache = MagicMock()
        bond_processor = MagicMock()
        alerter = MagicMock()

        loop = _make_loop(
            bond_processor=bond_processor,
            macro_cache=macro_cache,
            broker_router=broker_router,
            alerter=alerter,
        )
        TradingLoop._preflight_check(loop)
        alerter.on_error.assert_called_once()

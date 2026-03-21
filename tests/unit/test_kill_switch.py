"""Unit tests for KillSwitch orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from finalayze.core.kill_switch import KillSwitch, KillSwitchResult


@dataclass(frozen=True)
class _FakeOrderState:
    order_id: str


def _make_kill_switch(tmp_path):
    """Build a KillSwitch with fully mocked dependencies."""
    broker_router = MagicMock()
    trading_loop = MagicMock()
    alerter = MagicMock()
    cb_us = MagicMock()
    cb_moex = MagicMock()
    circuit_breakers = {"us": cb_us, "moex": cb_moex}
    flag_path = tmp_path / "killed_flag"

    ks = KillSwitch(
        broker_router=broker_router,
        trading_loop=trading_loop,
        circuit_breakers=circuit_breakers,
        alerter=alerter,
        flag_path=flag_path,
    )
    return ks, broker_router, trading_loop, circuit_breakers, alerter, flag_path


class TestKillSwitch:
    """KillSwitch activation tests."""

    def test_activate_cancels_all_orders(self, tmp_path):
        """Test 1: activate() calls cancel_order_safe for each open order across all markets."""
        ks, broker_router, _trading_loop, _cbs, _alerter, _flag_path = _make_kill_switch(tmp_path)

        # Two markets with orders
        broker_router.registered_markets = ["moex", "us"]
        moex_broker = MagicMock()
        us_broker = MagicMock()
        moex_broker.get_open_orders.return_value = [
            _FakeOrderState(order_id="ord1"),
            _FakeOrderState(order_id="ord2"),
        ]
        us_broker.get_open_orders.return_value = [_FakeOrderState(order_id="ord3")]
        broker_router.route.side_effect = lambda m: moex_broker if m == "moex" else us_broker

        result = ks.activate(reason="test")

        moex_broker.cancel_order_safe.assert_any_call("ord1")
        moex_broker.cancel_order_safe.assert_any_call("ord2")
        us_broker.cancel_order_safe.assert_called_once_with("ord3")
        assert result.orders_cancelled == 3  # noqa: PLR2004

    def test_activate_stops_trading_loop(self, tmp_path):
        """Test 2: activate() calls trading_loop.stop()."""
        ks, broker_router, trading_loop, _cbs, _alerter, _flag_path = _make_kill_switch(tmp_path)
        broker_router.registered_markets = []

        ks.activate(reason="test")

        trading_loop.stop.assert_called_once()

    def test_activate_escalates_breakers(self, tmp_path):
        """Test 3: activate() calls override_level(LIQUIDATE) for each breaker."""
        from finalayze.risk.circuit_breaker import CircuitLevel

        ks, broker_router, _trading_loop, cbs, _alerter, _flag_path = _make_kill_switch(tmp_path)
        broker_router.registered_markets = []

        result = ks.activate(reason="test")

        for cb in cbs.values():
            cb.override_level.assert_called_once_with(CircuitLevel.LIQUIDATE)
        assert result.breakers_escalated == 2  # noqa: PLR2004

    def test_activate_sends_critical_alert(self, tmp_path):
        """Test 4: activate() calls alerter.send_alert with CRITICAL priority."""
        from finalayze.core.alerts import AlertPriority

        ks, broker_router, _trading_loop, _cbs, alerter, _flag_path = _make_kill_switch(tmp_path)
        broker_router.registered_markets = []

        result = ks.activate(reason="test")

        alerter.send_alert.assert_called_once()
        call_kwargs = alerter.send_alert.call_args
        assert call_kwargs.kwargs["priority"] == AlertPriority.CRITICAL
        assert result.alert_sent is True

    def test_activate_creates_persistent_flag(self, tmp_path):
        """Test 5: activate() creates persistent flag file at configured path."""
        ks, broker_router, _trading_loop, _cbs, _alerter, flag_path = _make_kill_switch(tmp_path)
        broker_router.registered_markets = []

        ks.activate(reason="emergency")

        assert flag_path.exists()
        content = flag_path.read_text()
        assert "killed:emergency:" in content

    def test_is_killed_returns_correct_state(self, tmp_path):
        """Test 6: is_killed returns True when flag file exists, False when not."""
        ks, _broker_router, _trading_loop, _cbs, _alerter, flag_path = _make_kill_switch(tmp_path)

        assert ks.is_killed is False

        flag_path.write_text("killed:test:2026-01-01")
        assert ks.is_killed is True

    def test_clear_flag_removes_file(self, tmp_path):
        """Test 7: clear_flag() removes the flag file."""
        ks, _broker_router, _trading_loop, _cbs, _alerter, flag_path = _make_kill_switch(tmp_path)

        flag_path.write_text("killed:test:2026-01-01")
        assert flag_path.exists()

        ks.clear_flag()

        assert not flag_path.exists()

    def test_activate_returns_result(self, tmp_path):
        """Test 8: activate() returns KillSwitchResult with all fields."""
        ks, broker_router, _trading_loop, _cbs, _alerter, _flag_path = _make_kill_switch(tmp_path)
        broker_router.registered_markets = ["moex"]
        moex_broker = MagicMock()
        moex_broker.get_open_orders.return_value = [_FakeOrderState(order_id="o1")]
        broker_router.route.return_value = moex_broker

        result = ks.activate(reason="manual")

        assert isinstance(result, KillSwitchResult)
        assert result.orders_cancelled == 1
        assert result.scheduler_stopped is True
        assert result.breakers_escalated == 2  # noqa: PLR2004
        assert result.alert_sent is True
        assert result.elapsed_seconds >= 0

    def test_activate_completes_under_timing_sla(self, tmp_path):
        """Test 9: elapsed_seconds < 1.0 with mocked deps (proves no unnecessary delays)."""
        ks, broker_router, _trading_loop, _cbs, _alerter, _flag_path = _make_kill_switch(tmp_path)
        broker_router.registered_markets = []

        result = ks.activate(reason="test")

        assert result.elapsed_seconds < 1.0

    def test_activate_logs_critical_event(self, tmp_path):
        """Test 10: activate() logs critical event with structlog."""
        ks, broker_router, _trading_loop, _cbs, _alerter, _flag_path = _make_kill_switch(tmp_path)
        broker_router.registered_markets = []

        with patch("finalayze.core.kill_switch._log") as mock_log:
            ks.activate(reason="test_reason")
            mock_log.critical.assert_called_once()
            call_kwargs = mock_log.critical.call_args
            assert "test_reason" in str(call_kwargs)

    def test_activate_handles_broker_cancel_failure(self, tmp_path):
        """Test 11: activate() handles broker cancel failures gracefully."""
        ks, broker_router, _trading_loop, _cbs, _alerter, flag_path = _make_kill_switch(tmp_path)
        broker_router.registered_markets = ["moex"]
        moex_broker = MagicMock()
        moex_broker.get_open_orders.return_value = [
            _FakeOrderState(order_id="o1"),
            _FakeOrderState(order_id="o2"),
        ]
        # First cancel fails, second succeeds
        moex_broker.cancel_order_safe.side_effect = [Exception("network error"), True]
        broker_router.route.return_value = moex_broker

        result = ks.activate(reason="test")

        # Should still complete all other steps
        assert result.scheduler_stopped is True
        assert result.breakers_escalated == 2  # noqa: PLR2004
        assert result.alert_sent is True
        assert flag_path.exists()

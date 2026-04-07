"""Tests for TelegramBotHandler wiring in main.py lifespan.

Validates:
  - _bot_handler_instance is stored at module level after create_app()
  - lifespan() wires kill_switch to _bot_handler_instance
  - lifespan() wires go_no_go_reporter to _bot_handler_instance
  - lifespan() wires broker_router and circuit_breakers to _bot_handler_instance
  - lifespan() does not crash when _bot_handler_instance is None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class _FakeLoop:
    """Minimal fake trading loop with attributes that lifespan reads."""

    _kill_switch: object = None
    _broker_router: object = None
    _circuit_breakers: dict = field(default_factory=dict)
    _alerter_ref: object = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _lifespan_patches():
    """Return context managers that patch deferred imports inside lifespan()."""
    return (
        patch("finalayze.api.v1.system.set_kill_switch"),
        patch("finalayze.api.v1.system.set_tinkoff_broker"),
        patch("finalayze.api.v1.system.set_health_monitor"),
    )


@pytest.fixture
def patch_settings():
    """Patch get_settings to return settings with telegram config."""
    mock_settings = MagicMock()
    mock_settings.mode.value = "sandbox"
    mock_settings.cors_origins = []
    mock_settings.telegram_bot_token = "test-token"
    mock_settings.telegram_webhook_secret = "test-secret"
    mock_settings.telegram_chat_id = "12345"
    mock_settings.telegram_allowed_chat_ids = ["12345"]
    mock_settings.tinkoff_token = ""
    mock_settings.effective_risk_limits.return_value = MagicMock()
    return mock_settings


class TestBotHandlerModuleLevel:
    """Test that _bot_handler_instance is stored at module level."""

    def test_bot_handler_instance_exists_after_create_app(self, patch_settings: MagicMock) -> None:
        """After create_app with telegram config, _bot_handler_instance should be set."""
        with (
            patch("finalayze.main.get_settings", return_value=patch_settings),
            patch("finalayze.main._settings", patch_settings),
            patch("config.settings.get_settings", return_value=patch_settings),
        ):
            from finalayze import main

            # Reset state
            main._bot_handler_instance = None

            # Re-run create_app
            _app = main.create_app()

            assert main._bot_handler_instance is not None, (
                "_bot_handler_instance should be set after create_app()"
            )


class TestLifespanBotWiring:
    """Test that lifespan() wires dependencies into _bot_handler_instance."""

    @pytest.mark.anyio
    async def test_kill_switch_wired_to_bot_handler(self) -> None:
        """After lifespan runs, _bot_handler_instance._kill_switch is set."""
        from finalayze import main
        from finalayze.core.modes import WorkMode

        fake_kill_switch = MagicMock()
        fake_broker_router = MagicMock()
        fake_broker_router.registered_markets = []
        fake_circuit_breakers = {"moex": MagicMock()}

        fake_loop = _FakeLoop(
            _kill_switch=fake_kill_switch,
            _broker_router=fake_broker_router,
            _circuit_breakers=fake_circuit_breakers,
            _alerter_ref=MagicMock(),
        )

        fake_bot = MagicMock()
        fake_bot._kill_switch = None
        fake_bot._go_no_go_reporter = None
        fake_bot._broker_router = None
        fake_bot._circuit_breakers = {}

        mock_settings = MagicMock()
        mock_settings.mode = WorkMode.SANDBOX

        p1, p2, p3 = _lifespan_patches()
        with (
            patch.object(main, "_settings", mock_settings),
            patch.object(main, "_bot_handler_instance", fake_bot),
            patch.object(main, "_build_trading_loop", return_value=fake_loop),
            p1,
            p2,
            p3,
        ):
            app = MagicMock()
            async with main.lifespan(app):
                pass

            # Verify kill_switch was wired
            assert fake_bot._kill_switch == fake_kill_switch, (
                "_bot_handler_instance._kill_switch should be set to the loop's kill_switch"
            )

    @pytest.mark.anyio
    async def test_go_no_go_reporter_wired_to_bot_handler(self) -> None:
        """After lifespan runs, _bot_handler_instance._go_no_go_reporter is set."""
        from finalayze import main
        from finalayze.core.modes import WorkMode

        fake_loop = _FakeLoop(
            _kill_switch=MagicMock(),
            _broker_router=MagicMock(),
            _circuit_breakers={"moex": MagicMock()},
            _alerter_ref=MagicMock(),
        )
        fake_loop._broker_router.registered_markets = []

        fake_bot = MagicMock()
        fake_bot._kill_switch = None
        fake_bot._go_no_go_reporter = None

        mock_settings = MagicMock()
        mock_settings.mode = WorkMode.SANDBOX

        p1, p2, p3 = _lifespan_patches()
        with (
            patch.object(main, "_settings", mock_settings),
            patch.object(main, "_bot_handler_instance", fake_bot),
            patch.object(main, "_build_trading_loop", return_value=fake_loop),
            p1,
            p2,
            p3,
        ):
            app = MagicMock()
            async with main.lifespan(app):
                pass

            # Verify go_no_go_reporter was wired (may be GoNoGoReporter or MagicMock
            # depending on whether gate_thresholds.yaml exists; either way, the
            # attribute setter should have been called)
            assert fake_bot._go_no_go_reporter is not None, (
                "_bot_handler_instance._go_no_go_reporter should be set"
            )

    @pytest.mark.anyio
    async def test_broker_router_and_breakers_wired(self) -> None:
        """After lifespan, _bot_handler_instance gets broker_router and circuit_breakers."""
        from finalayze import main
        from finalayze.core.modes import WorkMode

        fake_broker_router = MagicMock()
        fake_broker_router.registered_markets = []
        fake_circuit_breakers = {"moex": MagicMock()}

        fake_loop = _FakeLoop(
            _kill_switch=MagicMock(),
            _broker_router=fake_broker_router,
            _circuit_breakers=fake_circuit_breakers,
            _alerter_ref=MagicMock(),
        )

        fake_bot = MagicMock()
        fake_bot._broker_router = None
        fake_bot._circuit_breakers = {}
        fake_bot._trading_loop = None

        mock_settings = MagicMock()
        mock_settings.mode = WorkMode.SANDBOX

        p1, p2, p3 = _lifespan_patches()
        with (
            patch.object(main, "_settings", mock_settings),
            patch.object(main, "_bot_handler_instance", fake_bot),
            patch.object(main, "_build_trading_loop", return_value=fake_loop),
            p1,
            p2,
            p3,
        ):
            app = MagicMock()
            async with main.lifespan(app):
                pass

            assert fake_bot._broker_router == fake_broker_router
            assert fake_bot._circuit_breakers == fake_circuit_breakers

    @pytest.mark.anyio
    async def test_no_crash_when_bot_handler_is_none(self) -> None:
        """When _bot_handler_instance is None (no telegram config), lifespan runs fine."""
        from finalayze import main
        from finalayze.core.modes import WorkMode

        fake_loop = _FakeLoop(
            _kill_switch=MagicMock(),
            _broker_router=MagicMock(),
            _circuit_breakers={"moex": MagicMock()},
            _alerter_ref=MagicMock(),
        )
        fake_loop._broker_router.registered_markets = []

        mock_settings = MagicMock()
        mock_settings.mode = WorkMode.SANDBOX

        p1, p2, p3 = _lifespan_patches()
        with (
            patch.object(main, "_settings", mock_settings),
            patch.object(main, "_bot_handler_instance", None),
            patch.object(main, "_build_trading_loop", return_value=fake_loop),
            p1,
            p2,
            p3,
        ):
            app = MagicMock()
            # Should not raise
            async with main.lifespan(app):
                pass

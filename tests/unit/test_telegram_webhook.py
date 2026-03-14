"""Tests for Telegram webhook endpoint and bot handler (05-03)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# These imports will fail until implementation exists
from finalayze.core.telegram_bot import TelegramBotHandler


def _make_handler(
    allowed_chat_ids: list[str] | None = None,
) -> tuple[TelegramBotHandler, dict[str, MagicMock]]:
    """Create a TelegramBotHandler with mocked dependencies."""
    alerter = MagicMock()
    alerter._send = AsyncMock(return_value=True)

    broker_router = MagicMock()
    circuit_breakers: dict[str, MagicMock] = {
        "us": MagicMock(),
        "moex": MagicMock(),
    }

    settings = MagicMock()
    settings.telegram_allowed_chat_ids = allowed_chat_ids or ["123456"]
    settings.telegram_webhook_secret = "test-secret"

    handler = TelegramBotHandler(
        alerter=alerter,
        broker_router=broker_router,
        circuit_breakers=circuit_breakers,
        settings=settings,
    )

    mocks = {
        "alerter": alerter,
        "broker_router": broker_router,
        "circuit_breakers": circuit_breakers,
        "settings": settings,
    }
    return handler, mocks


def _make_app(handler: TelegramBotHandler, webhook_secret: str = "test-secret") -> FastAPI:
    """Create a FastAPI app with telegram webhook endpoint."""
    from finalayze.api.v1.telegram import create_telegram_router

    app = FastAPI()
    router = create_telegram_router(handler, webhook_secret)
    app.include_router(router)
    return app


class TestWebhookSecretValidation:
    """POST /api/telegram/webhook validates X-Telegram-Bot-Api-Secret-Token."""

    def test_valid_secret_returns_200(self) -> None:
        """POST with valid secret token returns 200."""
        handler, _ = _make_handler()
        app = _make_app(handler)
        client = TestClient(app)

        resp = client.post(
            "/api/telegram/webhook",
            json={"message": {"chat": {"id": 123456}, "text": "/status"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )
        assert resp.status_code == 200

    def test_invalid_secret_returns_403(self) -> None:
        """POST with invalid secret token returns 403."""
        handler, _ = _make_handler()
        app = _make_app(handler)
        client = TestClient(app)

        resp = client.post(
            "/api/telegram/webhook",
            json={"message": {"chat": {"id": 123456}, "text": "/status"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
        )
        assert resp.status_code == 403

    def test_missing_secret_returns_403(self) -> None:
        """POST without secret header returns 403."""
        handler, _ = _make_handler()
        app = _make_app(handler)
        client = TestClient(app)

        resp = client.post(
            "/api/telegram/webhook",
            json={"message": {"chat": {"id": 123456}, "text": "/status"}},
        )
        assert resp.status_code == 403


class TestChatIdWhitelist:
    """Only whitelisted chat_ids receive responses."""

    def test_whitelisted_chat_id_gets_response(self) -> None:
        """Message from whitelisted chat_id returns ok."""
        handler, _ = _make_handler(allowed_chat_ids=["123456"])
        app = _make_app(handler)
        client = TestClient(app)

        resp = client.post(
            "/api/telegram/webhook",
            json={"message": {"chat": {"id": 123456}, "text": "/status"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )
        assert resp.status_code == 200
        assert resp.json().get("ok") != "ignored"

    def test_non_whitelisted_chat_id_returns_ignored(self) -> None:
        """Message from non-whitelisted chat_id returns {"ok": "ignored"}."""
        handler, _ = _make_handler(allowed_chat_ids=["123456"])
        app = _make_app(handler)
        client = TestClient(app)

        resp = client.post(
            "/api/telegram/webhook",
            json={"message": {"chat": {"id": 999999}, "text": "/status"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": "ignored"}


class TestCommandDispatch:
    """Bot dispatches /status and /breakers, ignores unknown commands."""

    def test_status_command_queries_broker(self) -> None:
        """handle_status queries broker_router for positions."""
        handler, mocks = _make_handler()

        # Setup portfolio response
        portfolio = MagicMock()
        portfolio.equity = Decimal("50000")
        portfolio.positions = {"AAPL": Decimal("10"), "MSFT": Decimal("5")}
        portfolio.cash = Decimal("10000")

        broker = MagicMock()
        broker.get_portfolio.return_value = portfolio
        mocks["broker_router"].route.return_value = broker
        mocks["broker_router"].registered_markets = ["us", "moex"]

        app = _make_app(handler)
        client = TestClient(app)

        resp = client.post(
            "/api/telegram/webhook",
            json={"message": {"chat": {"id": 123456}, "text": "/status"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )
        assert resp.status_code == 200
        # Should have called alerter._send with HTML content
        mocks["alerter"]._send.assert_called()
        call_text = mocks["alerter"]._send.call_args[0][0]
        assert (
            "AAPL" in call_text or "Portfolio" in call_text.lower() or "equity" in call_text.lower()
        )

    def test_breakers_command_shows_levels(self) -> None:
        """/breakers command responds with circuit breaker states."""
        handler, mocks = _make_handler()

        # Setup circuit breaker states
        from finalayze.risk.circuit_breaker import CircuitLevel

        for market_id, cb in mocks["circuit_breakers"].items():
            cb.level = CircuitLevel.NORMAL
            cb.market_id = market_id
            cb.baseline = Decimal("100000")

        app = _make_app(handler)
        client = TestClient(app)

        resp = client.post(
            "/api/telegram/webhook",
            json={"message": {"chat": {"id": 123456}, "text": "/breakers"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )
        assert resp.status_code == 200
        mocks["alerter"]._send.assert_called()
        call_text = mocks["alerter"]._send.call_args[0][0]
        assert "us" in call_text.lower() or "US" in call_text
        assert "normal" in call_text.lower() or "NORMAL" in call_text

    def test_unknown_command_returns_ok_no_response(self) -> None:
        """Unknown command returns ok but does not send a message."""
        handler, mocks = _make_handler()

        app = _make_app(handler)
        client = TestClient(app)

        resp = client.post(
            "/api/telegram/webhook",
            json={"message": {"chat": {"id": 123456}, "text": "/unknown"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )
        assert resp.status_code == 200
        mocks["alerter"]._send.assert_not_called()

    def test_malformed_json_returns_400(self) -> None:
        """Malformed JSON body returns 400."""
        handler, _ = _make_handler()
        app = _make_app(handler)
        client = TestClient(app)

        resp = client.post(
            "/api/telegram/webhook",
            content=b"not json",
            headers={
                "X-Telegram-Bot-Api-Secret-Token": "test-secret",
                "Content-Type": "application/json",
            },
        )
        # FastAPI returns 422 for invalid JSON by default, or we handle it as 400
        assert resp.status_code in (400, 422)


class TestWebhookMountedInCreateApp:
    """Telegram webhook route is mounted in create_app() when configured."""

    def test_webhook_route_mounted_with_token_and_secret(self) -> None:
        """create_app() mounts /api/telegram/webhook when token+secret configured."""
        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = "test-bot-token"  # noqa: S105
        mock_settings.telegram_webhook_secret = "test-secret"  # noqa: S105
        mock_settings.telegram_allowed_chat_ids = ["123456"]
        mock_settings.cors_origins = []
        mock_settings.mode = MagicMock()
        mock_settings.mode.value = "test"

        with patch("finalayze.main.get_settings", return_value=mock_settings), patch(
            "finalayze.main._settings", mock_settings
        ):
            from finalayze.main import create_app

            app = create_app()

        # Check that /api/telegram/webhook route exists
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/api/telegram/webhook" in routes

    def test_webhook_route_not_mounted_without_token(self) -> None:
        """create_app() does NOT mount webhook when token is empty."""
        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = ""
        mock_settings.telegram_webhook_secret = "test-secret"  # noqa: S105
        mock_settings.cors_origins = []
        mock_settings.mode = MagicMock()
        mock_settings.mode.value = "test"

        with patch("finalayze.main.get_settings", return_value=mock_settings), patch(
            "finalayze.main._settings", mock_settings
        ):
            from finalayze.main import create_app

            app = create_app()

        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/api/telegram/webhook" not in routes

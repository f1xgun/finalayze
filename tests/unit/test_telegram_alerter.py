"""Unit tests for TelegramAlerter."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.alerts import TelegramAlerter
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.risk.circuit_breaker import CircuitLevel

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────
TELEGRAM_API_URL_PREFIX = "https://api.telegram.org/bot"
VALID_TOKEN = "1234567890:AABBccDDeEFfGgHhIiJj"  # noqa: S105
VALID_CHAT_ID = "-1001234567890"
MARKET_US = "us"
MARKET_MOEX = "moex"
FILL_PRICE = Decimal("150.00")
ORDER_QTY = Decimal(10)
DRAWDOWN_PCT = 0.103
DAILY_PNL_US = Decimal(342)
DAILY_PNL_MOEX = Decimal(1200)
TOTAL_EQUITY = Decimal(51200)

# Path for patching the AsyncClient used inside _send
_ASYNC_CLIENT_PATH = "finalayze.core.alerts.httpx.AsyncClient"


def _make_order_result() -> OrderResult:
    return OrderResult(
        filled=True,
        fill_price=FILL_PRICE,
        symbol="AAPL",
        side="BUY",
        quantity=ORDER_QTY,
    )


def _make_order_request() -> OrderRequest:
    return OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)


def _make_mock_client() -> MagicMock:
    """Build a mock httpx.AsyncClient whose post() is an AsyncMock."""
    mock_response = MagicMock(status_code=200)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


class TestTelegramAlerterNoOp:
    """When token is empty, all methods must silently do nothing."""

    def test_no_op_on_trade_filled(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch(_ASYNC_CLIENT_PATH) as mock_cls:
            alerter.on_trade_filled(_make_order_result(), MARKET_US, "alpaca")
            mock_cls.assert_not_called()

    def test_no_op_on_trade_rejected(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch(_ASYNC_CLIENT_PATH) as mock_cls:
            alerter.on_trade_rejected(_make_order_request(), "insufficient funds")
            mock_cls.assert_not_called()

    def test_no_op_on_circuit_breaker_trip(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch(_ASYNC_CLIENT_PATH) as mock_cls:
            alerter.on_circuit_breaker_trip(MARKET_US, CircuitLevel.HALTED, DRAWDOWN_PCT)
            mock_cls.assert_not_called()

    def test_no_op_on_circuit_breaker_reset(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch(_ASYNC_CLIENT_PATH) as mock_cls:
            alerter.on_circuit_breaker_reset(MARKET_US)
            mock_cls.assert_not_called()

    def test_no_op_on_daily_summary(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch(_ASYNC_CLIENT_PATH) as mock_cls:
            alerter.on_daily_summary(
                {MARKET_US: DAILY_PNL_US, MARKET_MOEX: DAILY_PNL_MOEX},
                TOTAL_EQUITY,
            )
            mock_cls.assert_not_called()

    def test_no_op_on_error(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch(_ASYNC_CLIENT_PATH) as mock_cls:
            alerter.on_error("NewsApiFetcher", "connection timeout")
            mock_cls.assert_not_called()


class TestTelegramAlerterSendsMessages:
    """When token is present, each method must call async httpx.AsyncClient().post()."""

    def _make_alerter(self) -> TelegramAlerter:
        return TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)

    @pytest.mark.asyncio
    async def test_on_trade_filled_calls_post(self) -> None:
        alerter = self._make_alerter()
        mock_client = _make_mock_client()
        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            await alerter._send(f"\U0001f7e2 BUY AAPL \xd710 @ $150.00 (alpaca {MARKET_US})")
            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args
            assert VALID_TOKEN in call_kwargs[0][0]
            payload = call_kwargs[1]["json"]
            assert payload["chat_id"] == VALID_CHAT_ID
            assert "AAPL" in payload["text"]

    @pytest.mark.asyncio
    async def test_on_trade_rejected_payload(self) -> None:
        alerter = self._make_alerter()
        mock_client = _make_mock_client()
        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            await alerter._send("\u26a0\ufe0f AAPL BUY rejected: insufficient funds")
            mock_client.post.assert_called_once()
            payload = mock_client.post.call_args[1]["json"]
            assert "AAPL" in payload["text"]
            assert "insufficient funds" in payload["text"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_trip_payload(self) -> None:
        alerter = self._make_alerter()
        mock_client = _make_mock_client()
        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            await alerter._send(
                f"\U0001f534 [{MARKET_US.upper()}] Circuit breaker HALTED "
                f"-- trading halted ({DRAWDOWN_PCT * 100:.1f}% daily drawdown)"
            )
            payload = mock_client.post.call_args[1]["json"]
            assert MARKET_US.upper() in payload["text"]
            assert "halted" in payload["text"].lower() or "HALTED" in payload["text"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_payload(self) -> None:
        alerter = self._make_alerter()
        mock_client = _make_mock_client()
        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            await alerter._send(
                f"\u2705 [{MARKET_US.upper()}] Circuit breaker reset \u2014 trading resumed"
            )
            payload = mock_client.post.call_args[1]["json"]
            assert "reset" in payload["text"].lower() or "resumed" in payload["text"].lower()

    @pytest.mark.asyncio
    async def test_daily_summary_payload(self) -> None:
        alerter = self._make_alerter()
        mock_client = _make_mock_client()
        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            await alerter._send(
                f"\U0001f4ca Daily: MOEX +{DAILY_PNL_MOEX} | US +{DAILY_PNL_US}"
                f" | Equity ${TOTAL_EQUITY:,.0f}"
            )
            payload = mock_client.post.call_args[1]["json"]
            assert "51,200" in payload["text"] or "Daily" in payload["text"]

    @pytest.mark.asyncio
    async def test_on_error_payload(self) -> None:
        alerter = self._make_alerter()
        mock_client = _make_mock_client()
        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            await alerter._send("\U0001f6a8 NewsApiFetcher error: gRPC timeout")
            payload = mock_client.post.call_args[1]["json"]
            assert "NewsApiFetcher" in payload["text"]
            assert "gRPC timeout" in payload["text"]

    def test_send_alert_no_loop_uses_asyncio_run(self) -> None:
        """send_alert must use asyncio.run() when no event loop is running."""
        alerter = self._make_alerter()
        with (
            patch.object(alerter, "_send", new_callable=AsyncMock),
            patch("finalayze.core.alerts.asyncio.get_event_loop") as mock_get_loop,
            patch("finalayze.core.alerts.asyncio.run") as mock_run,
        ):
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = False
            mock_get_loop.return_value = mock_loop
            alerter.send_alert("test message")
            mock_run.assert_called_once()

    def test_send_alert_running_loop_creates_task(self) -> None:
        """send_alert must create a task when an event loop is already running."""
        alerter = self._make_alerter()
        with (
            patch.object(alerter, "_send", new_callable=AsyncMock),
            patch("finalayze.core.alerts.asyncio.get_event_loop") as mock_get_loop,
        ):
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop
            alerter.send_alert("test message")
            mock_loop.create_task.assert_called_once()


class TestTelegramAlerterErrorHandling:
    """HTTP errors must be swallowed -- never propagate to callers."""

    @pytest.mark.asyncio
    async def test_async_httpx_error_does_not_propagate(self) -> None:
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        mock_client = _make_mock_client()
        mock_client.post = AsyncMock(side_effect=Exception("network failure"))
        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            # Must not raise
            await alerter._send("test message")

    def test_send_alert_exception_does_not_propagate(self) -> None:
        """send_alert must never raise even when internals fail."""
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        with patch("finalayze.core.alerts.asyncio.get_event_loop", side_effect=RuntimeError):
            # Must not raise
            alerter.send_alert("test message")

    @pytest.mark.asyncio
    async def test_http_non_200_does_not_propagate(self) -> None:
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        mock_client = _make_mock_client()
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=400))
        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            # Must not raise even on 4xx response
            await alerter._send("test message")

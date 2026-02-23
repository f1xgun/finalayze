"""Unit tests for TelegramAlerter."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

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


class TestTelegramAlerterNoOp:
    """When token is empty, all methods must silently do nothing."""

    def test_no_op_on_trade_filled(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_trade_filled(_make_order_result(), MARKET_US, "alpaca")
            mock_post.assert_not_called()

    def test_no_op_on_trade_rejected(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_trade_rejected(_make_order_request(), "insufficient funds")
            mock_post.assert_not_called()

    def test_no_op_on_circuit_breaker_trip(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_circuit_breaker_trip(MARKET_US, CircuitLevel.HALTED, DRAWDOWN_PCT)
            mock_post.assert_not_called()

    def test_no_op_on_circuit_breaker_reset(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_circuit_breaker_reset(MARKET_US)
            mock_post.assert_not_called()

    def test_no_op_on_daily_summary(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_daily_summary(
                {MARKET_US: DAILY_PNL_US, MARKET_MOEX: DAILY_PNL_MOEX},
                TOTAL_EQUITY,
            )
            mock_post.assert_not_called()

    def test_no_op_on_error(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_error("NewsApiFetcher", "connection timeout")
            mock_post.assert_not_called()


class TestTelegramAlerterSendsMessages:
    """When token is present, each method must call httpx.post with correct payload."""

    def _make_alerter(self) -> TelegramAlerter:
        return TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)

    def test_on_trade_filled_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_trade_filled(_make_order_result(), MARKET_US, "alpaca")
            mock_post.assert_called_once()
            (url,) = mock_post.call_args.args
            assert VALID_TOKEN in url
            payload = mock_post.call_args.kwargs["json"]
            assert payload["chat_id"] == VALID_CHAT_ID
            assert "AAPL" in payload["text"]
            assert "BUY" in payload["text"]

    def test_on_trade_rejected_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_trade_rejected(_make_order_request(), "insufficient funds")
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "AAPL" in payload["text"]
            assert "insufficient funds" in payload["text"]

    def test_on_circuit_breaker_trip_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_circuit_breaker_trip(MARKET_US, CircuitLevel.HALTED, DRAWDOWN_PCT)
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert MARKET_US.upper() in payload["text"] or "us" in payload["text"].lower()
            assert "halted" in payload["text"].lower() or "HALTED" in payload["text"]

    def test_on_circuit_breaker_reset_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_circuit_breaker_reset(MARKET_US)
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "reset" in payload["text"].lower() or "resumed" in payload["text"].lower()

    def test_on_daily_summary_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_daily_summary(
                {MARKET_US: DAILY_PNL_US, MARKET_MOEX: DAILY_PNL_MOEX},
                TOTAL_EQUITY,
            )
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "51200" in payload["text"] or "Daily" in payload["text"]

    def test_on_error_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_error("NewsApiFetcher", "gRPC timeout")
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "NewsApiFetcher" in payload["text"]
            assert "gRPC timeout" in payload["text"]


class TestTelegramAlerterErrorHandling:
    """HTTP errors must be swallowed -- never propagate to callers."""

    def test_httpx_error_does_not_propagate(self) -> None:
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        with patch("httpx.post", side_effect=Exception("network failure")):
            # Must not raise
            alerter.on_error("component", "message")

    def test_http_non_200_does_not_propagate(self) -> None:
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=400)
            # Must not raise even on 4xx response
            alerter.on_trade_filled(_make_order_result(), MARKET_US, "alpaca")

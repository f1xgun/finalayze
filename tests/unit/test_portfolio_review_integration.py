"""Integration tests for portfolio review agent wiring in TradingLoop.

Verifies PFRA-01 (daily LLM review dispatch), PFRA-02 (Telegram delivery),
PFRA-03 (handler writes ONLY to TelegramAlerter -- no order pipeline).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Reset structlog config so capture_logs() works even after setup_logging().

    Other tests (API) may call setup_logging() which sets
    cache_logger_on_first_use=True and installs JSONRenderer.
    Module-level loggers (trading_loop._log) get cached with that config
    and won't route through capture_logs. We must swap the module-level
    _log with a fresh proxy.
    """
    structlog.reset_defaults()
    # Replace the module-level cached _log with a fresh proxy
    import finalayze.orchestration.trading_loop as tl_mod  # noqa: PLC0415

    tl_mod._log = structlog.get_logger()

from finalayze.analysis.portfolio_review_agent import (  # noqa: E402
    PORTFOLIO_REVIEW_SYSTEM_PROMPT,
    PortfolioReviewResult,
)
from finalayze.core.alerts import TelegramAlerter  # noqa: E402

# -- Helpers ------------------------------------------------------------------

_REVIEW_RESULT = PortfolioReviewResult(
    reviewed_at=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    positions=[],
    concentration_warnings=[],
    catalyst_events=[],
    overall_assessment="Portfolio is well-balanced.",
    risk_score=0.3,
)


def _make_tl() -> object:
    """Create a minimal TradingLoop instance without calling __init__."""
    from finalayze.core.trading_loop import TradingLoop

    return object.__new__(TradingLoop)


def _make_portfolio_mock() -> MagicMock:
    """Create a mock portfolio with equity and cash."""
    portfolio = MagicMock()
    portfolio.equity = Decimal(500000)
    portfolio.cash = Decimal(100000)
    return portfolio


# -- PFRA-01: Cron Dispatch ---------------------------------------------------


class TestCronDispatch:
    """APScheduler callback dispatches async review correctly."""

    def test_portfolio_review_cycle_dispatches_coroutine(self) -> None:
        """_portfolio_review_cycle calls run_coroutine_threadsafe on _async_loop."""
        tl = _make_tl()
        tl._llm_client = MagicMock()  # type: ignore[attr-defined]
        tl._async_loop = MagicMock()  # type: ignore[attr-defined]
        tl._async_loop.is_closed.return_value = False

        with patch("asyncio.run_coroutine_threadsafe") as mock_dispatch:
            tl._portfolio_review_cycle()  # type: ignore[attr-defined]

        mock_dispatch.assert_called_once()
        # First arg should be the coroutine from _run_portfolio_review_async
        # Second arg should be the async loop
        call_args = mock_dispatch.call_args
        assert call_args[0][1] is tl._async_loop  # type: ignore[attr-defined]

    def test_portfolio_review_cycle_noop_when_no_llm_client(self) -> None:
        """No dispatch when _llm_client is None."""
        tl = _make_tl()
        tl._llm_client = None  # type: ignore[attr-defined]
        tl._async_loop = MagicMock()  # type: ignore[attr-defined]

        with patch("asyncio.run_coroutine_threadsafe") as mock_dispatch:
            tl._portfolio_review_cycle()  # type: ignore[attr-defined]

        mock_dispatch.assert_not_called()

    def test_portfolio_review_cycle_noop_when_async_loop_none(self) -> None:
        """No dispatch when _async_loop is None."""
        tl = _make_tl()
        tl._llm_client = MagicMock()  # type: ignore[attr-defined]
        tl._async_loop = None  # type: ignore[attr-defined]

        with patch("asyncio.run_coroutine_threadsafe") as mock_dispatch:
            tl._portfolio_review_cycle()  # type: ignore[attr-defined]

        mock_dispatch.assert_not_called()

    def test_portfolio_review_cycle_noop_when_async_loop_closed(self) -> None:
        """No dispatch when _async_loop.is_closed() returns True."""
        tl = _make_tl()
        tl._llm_client = MagicMock()  # type: ignore[attr-defined]
        tl._async_loop = MagicMock()  # type: ignore[attr-defined]
        tl._async_loop.is_closed.return_value = True

        with patch("asyncio.run_coroutine_threadsafe") as mock_dispatch:
            tl._portfolio_review_cycle()  # type: ignore[attr-defined]

        mock_dispatch.assert_not_called()

    def test_portfolio_review_cycle_logs_skip_when_no_llm(self) -> None:
        """Logs portfolio_review_skipped when LLM client is None."""
        tl = _make_tl()
        tl._llm_client = None  # type: ignore[attr-defined]
        tl._async_loop = MagicMock()  # type: ignore[attr-defined]

        with structlog.testing.capture_logs() as captured:
            tl._portfolio_review_cycle()  # type: ignore[attr-defined]

        skip_logs = [log for log in captured if log.get("event") == "portfolio_review_skipped"]
        assert len(skip_logs) >= 1, f"Expected portfolio_review_skipped log, got: {captured}"
        assert skip_logs[0]["reason"] == "no LLM client configured"


# -- PFRA-01 / PFRA-02: LLM Call + Telegram Delivery -------------------------


class TestLLMCallAndTelegramDelivery:
    """_run_portfolio_review_async calls parse_structured and sends via Telegram."""

    @pytest.mark.asyncio
    async def test_run_review_calls_parse_structured_with_correct_model(self) -> None:
        """parse_structured receives PortfolioReviewResult as response_model."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock()

        llm_client = AsyncMock()
        llm_client.parse_structured = AsyncMock(return_value=_REVIEW_RESULT)

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]
        tl._circuit_breakers = {}  # type: ignore[attr-defined]
        tl._broker_router = MagicMock()  # type: ignore[attr-defined]

        await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

        llm_client.parse_structured.assert_called_once()
        call_kwargs = llm_client.parse_structured.call_args
        assert call_kwargs.kwargs.get("response_model") is PortfolioReviewResult or (
            call_kwargs[1].get("response_model") is PortfolioReviewResult
            if len(call_kwargs) > 1
            else call_kwargs.kwargs.get("response_model") is PortfolioReviewResult
        )

    @pytest.mark.asyncio
    async def test_run_review_sends_formatted_telegram(self) -> None:
        """Formatted review message is sent via _alerter._send."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock()

        llm_client = AsyncMock()
        llm_client.parse_structured = AsyncMock(return_value=_REVIEW_RESULT)

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]
        tl._circuit_breakers = {}  # type: ignore[attr-defined]
        tl._broker_router = MagicMock()  # type: ignore[attr-defined]

        await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

        alerter._send.assert_called_once()
        sent_text = alerter._send.call_args[0][0]
        assert "Portfolio Review" in sent_text
        assert "Portfolio is well-balanced." in sent_text

    @pytest.mark.asyncio
    async def test_run_review_uses_correct_system_prompt(self) -> None:
        """parse_structured receives PORTFOLIO_REVIEW_SYSTEM_PROMPT."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock()

        llm_client = AsyncMock()
        llm_client.parse_structured = AsyncMock(return_value=_REVIEW_RESULT)

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]
        tl._circuit_breakers = {}  # type: ignore[attr-defined]
        tl._broker_router = MagicMock()  # type: ignore[attr-defined]

        await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

        call_kwargs = llm_client.parse_structured.call_args
        # system prompt passed as keyword arg
        system_arg = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        assert system_arg == PORTFOLIO_REVIEW_SYSTEM_PROMPT


# -- PFRA-01: Graceful Degradation -------------------------------------------


class TestGracefulDegradation:
    """LLM timeout/error does not crash or block the trading loop."""

    @pytest.mark.asyncio
    async def test_llm_timeout_logs_failure_no_telegram(self) -> None:
        """TimeoutError from parse_structured is caught, no Telegram sent."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock()

        llm_client = AsyncMock()
        llm_client.parse_structured = AsyncMock(side_effect=TimeoutError("LLM timeout"))

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]
        tl._circuit_breakers = {}  # type: ignore[attr-defined]
        tl._broker_router = MagicMock()  # type: ignore[attr-defined]

        with structlog.testing.capture_logs() as captured:
            await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

        alerter._send.assert_not_called()
        failure_logs = [
            log for log in captured if log.get("event") == "portfolio_review_llm_failure"
        ]
        assert len(failure_logs) >= 1, f"Expected portfolio_review_llm_failure log, got: {captured}"

    @pytest.mark.asyncio
    async def test_llm_exception_logs_failure_no_telegram(self) -> None:
        """RuntimeError from parse_structured is caught, no Telegram sent."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock()

        llm_client = AsyncMock()
        llm_client.parse_structured = AsyncMock(side_effect=RuntimeError("LLM provider down"))

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]
        tl._circuit_breakers = {}  # type: ignore[attr-defined]
        tl._broker_router = MagicMock()  # type: ignore[attr-defined]

        with structlog.testing.capture_logs() as captured:
            await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

        alerter._send.assert_not_called()
        failure_logs = [
            log for log in captured if log.get("event") == "portfolio_review_llm_failure"
        ]
        assert len(failure_logs) >= 1

    @pytest.mark.asyncio
    async def test_run_review_never_raises(self) -> None:
        """_run_portfolio_review_async swallows ALL exceptions."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock(side_effect=Exception("Telegram down"))

        llm_client = AsyncMock()
        llm_client.parse_structured = AsyncMock(return_value=_REVIEW_RESULT)

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]
        tl._circuit_breakers = {}  # type: ignore[attr-defined]
        tl._broker_router = MagicMock()  # type: ignore[attr-defined]

        # Must not raise
        await tl._run_portfolio_review_async()  # type: ignore[attr-defined]


# -- PFRA-01: Broker Data Gathering ------------------------------------------


class TestGatherPortfolioData:
    """_gather_portfolio_data collects from all markets, handles errors."""

    def test_gather_returns_data_for_all_markets(self) -> None:
        """Returns dict with equity/cash/positions for each market."""
        portfolio = _make_portfolio_mock()
        positions = {"SBER": Decimal(100), "GAZP": Decimal(50)}

        broker = MagicMock()
        broker.get_portfolio.return_value = portfolio
        broker.get_positions.return_value = positions

        broker_router = MagicMock()
        broker_router.route.return_value = broker

        tl = _make_tl()
        tl._circuit_breakers = {"moex": MagicMock()}  # type: ignore[attr-defined]
        tl._broker_router = broker_router  # type: ignore[attr-defined]

        result = tl._gather_portfolio_data()  # type: ignore[attr-defined]

        assert "moex" in result
        market_data = result["moex"]
        assert isinstance(market_data, dict)
        assert market_data["equity"] == Decimal(500000)
        assert market_data["cash"] == Decimal(100000)
        assert market_data["positions"] == positions

    def test_gather_handles_broker_error_gracefully(self) -> None:
        """Broker error for one market does not block others."""
        good_portfolio = _make_portfolio_mock()
        good_positions = {"LKOH": Decimal(10)}

        good_broker = MagicMock()
        good_broker.get_portfolio.return_value = good_portfolio
        good_broker.get_positions.return_value = good_positions

        bad_broker = MagicMock()
        bad_broker.get_portfolio.side_effect = RuntimeError("Connection failed")

        broker_router = MagicMock()
        broker_router.route.side_effect = lambda m: bad_broker if m == "us" else good_broker

        tl = _make_tl()
        tl._circuit_breakers = {  # type: ignore[attr-defined]
            "us": MagicMock(),
            "moex": MagicMock(),
        }
        tl._broker_router = broker_router  # type: ignore[attr-defined]

        with structlog.testing.capture_logs() as captured:
            result = tl._gather_portfolio_data()  # type: ignore[attr-defined]

        # moex should succeed, us should fail silently
        assert "moex" in result
        assert "us" not in result

        error_logs = [
            log for log in captured if log.get("event") == "portfolio_review_broker_error"
        ]
        assert len(error_logs) >= 1
        assert error_logs[0]["market_id"] == "us"

    def test_gather_returns_empty_dict_when_no_markets(self) -> None:
        """Returns empty dict when no circuit_breakers configured."""
        tl = _make_tl()
        tl._circuit_breakers = {}  # type: ignore[attr-defined]
        tl._broker_router = MagicMock()  # type: ignore[attr-defined]

        result = tl._gather_portfolio_data()  # type: ignore[attr-defined]
        assert result == {}


# -- PFRA-02: Cron Registration -----------------------------------------------


class TestCronRegistration:
    """Cron job for _portfolio_review_cycle is registered in start()."""

    def test_start_registers_portfolio_review_cron_job(self) -> None:
        """start() registers _portfolio_review_cycle at hour=16, minute=0."""
        import inspect

        from finalayze.core.trading_loop import TradingLoop

        source = inspect.getsource(TradingLoop.start)

        # Verify _portfolio_review_cycle is registered as a cron job
        assert "_portfolio_review_cycle" in source, (
            "_portfolio_review_cycle not found in start() source"
        )
        assert "hour=16" in source, "Cron job must fire at hour=16 UTC"
        assert "minute=0" in source, "Cron job must fire at minute=0"


# -- PFRA-03: Handler Safety --------------------------------------------------


class TestHandlerSafety:
    """_run_portfolio_review_async writes ONLY to TelegramAlerter."""

    def test_no_order_pipeline_references_in_run_review(self) -> None:
        """_run_portfolio_review_async must not reference order pipeline methods."""
        import inspect

        from finalayze.core.trading_loop import TradingLoop

        source = inspect.getsource(TradingLoop._run_portfolio_review_async)

        forbidden = ["place_order", "generate_signal", "submit_order", "_submit_order"]
        for term in forbidden:
            assert term not in source, f"_run_portfolio_review_async must not reference '{term}'"

    def test_no_broker_router_in_run_review(self) -> None:
        """_run_portfolio_review_async must not call _broker_router directly."""
        import inspect

        from finalayze.core.trading_loop import TradingLoop

        source = inspect.getsource(TradingLoop._run_portfolio_review_async)
        assert "_broker_router" not in source, (
            "_run_portfolio_review_async must not access _broker_router (only _gather does)"
        )


# -- Fire-and-forget pattern --------------------------------------------------


class TestFireAndForget:
    """Verify fire-and-forget pattern: no .result() call."""

    def test_cycle_does_not_call_result(self) -> None:
        """_portfolio_review_cycle must NOT call .result() on the future."""
        import inspect

        from finalayze.core.trading_loop import TradingLoop

        source = inspect.getsource(TradingLoop._portfolio_review_cycle)
        # Strip comment lines before checking for .result() calls
        code_lines = [
            line
            for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert ".result()" not in code_only, (
            "_portfolio_review_cycle must not call .result() (fire-and-forget)"
        )

"""Unit tests for BrokerBase ABC, SimulatedBroker, and TinkoffBroker resilience."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.exceptions import BrokerError
from finalayze.core.schemas import Candle, PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult
from finalayze.execution.simulated_broker import SimulatedBroker

INITIAL_CASH = Decimal(100000)
SHARE_PRICE = Decimal(150)
ORDER_QTY = Decimal(10)
STOP_PRICE = Decimal(140)
LOW_PRICE = Decimal(135)
VOLUME = 1_000_000


def _candle(
    price: Decimal,
    day: int = 0,
    *,
    symbol: str = "AAPL",
    low: Decimal | None = None,
) -> Candle:
    """Create a candle at the given price."""
    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=day),
        open=price,
        high=price + Decimal(5),
        low=low if low is not None else price - Decimal(5),
        close=price,
        volume=VOLUME,
    )


class TestBrokerBase:
    """BrokerBase is an abstract class and cannot be instantiated."""

    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BrokerBase()  # type: ignore[abstract]

    def test_order_request_creation(self) -> None:
        req = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)
        assert req.symbol == "AAPL"
        assert req.side == "BUY"
        assert req.quantity == ORDER_QTY

    def test_order_result_defaults(self) -> None:
        result = OrderResult(filled=False)
        assert result.filled is False
        assert result.fill_price is None
        assert result.symbol == ""
        assert result.side == ""
        assert result.quantity == Decimal(0)
        assert result.reason == ""


class TestSimulatedBrokerInitialState:
    """Initial portfolio should reflect starting cash."""

    def test_initial_portfolio_cash(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        portfolio = broker.get_portfolio()
        assert portfolio.cash == INITIAL_CASH

    def test_initial_portfolio_equity(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        portfolio = broker.get_portfolio()
        assert portfolio.equity == INITIAL_CASH

    def test_initial_portfolio_no_positions(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        portfolio = broker.get_portfolio()
        assert portfolio.positions == {}


class TestSimulatedBrokerBuy:
    """Buy orders should fill at candle open, deduct cash, create position."""

    def test_buy_fills_at_open(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        candle = _candle(SHARE_PRICE)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)

        result = broker.submit_order(order, candle)

        assert result.filled is True
        assert result.fill_price == SHARE_PRICE
        assert result.symbol == "AAPL"
        assert result.side == "BUY"
        assert result.quantity == ORDER_QTY

    def test_buy_deducts_cash(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        candle = _candle(SHARE_PRICE)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)
        broker.submit_order(order, candle)

        expected_cash = INITIAL_CASH - SHARE_PRICE * ORDER_QTY
        assert broker.get_portfolio().cash == expected_cash

    def test_buy_creates_position(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        candle = _candle(SHARE_PRICE)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)
        broker.submit_order(order, candle)

        portfolio = broker.get_portfolio()
        assert portfolio.positions["AAPL"] == ORDER_QTY


class TestSimulatedBrokerSell:
    """Sell orders should fill at candle open, increase cash, remove position."""

    def test_sell_fills_at_open(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)

        sell_price = Decimal(160)
        sell_candle = _candle(sell_price, day=1)
        result = broker.submit_order(OrderRequest("AAPL", "SELL", ORDER_QTY), sell_candle)

        assert result.filled is True
        assert result.fill_price == sell_price
        assert result.side == "SELL"

    def test_sell_adds_cash(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)

        sell_price = Decimal(160)
        sell_candle = _candle(sell_price, day=1)
        broker.submit_order(OrderRequest("AAPL", "SELL", ORDER_QTY), sell_candle)

        expected = INITIAL_CASH - SHARE_PRICE * ORDER_QTY + sell_price * ORDER_QTY
        assert broker.get_portfolio().cash == expected

    def test_sell_removes_position(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)

        sell_candle = _candle(Decimal(160), day=1)
        broker.submit_order(OrderRequest("AAPL", "SELL", ORDER_QTY), sell_candle)

        assert "AAPL" not in broker.get_portfolio().positions

    def test_sell_partial_position(self) -> None:
        """Selling more than held should sell only what is held."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", Decimal(5)), buy_candle)

        sell_candle = _candle(Decimal(160), day=1)
        result = broker.submit_order(OrderRequest("AAPL", "SELL", Decimal(20)), sell_candle)

        assert result.filled is True
        assert result.quantity == Decimal(5)
        assert "AAPL" not in broker.get_portfolio().positions


class TestSimulatedBrokerInsufficientCash:
    """Orders that exceed available cash should be rejected."""

    def test_insufficient_cash_rejected(self) -> None:
        broker = SimulatedBroker(initial_cash=Decimal(100))
        candle = _candle(SHARE_PRICE)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)

        result = broker.submit_order(order, candle)

        assert result.filled is False
        assert result.reason != ""

    def test_insufficient_cash_no_position(self) -> None:
        broker = SimulatedBroker(initial_cash=Decimal(100))
        candle = _candle(SHARE_PRICE)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)
        broker.submit_order(order, candle)

        assert broker.get_portfolio().positions == {}


class TestSimulatedBrokerStopLoss:
    """Stop losses should trigger when candle low drops to or below stop price."""

    def test_stop_loss_triggers(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)
        broker.set_stop_loss("AAPL", STOP_PRICE)

        trigger_candle = _candle(SHARE_PRICE, day=1, low=LOW_PRICE)
        results = broker.check_stop_losses(trigger_candle)

        assert len(results) == 1
        assert results[0].filled is True
        assert results[0].fill_price == STOP_PRICE
        assert results[0].side == "SELL"

    def test_stop_loss_closes_position(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)
        broker.set_stop_loss("AAPL", STOP_PRICE)

        trigger_candle = _candle(SHARE_PRICE, day=1, low=LOW_PRICE)
        broker.check_stop_losses(trigger_candle)

        assert "AAPL" not in broker.get_portfolio().positions

    def test_stop_loss_no_trigger(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)
        broker.set_stop_loss("AAPL", STOP_PRICE)

        safe_candle = _candle(SHARE_PRICE, day=1, low=Decimal(145))
        results = broker.check_stop_losses(safe_candle)

        assert len(results) == 0
        assert "AAPL" in broker.get_portfolio().positions


class TestSimulatedBrokerEquity:
    """Equity should reflect cash + market value of positions."""

    def test_equity_with_position(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), candle)

        portfolio = broker.get_portfolio()
        expected_cash = INITIAL_CASH - SHARE_PRICE * ORDER_QTY
        expected_equity = expected_cash + candle.close * ORDER_QTY
        assert portfolio.equity == expected_equity

    def test_equity_updates_with_price(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), candle)

        new_price = Decimal(200)
        new_candle = _candle(new_price, day=1)
        broker.update_prices(new_candle)

        portfolio = broker.get_portfolio()
        expected_cash = INITIAL_CASH - SHARE_PRICE * ORDER_QTY
        expected_equity = expected_cash + new_price * ORDER_QTY
        assert portfolio.equity == expected_equity

    def test_sell_clears_stop_loss_entry(self) -> None:
        """Selling a position must remove its stop-loss to avoid stale entries."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)
        broker.set_stop_loss("AAPL", STOP_PRICE)

        sell_candle = _candle(Decimal(160), day=1)
        broker.submit_order(OrderRequest("AAPL", "SELL", ORDER_QTY), sell_candle)

        # Stop-loss entry must be cleared after position is fully closed
        assert "AAPL" not in broker._stop_states


class TestSimulatedBrokerFillCandleOptional:
    """SimulatedBroker must raise ValueError when fill_candle is None."""

    def test_submit_order_raises_if_no_candle(self) -> None:
        """SimulatedBroker must reject orders when no candle is provided."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(1))
        with pytest.raises(ValueError, match="fill_candle"):
            broker.submit_order(order, fill_candle=None)


# ---------------------------------------------------------------------------
# TinkoffBroker portfolio cache fallback and 70001 auto-reconnect tests
# ---------------------------------------------------------------------------


def _make_tinkoff_broker() -> object:
    """Create a TinkoffBroker with mocked dependencies for unit testing."""
    from finalayze.execution.tinkoff_broker import TinkoffBroker

    registry = MagicMock()
    broker = TinkoffBroker(
        token="test-token",
        registry=registry,
        sandbox=True,
    )
    # Pre-set account ID to skip _ensure_account_id
    broker._account_id = "test-account-id"
    return broker


def _make_portfolio_state(cash: Decimal = Decimal(50000)) -> PortfolioState:
    """Create a sample PortfolioState for testing."""
    return PortfolioState(
        cash=cash,
        positions={"BBG000BBJQV0": Decimal(10)},
        equity=Decimal(100000),
        timestamp=datetime.now(tz=UTC),
    )


class TestPortfolioFallbackCacheOnSuccess:
    """get_portfolio() caches result on success."""

    def test_successful_fetch_caches_portfolio(self) -> None:
        """After a successful get_portfolio, _last_known_portfolio is set."""
        broker = _make_tinkoff_broker()
        _make_portfolio_state()

        # Mock the async portfolio fetch to return a mock response
        mock_portfolio_response = MagicMock()
        mock_portfolio_response.total_amount_portfolio = MagicMock(units=100000, nano=0)
        mock_portfolio_response.positions = []

        with (
            patch.object(broker, "_call", return_value=mock_portfolio_response),
            patch.object(broker, "_run_async", return_value=mock_portfolio_response),
        ):
            # We need to mock at a higher level -- mock get_portfolio to test caching
            pass

        # Instead, directly test the caching mechanism by calling get_portfolio
        # with a properly mocked internal chain
        mock_response = MagicMock()
        mock_response.total_amount_portfolio = MagicMock(units=100000, nano=0)
        mock_response.positions = []

        def fake_call(fn: object) -> object:
            return mock_response

        broker._call = fake_call  # type: ignore[assignment]
        broker._run_async = lambda coro: mock_response  # type: ignore[assignment]

        broker.get_portfolio()

        assert broker._last_known_portfolio is not None
        assert broker._last_known_portfolio.equity == Decimal(100000)
        assert broker._last_portfolio_at is not None

    def test_successful_fetch_resets_error_counter(self) -> None:
        """Successful get_portfolio resets _consecutive_70001_errors to 0."""
        broker = _make_tinkoff_broker()
        broker._consecutive_70001_errors = 3

        mock_response = MagicMock()
        mock_response.total_amount_portfolio = MagicMock(units=50000, nano=0)
        mock_response.positions = []

        broker._call = lambda fn: mock_response  # type: ignore[assignment]
        broker._run_async = lambda coro: mock_response  # type: ignore[assignment]

        broker.get_portfolio()

        assert broker._consecutive_70001_errors == 0


class TestPortfolioFallback70001WithCache:
    """get_portfolio() returns cached result on 70001 failure when cache exists."""

    def test_returns_cached_on_70001(self) -> None:
        """On 70001 error with cache, returns _last_known_portfolio."""
        broker = _make_tinkoff_broker()
        cached = _make_portfolio_state()
        broker._last_known_portfolio = cached
        broker._last_portfolio_at = datetime.now(tz=UTC) - timedelta(minutes=5)

        def raise_70001(fn: object) -> object:
            msg = "Tinkoff portfolio fetch failed: 70001"
            raise BrokerError(msg)

        broker._call = raise_70001  # type: ignore[assignment]

        result = broker.get_portfolio()

        assert result is cached

    def test_increments_consecutive_errors(self) -> None:
        """Each 70001 fallback increments _consecutive_70001_errors."""
        broker = _make_tinkoff_broker()
        broker._last_known_portfolio = _make_portfolio_state()
        broker._last_portfolio_at = datetime.now(tz=UTC)
        broker._consecutive_70001_errors = 2

        def raise_70001(fn: object) -> object:
            msg = "Tinkoff portfolio fetch failed: 70001"
            raise BrokerError(msg)

        broker._call = raise_70001  # type: ignore[assignment]

        broker.get_portfolio()

        assert broker._consecutive_70001_errors == 3


class TestPortfolioFallback70001NoCache:
    """get_portfolio() raises BrokerError on 70001 when no cache exists."""

    def test_raises_when_no_cache(self) -> None:
        """On 70001 with no cached portfolio, BrokerError is raised."""
        broker = _make_tinkoff_broker()
        assert broker._last_known_portfolio is None

        def raise_70001(fn: object) -> object:
            msg = "Tinkoff portfolio fetch failed: 70001"
            raise BrokerError(msg)

        broker._call = raise_70001  # type: ignore[assignment]

        with pytest.raises(BrokerError, match="70001"):
            broker.get_portfolio()


class TestPortfolioFallbackNon70001:
    """get_portfolio() raises BrokerError for non-70001 errors (no fallback)."""

    def test_raises_on_non_70001_error(self) -> None:
        """Non-70001 errors should raise even if cache exists."""
        broker = _make_tinkoff_broker()
        broker._last_known_portfolio = _make_portfolio_state()
        broker._last_portfolio_at = datetime.now(tz=UTC)

        def raise_other(fn: object) -> object:
            msg = "Tinkoff portfolio fetch failed: connection timeout"
            raise BrokerError(msg)

        broker._call = raise_other  # type: ignore[assignment]

        with pytest.raises(BrokerError, match="connection timeout"):
            broker.get_portfolio()


class TestPortfolioFallbackAutoReconnect:
    """After 5 consecutive 70001 errors, reconnect_client() is called."""

    def test_reconnect_after_threshold(self) -> None:
        """5 consecutive 70001 errors triggers reconnect_client."""
        broker = _make_tinkoff_broker()
        broker._last_known_portfolio = _make_portfolio_state()
        broker._last_portfolio_at = datetime.now(tz=UTC)
        broker._consecutive_70001_errors = 4  # one more will hit threshold of 5

        def raise_70001(fn: object) -> object:
            msg = "Tinkoff portfolio fetch failed: 70001"
            raise BrokerError(msg)

        broker._call = raise_70001  # type: ignore[assignment]

        with patch.object(broker, "reconnect_client", return_value=True) as mock_reconnect:
            broker.get_portfolio()

        mock_reconnect.assert_called_once()
        assert broker._consecutive_70001_errors == 0

    def test_no_reconnect_below_threshold(self) -> None:
        """Below threshold, reconnect_client is NOT called."""
        broker = _make_tinkoff_broker()
        broker._last_known_portfolio = _make_portfolio_state()
        broker._last_portfolio_at = datetime.now(tz=UTC)
        broker._consecutive_70001_errors = 2  # will become 3, below 5

        def raise_70001(fn: object) -> object:
            msg = "Tinkoff portfolio fetch failed: 70001"
            raise BrokerError(msg)

        broker._call = raise_70001  # type: ignore[assignment]

        with patch.object(broker, "reconnect_client", return_value=True) as mock_reconnect:
            broker.get_portfolio()

        mock_reconnect.assert_not_called()

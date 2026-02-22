"""Backtest engine -- iterates candles and runs a strategy with risk management.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

from finalayze.core.schemas import (
    Candle,
    PortfolioState,
    SignalDirection,
    TradeResult,
)
from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.simulated_broker import SimulatedBroker
from finalayze.risk.position_sizer import compute_position_size
from finalayze.risk.pre_trade_check import PreTradeChecker
from finalayze.risk.stop_loss import compute_atr_stop_loss

if TYPE_CHECKING:
    from finalayze.strategies.base import BaseStrategy

# Default Half-Kelly parameters
_DEFAULT_WIN_RATE = 0.5
_DEFAULT_AVG_WIN_RATIO = Decimal("1.5")


class BacktestEngine:
    """Iterate candles and execute a strategy with risk management."""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_cash: Decimal = Decimal(100000),
        max_position_pct: float = 0.20,
        max_positions: int = 10,
        kelly_fraction: float = 0.5,
        atr_multiplier: Decimal = Decimal("2.0"),
    ) -> None:
        self._strategy = strategy
        self._initial_cash = initial_cash
        self._max_position_pct = max_position_pct
        self._max_positions = max_positions
        self._kelly_fraction = kelly_fraction
        self._atr_multiplier = atr_multiplier

    def run(
        self,
        symbol: str,
        segment_id: str,
        candles: list[Candle],
    ) -> tuple[list[TradeResult], list[PortfolioState]]:
        """Run the backtest over the given candle series.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            segment_id: Market segment identifier (e.g. "us_large_cap").
            candles: Chronologically ordered candle data.

        Returns:
            A tuple of (trades, portfolio_snapshots).
        """
        checker = PreTradeChecker(
            max_position_pct=self._max_position_pct,
            max_positions_per_market=self._max_positions,
        )
        broker = SimulatedBroker(initial_cash=self._initial_cash)

        trades: list[TradeResult] = []
        snapshots: list[PortfolioState] = []
        entry_prices: dict[str, Decimal] = {}

        for i in range(len(candles)):
            candle = candles[i]

            # (a) Check stop-losses
            stop_results = broker.check_stop_losses(candle)
            for sr in stop_results:
                if sr.filled and sr.fill_price is not None:
                    entry = entry_prices.pop(sr.symbol, sr.fill_price)
                    pnl = (sr.fill_price - entry) * sr.quantity
                    pnl_pct = (sr.fill_price - entry) / entry if entry != 0 else Decimal(0)
                    trades.append(
                        TradeResult(
                            signal_id=uuid4(),
                            symbol=sr.symbol,
                            side="SELL",
                            quantity=sr.quantity,
                            entry_price=entry,
                            exit_price=sr.fill_price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )
                    )

            # (b) Update broker prices
            broker.update_prices(candle)

            # (c) Generate signal from strategy
            history = candles[: i + 1]
            signal = self._strategy.generate_signal(symbol, history, segment_id)

            if signal is not None and i + 1 < len(candles):
                fill_candle = candles[i + 1]

                if signal.direction == SignalDirection.BUY:
                    self._handle_buy(broker, checker, fill_candle, symbol, history, entry_prices)

                elif signal.direction == SignalDirection.SELL:
                    self._handle_sell(broker, fill_candle, symbol, entry_prices, trades)

            # (f) Record portfolio snapshot
            snapshots.append(broker.get_portfolio())

        return trades, snapshots

    def _handle_buy(
        self,
        broker: SimulatedBroker,
        checker: PreTradeChecker,
        fill_candle: Candle,
        symbol: str,
        history: list[Candle],
        entry_prices: dict[str, Decimal],
    ) -> None:
        """Process a BUY signal: size, check, fill, stop-loss."""
        portfolio = broker.get_portfolio()

        # Compute position size via Half-Kelly
        position_value = compute_position_size(
            win_rate=_DEFAULT_WIN_RATE,
            avg_win_ratio=_DEFAULT_AVG_WIN_RATIO,
            equity=portfolio.equity,
            kelly_fraction=self._kelly_fraction,
            max_position_pct=self._max_position_pct,
        )

        if position_value <= 0:
            return

        # Pre-trade check
        result = checker.check(
            order_value=position_value,
            portfolio_equity=portfolio.equity,
            available_cash=portfolio.cash,
            open_position_count=len(portfolio.positions),
        )
        if not result.passed:
            return

        # Compute quantity at fill price
        fill_price = fill_candle.open
        if fill_price <= 0:
            return
        quantity = (position_value / fill_price).to_integral_value()
        if quantity <= 0:
            return

        order = OrderRequest(symbol=symbol, side="BUY", quantity=quantity)
        order_result = broker.submit_order(order, fill_candle)

        if order_result.filled and order_result.fill_price is not None:
            entry_prices[symbol] = order_result.fill_price

            # Set ATR stop-loss
            stop_price = compute_atr_stop_loss(
                entry_price=order_result.fill_price,
                candles=history,
                atr_multiplier=self._atr_multiplier,
            )
            if stop_price is not None:
                broker.set_stop_loss(symbol, stop_price)

    def _handle_sell(
        self,
        broker: SimulatedBroker,
        fill_candle: Candle,
        symbol: str,
        entry_prices: dict[str, Decimal],
        trades: list[TradeResult],
    ) -> None:
        """Process a SELL signal: sell all held quantity."""
        portfolio = broker.get_portfolio()
        held = portfolio.positions.get(symbol, Decimal(0))
        if held <= 0:
            return

        order = OrderRequest(symbol=symbol, side="SELL", quantity=held)
        order_result = broker.submit_order(order, fill_candle)

        if order_result.filled and order_result.fill_price is not None:
            entry = entry_prices.pop(symbol, order_result.fill_price)
            pnl = (order_result.fill_price - entry) * order_result.quantity
            pnl_pct = (order_result.fill_price - entry) / entry if entry != 0 else Decimal(0)
            trades.append(
                TradeResult(
                    signal_id=uuid4(),
                    symbol=symbol,
                    side="SELL",
                    quantity=order_result.quantity,
                    entry_price=entry,
                    exit_price=order_result.fill_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            )

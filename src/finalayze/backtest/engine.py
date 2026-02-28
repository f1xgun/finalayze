"""Backtest engine -- iterates candles and runs a strategy with risk management.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
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
from finalayze.risk.kelly import TradeRecord
from finalayze.risk.position_sizer import compute_position_size
from finalayze.risk.pre_trade_check import PreTradeChecker
from finalayze.risk.stop_loss import compute_atr_stop_loss

if TYPE_CHECKING:
    from finalayze.backtest.costs import TransactionCosts
    from finalayze.risk.circuit_breaker import CircuitBreaker
    from finalayze.risk.kelly import RollingKelly
    from finalayze.risk.loss_limits import LossLimitTracker
    from finalayze.strategies.base import BaseStrategy

# Default Half-Kelly parameters (used when no RollingKelly is provided)
_DEFAULT_WIN_RATE = Decimal("0.5")
_DEFAULT_AVG_WIN_RATIO = Decimal("1.5")


class BacktestEngine:
    """Iterate candles and execute a strategy with risk management."""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_cash: Decimal = Decimal(100000),
        max_position_pct: Decimal = Decimal("0.20"),
        max_positions: int = 10,
        kelly_fraction: Decimal = Decimal("0.5"),
        atr_multiplier: Decimal = Decimal("2.0"),
        transaction_costs: TransactionCosts | None = None,
        trail_activation_atr: Decimal = Decimal("1.0"),
        trail_distance_atr: Decimal = Decimal("1.5"),
        circuit_breaker: CircuitBreaker | None = None,
        rolling_kelly: RollingKelly | None = None,
        loss_limits: LossLimitTracker | None = None,
    ) -> None:
        self._strategy = strategy
        self._initial_cash = initial_cash
        self._max_position_pct = max_position_pct
        self._max_positions = max_positions
        self._kelly_fraction = kelly_fraction
        self._atr_multiplier = atr_multiplier
        self._transaction_costs = transaction_costs
        self._trail_activation_atr = trail_activation_atr
        self._trail_distance_atr = trail_distance_atr
        self._circuit_breaker = circuit_breaker
        self._rolling_kelly = rolling_kelly
        self._loss_limits = loss_limits

    def run(  # noqa: PLR0912, PLR0915
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

        # Set initial baseline for circuit breaker
        if self._circuit_breaker is not None:
            self._circuit_breaker.reset_daily(self._initial_cash)

        # Loss limit tracking
        current_day = None
        current_week = None
        if self._loss_limits is not None and candles:
            first_ts = candles[0].timestamp
            self._loss_limits.reset_day(first_ts, self._initial_cash)
            self._loss_limits.reset_week(first_ts, self._initial_cash)
            current_day = first_ts.date()
            iso = first_ts.date().isocalendar()
            current_week = (iso[0], iso[1])

        for i in range(len(candles)):
            candle = candles[i]

            # (a) Update simulation timestamp
            broker.set_timestamp(candle.timestamp)

            # (a2) Reset loss limits on day/week boundary
            if self._loss_limits is not None:
                candle_date = candle.timestamp.date()
                iso = candle_date.isocalendar()
                candle_week = (iso[0], iso[1])
                portfolio_eq = broker.get_portfolio().equity
                if candle_date != current_day:
                    current_day = candle_date
                    self._loss_limits.reset_day(candle.timestamp, portfolio_eq)
                if candle_week != current_week:
                    current_week = candle_week
                    self._loss_limits.reset_week(candle.timestamp, portfolio_eq)

            # (b) Update broker prices first (before stop-loss check)
            broker.update_prices(candle)

            # (c) Check stop-losses after prices are updated
            stop_results = broker.check_stop_losses(candle)
            for sr in stop_results:
                if sr.filled and sr.fill_price is not None:
                    entry = entry_prices.pop(sr.symbol, sr.fill_price)
                    pnl = (sr.fill_price - entry) * sr.quantity
                    # Deduct exit transaction costs
                    if self._transaction_costs is not None:
                        pnl -= self._transaction_costs.total_cost(sr.fill_price, sr.quantity)
                    pnl_pct = (sr.fill_price - entry) / entry if entry != 0 else Decimal(0)
                    trade = TradeResult(
                        signal_id=uuid4(),
                        symbol=sr.symbol,
                        side="SELL",
                        quantity=sr.quantity,
                        entry_price=entry,
                        exit_price=sr.fill_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
                    trades.append(trade)
                    self._record_trade(trade)

            # (c2) Check circuit breaker level
            if self._circuit_breaker is not None:
                portfolio = broker.get_portfolio()
                cb_level = self._circuit_breaker.check(
                    current_equity=portfolio.equity,
                    baseline_equity=self._circuit_breaker.baseline,
                )

                # L3: liquidate all positions
                if cb_level == "liquidate" and i + 1 < len(candles):
                    fill_candle = candles[i + 1]
                    for open_sym, qty in broker.get_positions().items():
                        order = OrderRequest(symbol=open_sym, side="SELL", quantity=qty)
                        order_result = broker.submit_order(order, fill_candle)
                        if order_result.filled and order_result.fill_price is not None:
                            entry = entry_prices.pop(open_sym, order_result.fill_price)
                            pnl = (order_result.fill_price - entry) * order_result.quantity
                            if self._transaction_costs is not None:
                                pnl -= self._transaction_costs.total_cost(
                                    order_result.fill_price, order_result.quantity
                                )
                            pnl_pct = (
                                (order_result.fill_price - entry) / entry
                                if entry != 0
                                else Decimal(0)
                            )
                            trade = TradeResult(
                                signal_id=uuid4(),
                                symbol=open_sym,
                                side="SELL",
                                quantity=order_result.quantity,
                                entry_price=entry,
                                exit_price=order_result.fill_price,
                                pnl=pnl,
                                pnl_pct=pnl_pct,
                            )
                            trades.append(trade)
                            self._record_trade(trade)
                    snapshots.append(broker.get_portfolio())
                    continue

                # L2+: suppress new entries
                if cb_level in ("halted", "liquidate"):
                    snapshots.append(broker.get_portfolio())
                    continue

            # (c3) Check loss limits
            if self._loss_limits is not None:
                portfolio = broker.get_portfolio()
                if self._loss_limits.is_halted(candle.timestamp, portfolio.equity):
                    snapshots.append(broker.get_portfolio())
                    continue

            # (d) Generate signal from strategy
            history = candles[: i + 1]
            signal = self._strategy.generate_signal(symbol, history, segment_id)

            if signal is not None and i + 1 < len(candles):
                fill_candle = candles[i + 1]

                if signal.direction == SignalDirection.BUY:
                    self._handle_buy(broker, checker, fill_candle, symbol, history, entry_prices)

                elif signal.direction == SignalDirection.SELL:
                    self._handle_sell(broker, fill_candle, symbol, entry_prices, trades)

            # (e) Record portfolio snapshot
            snapshots.append(broker.get_portfolio())

        # Close any remaining open positions at the last candle's close price
        if candles:
            last_candle = candles[-1]
            for open_symbol, qty in broker.get_positions().items():
                close_price = last_candle.close
                entry = entry_prices.pop(open_symbol, close_price)
                pnl = (close_price - entry) * qty
                if self._transaction_costs is not None:
                    pnl -= self._transaction_costs.total_cost(close_price, qty)
                pnl_pct = (close_price - entry) / entry if entry != 0 else Decimal(0)
                trade = TradeResult(
                    signal_id=uuid4(),
                    symbol=open_symbol,
                    side="SELL",
                    quantity=qty,
                    entry_price=entry,
                    exit_price=close_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
                trades.append(trade)
                self._record_trade(trade)

        return trades, snapshots

    def _record_trade(self, trade: TradeResult) -> None:
        """Record a completed trade in the Rolling Kelly estimator."""
        if self._rolling_kelly is not None:
            self._rolling_kelly.update(TradeRecord(pnl=trade.pnl, pnl_pct=trade.pnl_pct))

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
        # Skip if a position is already open for this symbol
        if broker.has_position(symbol):
            return

        portfolio = broker.get_portfolio()

        # Compute position size: Rolling Kelly if available, else default Half-Kelly
        if self._rolling_kelly is not None:
            kelly_frac = self._rolling_kelly.optimal_fraction()
            position_value = min(
                portfolio.equity * kelly_frac,
                portfolio.equity * self._max_position_pct,
            )
        else:
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
        quantity = (position_value / fill_price).to_integral_value(rounding=ROUND_DOWN)
        if quantity <= 0:
            return

        order = OrderRequest(symbol=symbol, side="BUY", quantity=quantity)
        order_result = broker.submit_order(order, fill_candle)

        if order_result.filled and order_result.fill_price is not None:
            entry_prices[symbol] = order_result.fill_price

            # Deduct entry transaction costs from cash
            if self._transaction_costs is not None:
                cost = self._transaction_costs.total_cost(
                    order_result.fill_price, order_result.quantity
                )
                broker._cash -= cost

            # Set ATR stop-loss (trailing or fixed)
            stop_price = compute_atr_stop_loss(
                entry_price=order_result.fill_price,
                candles=history,
                atr_multiplier=self._atr_multiplier,
            )
            if stop_price is not None:
                # Compute ATR value for trailing stop
                atr_value = (
                    (order_result.fill_price - stop_price) / self._atr_multiplier
                    if self._atr_multiplier > 0
                    else Decimal(0)
                )
                broker.set_trailing_stop(
                    symbol=symbol,
                    entry_price=order_result.fill_price,
                    initial_stop=stop_price,
                    atr_value=atr_value,
                    activation_atr=self._trail_activation_atr,
                    trail_atr=self._trail_distance_atr,
                )

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
            if self._transaction_costs is not None:
                pnl -= self._transaction_costs.total_cost(
                    order_result.fill_price, order_result.quantity
                )
            pnl_pct = (order_result.fill_price - entry) / entry if entry != 0 else Decimal(0)
            trade = TradeResult(
                signal_id=uuid4(),
                symbol=symbol,
                side="SELL",
                quantity=order_result.quantity,
                entry_price=entry,
                exit_price=order_result.fill_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
            )
            trades.append(trade)
            self._record_trade(trade)

"""Backtest engine -- iterates candles and runs a strategy with risk management.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from datetime import UTC, datetime, time
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

from finalayze.backtest.config import BacktestConfig, resolve_max_hold_bars
from finalayze.backtest.decision_journal import (
    CandleSnapshot,
    DecisionJournal,
    FinalAction,
    StrategySignalRecord,
)
from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
from finalayze.core.schemas import (
    Candle,
    PortfolioState,
    Signal,
    SignalDirection,
    TradeResult,
)
from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.simulated_broker import SimulatedBroker
from finalayze.risk.chandelier_exit import compute_chandelier_stop, get_chandelier_multiplier
from finalayze.risk.kelly import TradeRecord
from finalayze.risk.position_sizer import (
    compute_position_size,
    compute_realized_vol,
    compute_vol_adjusted_position_size,
)
from finalayze.risk.pre_trade_check import PreTradeChecker
from finalayze.risk.stop_loss import compute_atr_stop_loss

if TYPE_CHECKING:
    from finalayze.backtest.costs import TransactionCosts
    from finalayze.risk.circuit_breaker import CircuitBreaker
    from finalayze.risk.kelly import RollingKelly
    from finalayze.risk.loss_limits import LossLimitTracker
    from finalayze.risk.regime import RegimeProvider
    from finalayze.strategies.base import BaseStrategy

# Default Half-Kelly parameters (used when no RollingKelly is provided)
_DEFAULT_WIN_RATE = Decimal("0.5")
_DEFAULT_AVG_WIN_RATIO = Decimal("1.5")

# Default market open time (US 9:30 ET = 14:30 UTC) used to adjust daily
# candle timestamps so the pre-trade market-hours check passes during backtest.
_US_MARKET_OPEN_UTC = time(14, 30, tzinfo=UTC)


class BacktestEngine:
    """Iterate candles and execute a strategy with risk management."""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_cash: Decimal = Decimal(100000),
        max_position_pct: Decimal = Decimal("0.20"),
        max_positions: int = 10,
        kelly_fraction: Decimal = Decimal("0.5"),
        atr_multiplier: Decimal = Decimal("3.0"),
        transaction_costs: TransactionCosts | None = None,
        trail_activation_atr: Decimal = Decimal("1.0"),
        trail_distance_atr: Decimal = Decimal("1.5"),
        circuit_breaker: CircuitBreaker | None = None,
        rolling_kelly: RollingKelly | None = None,
        loss_limits: LossLimitTracker | None = None,
        target_vol: Decimal | None = None,
        decision_journal: DecisionJournal | None = None,
        profit_target_atr: Decimal = Decimal("5.0"),
        max_hold_bars: int | dict[str, int] = 30,
        *,
        config: BacktestConfig | None = None,
        regime_provider: RegimeProvider | None = None,
    ) -> None:
        cfg = config or BacktestConfig(
            initial_cash=initial_cash,
            max_position_pct=max_position_pct,
            max_positions=max_positions,
            kelly_fraction=kelly_fraction,
            atr_multiplier=atr_multiplier,
            transaction_costs=transaction_costs,
            trail_activation_atr=trail_activation_atr,
            trail_distance_atr=trail_distance_atr,
            circuit_breaker=circuit_breaker,
            rolling_kelly=rolling_kelly,
            loss_limits=loss_limits,
            target_vol=target_vol,
            decision_journal=decision_journal,
            profit_target_atr=profit_target_atr,
            max_hold_bars=max_hold_bars,
        )
        self._config = cfg
        self._strategy = strategy
        self._initial_cash = cfg.initial_cash
        self._max_position_pct = cfg.max_position_pct
        self._max_positions = cfg.max_positions
        self._kelly_fraction = cfg.kelly_fraction
        self._atr_multiplier = cfg.atr_multiplier
        self._transaction_costs = cfg.transaction_costs
        self._trail_activation_atr = cfg.trail_activation_atr
        self._trail_distance_atr = cfg.trail_distance_atr
        self._circuit_breaker = cfg.circuit_breaker
        self._rolling_kelly = cfg.rolling_kelly
        self._loss_limits = cfg.loss_limits
        self._target_vol = cfg.target_vol
        self._decision_journal = cfg.decision_journal
        self._profit_target_atr = cfg.profit_target_atr
        self._max_hold_bars = cfg.max_hold_bars
        self._stop_loss_mode = cfg.stop_loss_mode
        self._regime_provider = regime_provider

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
        entry_bars: dict[str, int] = {}
        entry_strategies: dict[str, str] = {}
        chandelier_stops: dict[str, Decimal] = {}

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

            # (b2) Update Chandelier stops (monotonic ratchet)
            if self._stop_loss_mode == "chandelier" and symbol in chandelier_stops:
                history_so_far = candles[: i + 1]
                segment_mult = get_chandelier_multiplier(segment_id)
                candidate = compute_chandelier_stop(
                    history_so_far,
                    atr_period=22,
                    multiplier=segment_mult,
                )
                if candidate is not None:
                    new_stop = max(chandelier_stops[symbol], candidate)
                    chandelier_stops[symbol] = new_stop
                    # Update broker stop state to match
                    if symbol in broker._stop_states:
                        broker._stop_states[symbol].current_stop = new_stop

            # (c) Check stop-losses after prices are updated
            stop_results = broker.check_stop_losses(candle)
            for sr in stop_results:
                if sr.filled and sr.fill_price is not None:
                    entry = entry_prices.pop(sr.symbol, sr.fill_price)
                    entry_bars.pop(sr.symbol, None)
                    entry_strategies.pop(sr.symbol, None)
                    chandelier_stops.pop(sr.symbol, None)
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
                            entry_bars.pop(open_sym, None)
                            entry_strategies.pop(open_sym, None)
                            chandelier_stops.pop(open_sym, None)
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
                    self._journal_skip(
                        timestamp=candle.timestamp,
                        symbol=symbol,
                        segment_id=segment_id,
                        broker=broker,
                        history=candles[: i + 1],
                        skip_reason=f"circuit_breaker_{cb_level}",
                        cb_level=str(cb_level),
                    )
                    snapshots.append(broker.get_portfolio())
                    continue

            # (c3) Check loss limits
            if self._loss_limits is not None:
                portfolio = broker.get_portfolio()
                if self._loss_limits.is_halted(candle.timestamp, portfolio.equity):
                    self._journal_skip(
                        timestamp=candle.timestamp,
                        symbol=symbol,
                        segment_id=segment_id,
                        broker=broker,
                        history=candles[: i + 1],
                        skip_reason="loss_limit_halted",
                    )
                    snapshots.append(broker.get_portfolio())
                    continue

            # (c4) Check profit target
            if (
                self._profit_target_atr > 0
                and symbol in entry_prices
                and broker.has_position(symbol)
                and i + 1 < len(candles)
            ):
                entry_atr = broker.get_entry_atr(symbol)
                if entry_atr is not None and entry_atr > 0:
                    target_price = entry_prices[symbol] + self._profit_target_atr * entry_atr
                    if candle.high >= target_price:
                        fill_candle = candles[i + 1]
                        held = broker.get_positions().get(symbol, Decimal(0))
                        if held > 0:
                            order = OrderRequest(symbol=symbol, side="SELL", quantity=held)
                            order_result = broker.submit_order(order, fill_candle)
                            if order_result.filled and order_result.fill_price is not None:
                                entry = entry_prices.pop(symbol, order_result.fill_price)
                                entry_bars.pop(symbol, None)
                                entry_strategies.pop(symbol, None)
                                chandelier_stops.pop(symbol, None)
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
                                self._journal_skip(
                                    timestamp=candle.timestamp,
                                    symbol=symbol,
                                    segment_id=segment_id,
                                    broker=broker,
                                    history=candles[: i + 1],
                                    skip_reason="profit_target",
                                )
                            snapshots.append(broker.get_portfolio())
                            continue

            # (c5) Check time-based exit (max holding period)
            effective_max_hold = self._resolve_hold_bars(symbol, entry_strategies)
            if (
                effective_max_hold > 0
                and symbol in entry_bars
                and broker.has_position(symbol)
                and i + 1 < len(candles)
            ):
                bars_held = i - entry_bars[symbol]
                if bars_held >= effective_max_hold:
                    fill_candle = candles[i + 1]
                    held = broker.get_positions().get(symbol, Decimal(0))
                    if held > 0:
                        order = OrderRequest(symbol=symbol, side="SELL", quantity=held)
                        order_result = broker.submit_order(order, fill_candle)
                        if order_result.filled and order_result.fill_price is not None:
                            entry = entry_prices.pop(symbol, order_result.fill_price)
                            entry_bars.pop(symbol, None)
                            entry_strategies.pop(symbol, None)
                            chandelier_stops.pop(symbol, None)
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
                            self._journal_skip(
                                timestamp=candle.timestamp,
                                symbol=symbol,
                                segment_id=segment_id,
                                broker=broker,
                                history=candles[: i + 1],
                                skip_reason="time_exit",
                            )
                        snapshots.append(broker.get_portfolio())
                        continue

            # (d) Query regime provider (if set)
            history = candles[: i + 1]
            regime_state = None
            if self._regime_provider is not None:
                regime_state = self._regime_provider.get_regime(history, i)

            # (e) Generate signal from strategy
            signal = self._strategy.generate_signal(
                symbol,
                history,
                segment_id,
                has_open_position=broker.has_position(symbol),
            )

            if signal is not None and i + 1 < len(candles):
                fill_candle = candles[i + 1]

                if signal.direction == SignalDirection.BUY:
                    # Skip BUY if regime blocks new longs
                    if regime_state is not None and not regime_state.allow_new_longs:
                        self._journal_skip(
                            timestamp=candle.timestamp,
                            symbol=symbol,
                            segment_id=segment_id,
                            broker=broker,
                            history=history,
                            skip_reason="regime_blocks_longs",
                        )
                        snapshots.append(broker.get_portfolio())
                        continue

                    self._handle_buy(
                        broker,
                        checker,
                        fill_candle,
                        symbol,
                        history,
                        entry_prices,
                        segment_id=segment_id,
                        signal=signal,
                        entry_bars=entry_bars,
                        bar_index=i,
                        regime_position_scale=(
                            regime_state.position_scale if regime_state is not None else None
                        ),
                        entry_strategies=entry_strategies,
                        chandelier_stops=chandelier_stops,
                    )

                elif signal.direction == SignalDirection.SELL:
                    self._handle_sell(
                        broker,
                        fill_candle,
                        symbol,
                        entry_prices,
                        trades,
                        segment_id=segment_id,
                        signal=signal,
                        history=history,
                        entry_bars=entry_bars,
                        entry_strategies=entry_strategies,
                        chandelier_stops=chandelier_stops,
                    )
            elif signal is None:
                self._journal_skip(
                    timestamp=candle.timestamp,
                    symbol=symbol,
                    segment_id=segment_id,
                    broker=broker,
                    history=history,
                    skip_reason="no_signal",
                )

            # (f) Record portfolio snapshot
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

    def run_portfolio(  # noqa: PLR0912, PLR0915
        self,
        symbols: list[str],
        segment_id: str,
        candles_by_symbol: dict[str, list[Candle]],
    ) -> tuple[list[TradeResult], list[PortfolioState]]:
        """Run a portfolio-level backtest over multiple symbols.

        Iterates through a unified timeline, generating signals for each symbol
        on each bar, managing shared capital across all positions.

        Args:
            symbols: List of ticker symbols to trade.
            segment_id: Market segment identifier.
            candles_by_symbol: Candle data keyed by symbol.

        Returns:
            A tuple of (trades, portfolio_snapshots).
        """
        if not symbols or not candles_by_symbol:
            return [], []

        checker = PreTradeChecker(
            max_position_pct=self._max_position_pct,
            max_positions_per_market=self._max_positions,
        )
        broker = SimulatedBroker(initial_cash=self._initial_cash)

        trades: list[TradeResult] = []
        snapshots: list[PortfolioState] = []
        entry_prices: dict[str, Decimal] = {}
        entry_bars: dict[str, int] = {}
        entry_strategies: dict[str, str] = {}
        chandelier_stops: dict[str, Decimal] = {}
        # Track bar count per symbol for time-based exit in portfolio mode
        bar_counts: dict[str, int] = {}

        # Build per-symbol candle index keyed by timestamp
        candle_index: dict[str, dict[datetime, int]] = {}
        for sym in symbols:
            candle_index[sym] = {}
            for i, c in enumerate(candles_by_symbol.get(sym, [])):
                candle_index[sym][c.timestamp] = i

        # Build unified timeline
        all_timestamps = sorted(
            {c.timestamp for candles in candles_by_symbol.values() for c in candles}
        )

        ts_index = 0
        for ts in all_timestamps:
            broker.set_timestamp(ts)

            # Update prices for all symbols that have data at this timestamp
            for sym in symbols:
                sym_candles = candles_by_symbol.get(sym, [])
                if sym in candle_index and ts in candle_index[sym]:
                    idx = candle_index[sym][ts]
                    broker.update_prices(sym_candles[idx])
                    bar_counts[sym] = bar_counts.get(sym, 0) + 1

            # Update Chandelier stops for all symbols in portfolio mode
            if self._stop_loss_mode == "chandelier":
                for sym in symbols:
                    if sym not in chandelier_stops:
                        continue
                    if sym not in candle_index or ts not in candle_index[sym]:
                        continue
                    sym_candles = candles_by_symbol.get(sym, [])
                    idx = candle_index[sym][ts]
                    history_so_far = sym_candles[: idx + 1]
                    segment_mult = get_chandelier_multiplier(segment_id)
                    candidate = compute_chandelier_stop(
                        history_so_far, atr_period=22, multiplier=segment_mult
                    )
                    if candidate is not None:
                        new_stop = max(chandelier_stops[sym], candidate)
                        chandelier_stops[sym] = new_stop
                        if sym in broker._stop_states:
                            broker._stop_states[sym].current_stop = new_stop

            # Check stop-losses for all symbols
            for sym in symbols:
                if sym in candle_index and ts in candle_index[sym]:
                    sym_candles = candles_by_symbol.get(sym, [])
                    idx = candle_index[sym][ts]
                    stop_results = broker.check_stop_losses(sym_candles[idx])
                    for sr in stop_results:
                        if sr.filled and sr.fill_price is not None:
                            entry = entry_prices.pop(sr.symbol, sr.fill_price)
                            entry_bars.pop(sr.symbol, None)
                            entry_strategies.pop(sr.symbol, None)
                            chandelier_stops.pop(sr.symbol, None)
                            pnl = (sr.fill_price - entry) * sr.quantity
                            if self._transaction_costs is not None:
                                pnl -= self._transaction_costs.total_cost(
                                    sr.fill_price, sr.quantity
                                )
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

            # Check profit target and time exit for all symbols
            for sym in symbols:
                if sym not in candle_index or ts not in candle_index[sym]:
                    continue
                sym_candles = candles_by_symbol.get(sym, [])
                idx = candle_index[sym][ts]
                candle = sym_candles[idx]

                if not broker.has_position(sym) or sym not in entry_prices:
                    continue

                # Profit target check
                if self._profit_target_atr > 0 and idx + 1 < len(sym_candles):
                    entry_atr = broker.get_entry_atr(sym)
                    if entry_atr is not None and entry_atr > 0:
                        target_price = entry_prices[sym] + self._profit_target_atr * entry_atr
                        if candle.high >= target_price:
                            fill_candle = sym_candles[idx + 1]
                            held = broker.get_positions().get(sym, Decimal(0))
                            if held > 0:
                                order = OrderRequest(symbol=sym, side="SELL", quantity=held)
                                order_result = broker.submit_order(order, fill_candle)
                                if order_result.filled and order_result.fill_price is not None:
                                    entry = entry_prices.pop(sym, order_result.fill_price)
                                    entry_bars.pop(sym, None)
                                    entry_strategies.pop(sym, None)
                                    chandelier_stops.pop(sym, None)
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
                                        symbol=sym,
                                        side="SELL",
                                        quantity=order_result.quantity,
                                        entry_price=entry,
                                        exit_price=order_result.fill_price,
                                        pnl=pnl,
                                        pnl_pct=pnl_pct,
                                    )
                                    trades.append(trade)
                                    self._record_trade(trade)
                                continue

                # Time-based exit check
                effective_max_hold = self._resolve_hold_bars(sym, entry_strategies)
                if effective_max_hold > 0 and sym in entry_bars and idx + 1 < len(sym_candles):
                    bars_since_entry = bar_counts.get(sym, 0) - entry_bars.get(sym, 0)
                    if bars_since_entry >= effective_max_hold:
                        fill_candle = sym_candles[idx + 1]
                        held = broker.get_positions().get(sym, Decimal(0))
                        if held > 0:
                            order = OrderRequest(symbol=sym, side="SELL", quantity=held)
                            order_result = broker.submit_order(order, fill_candle)
                            if order_result.filled and order_result.fill_price is not None:
                                entry = entry_prices.pop(sym, order_result.fill_price)
                                entry_bars.pop(sym, None)
                                entry_strategies.pop(sym, None)
                                chandelier_stops.pop(sym, None)
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
                                    symbol=sym,
                                    side="SELL",
                                    quantity=order_result.quantity,
                                    entry_price=entry,
                                    exit_price=order_result.fill_price,
                                    pnl=pnl,
                                    pnl_pct=pnl_pct,
                                )
                                trades.append(trade)
                                self._record_trade(trade)
                            continue

            # Generate signals for each symbol
            for sym in symbols:
                sym_candles = candles_by_symbol.get(sym, [])
                if sym not in candle_index or ts not in candle_index[sym]:
                    continue
                idx = candle_index[sym][ts]

                history = sym_candles[: idx + 1]

                # Query regime provider for this symbol's history
                regime_state = None
                if self._regime_provider is not None:
                    regime_state = self._regime_provider.get_regime(history, idx)

                signal = self._strategy.generate_signal(
                    sym,
                    history,
                    segment_id,
                    has_open_position=broker.has_position(sym),
                )

                if signal is not None and idx + 1 < len(sym_candles):
                    fill_candle = sym_candles[idx + 1]

                    if signal.direction == SignalDirection.BUY:
                        # Skip BUY if regime blocks new longs
                        if regime_state is not None and not regime_state.allow_new_longs:
                            continue

                        self._handle_buy(
                            broker,
                            checker,
                            fill_candle,
                            sym,
                            history,
                            entry_prices,
                            segment_id=segment_id,
                            signal=signal,
                            entry_bars=entry_bars,
                            bar_index=bar_counts.get(sym, 0),
                            regime_position_scale=(
                                regime_state.position_scale if regime_state is not None else None
                            ),
                            entry_strategies=entry_strategies,
                            chandelier_stops=chandelier_stops,
                        )
                    elif signal.direction == SignalDirection.SELL:
                        self._handle_sell(
                            broker,
                            fill_candle,
                            sym,
                            entry_prices,
                            trades,
                            entry_bars=entry_bars,
                            entry_strategies=entry_strategies,
                            chandelier_stops=chandelier_stops,
                        )

            snapshots.append(broker.get_portfolio())
            ts_index += 1

        # Close remaining open positions
        if candles_by_symbol:
            for sym in symbols:
                sym_candles = candles_by_symbol.get(sym, [])
                if not sym_candles:
                    continue
                qty = broker.get_positions().get(sym, Decimal(0))
                if qty <= 0:
                    continue
                close_price = sym_candles[-1].close
                entry = entry_prices.pop(sym, close_price)
                pnl = (close_price - entry) * qty
                if self._transaction_costs is not None:
                    pnl -= self._transaction_costs.total_cost(close_price, qty)
                pnl_pct = (close_price - entry) / entry if entry != 0 else Decimal(0)
                trade = TradeResult(
                    signal_id=uuid4(),
                    symbol=sym,
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

    def _resolve_hold_bars(
        self,
        symbol: str,
        entry_strategies: dict[str, str],
    ) -> int:
        """Resolve the effective max hold bars for a given symbol's position.

        Uses the strategy name that opened the position to look up
        per-strategy hold limits when ``max_hold_bars`` is a dict.
        """
        strategy_name = entry_strategies.get(symbol, "")
        return resolve_max_hold_bars(self._max_hold_bars, strategy_name)

    def _record_trade(self, trade: TradeResult) -> None:
        """Record a completed trade in the Rolling Kelly estimator."""
        if self._rolling_kelly is not None:
            self._rolling_kelly.update(TradeRecord(pnl=trade.pnl, pnl_pct=trade.pnl_pct))

    def _journal_decision(
        self,
        *,
        action: FinalAction,
        timestamp: datetime,
        symbol: str,
        segment_id: str,
        broker: SimulatedBroker,
        history: list[Candle] | None = None,
        signal: Signal | None = None,
        skip_reason: str | None = None,
        pre_trade_passed: bool | None = None,
        pre_trade_violations: list[str] | None = None,
        position_value: Decimal | None = None,
        quantity: Decimal | None = None,
        fill_price: Decimal | None = None,
        stop_loss_price: Decimal | None = None,
        cb_level: str = "normal",
    ) -> None:
        """Record a decision in the journal (no-op if journal is None)."""
        if self._decision_journal is None:
            return

        portfolio = broker.get_portfolio()

        # Build recent candle snapshots (last 5)
        recent: list[CandleSnapshot] = [
            CandleSnapshot(
                timestamp=c.timestamp,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
            )
            for c in (history[-5:] if history else [])
        ]

        # Extract per-strategy signals if using JournalingStrategyCombiner
        strategy_signals: list[StrategySignalRecord] = []
        net_score: float | None = None
        if isinstance(self._strategy, JournalingStrategyCombiner):
            for name, sig in self._strategy.last_signals.items():
                weight = self._strategy.last_weights.get(name, Decimal("1.0"))
                if sig is not None:
                    dir_score = Decimal(1) if sig.direction == SignalDirection.BUY else Decimal(-1)
                    contribution = dir_score * Decimal(str(sig.confidence)) * weight
                    strategy_signals.append(
                        StrategySignalRecord(
                            strategy_name=name,
                            direction=sig.direction.value,
                            confidence=sig.confidence,
                            weight=weight,
                            contribution=contribution,
                        )
                    )
                else:
                    strategy_signals.append(
                        StrategySignalRecord(
                            strategy_name=name,
                            direction=None,
                            confidence=None,
                            weight=weight,
                            contribution=Decimal(0),
                        )
                    )
            net_score = self._strategy.last_net_score

        # Identify the strategy with the highest absolute contribution
        dominant: str | None = None
        if strategy_signals:
            firing = [s for s in strategy_signals if s.direction is not None]
            if firing:
                dominant = max(firing, key=lambda s: abs(s.contribution)).strategy_name

        self._decision_journal.record(
            self._decision_journal.make_record(
                timestamp=timestamp,
                symbol=symbol,
                segment_id=segment_id,
                final_action=action,
                skip_reason=skip_reason,
                strategy_signals=strategy_signals,
                combined_direction=signal.direction.value if signal else None,
                combined_confidence=signal.confidence if signal else None,
                net_weighted_score=net_score,
                dominant_strategy=dominant,
                pre_trade_passed=pre_trade_passed,
                pre_trade_violations=pre_trade_violations or [],
                position_value=position_value,
                quantity=quantity,
                fill_price=fill_price,
                stop_loss_price=stop_loss_price,
                circuit_breaker_level=cb_level,
                portfolio_equity=portfolio.equity,
                portfolio_cash=portfolio.cash,
                open_position_count=len(portfolio.positions),
                recent_candles=recent,
            )
        )

    def _journal_skip(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        segment_id: str,
        broker: SimulatedBroker,
        history: list[Candle] | None = None,
        skip_reason: str,
        cb_level: str = "normal",
    ) -> None:
        """Convenience wrapper for journaling a SKIP decision."""
        if self._decision_journal is None:
            return
        self._journal_decision(
            action=FinalAction.SKIP,
            timestamp=timestamp,
            symbol=symbol,
            segment_id=segment_id,
            broker=broker,
            history=history,
            skip_reason=skip_reason,
            cb_level=cb_level,
        )

    def _handle_buy(  # noqa: PLR0912, PLR0915
        self,
        broker: SimulatedBroker,
        checker: PreTradeChecker,
        fill_candle: Candle,
        symbol: str,
        history: list[Candle],
        entry_prices: dict[str, Decimal],
        segment_id: str = "",
        signal: Signal | None = None,
        entry_bars: dict[str, int] | None = None,
        bar_index: int = 0,
        regime_position_scale: float | None = None,
        entry_strategies: dict[str, str] | None = None,
        chandelier_stops: dict[str, Decimal] | None = None,
    ) -> None:
        """Process a BUY signal: size, check, fill, stop-loss."""
        # Skip if a position is already open for this symbol
        if broker.has_position(symbol):
            self._journal_skip(
                timestamp=fill_candle.timestamp,
                symbol=symbol,
                segment_id=segment_id,
                broker=broker,
                history=history,
                skip_reason="position_already_open",
            )
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

        # Apply volatility-adjusted sizing if target_vol is configured
        if self._target_vol is not None:
            asset_vol = compute_realized_vol(history)
            if asset_vol is not None and asset_vol > 0:
                position_value = compute_vol_adjusted_position_size(
                    base_position=position_value,
                    target_vol=self._target_vol,
                    asset_vol=asset_vol,
                )

        # Apply confidence scaling from signal
        if signal is not None:
            confidence_scale = Decimal(str(0.5 + signal.confidence * 0.5))  # [0.5x, 1.0x]
            position_value = position_value * confidence_scale

        # Apply regime-based position scaling
        if regime_position_scale is not None:
            position_value = position_value * Decimal(str(regime_position_scale))

        # Hard cap at max_position_pct
        max_allowed = portfolio.equity * self._max_position_pct
        position_value = min(position_value, max_allowed)

        if position_value <= 0:
            self._journal_skip(
                timestamp=fill_candle.timestamp,
                symbol=symbol,
                segment_id=segment_id,
                broker=broker,
                history=history,
                skip_reason="position_value_zero",
            )
            return

        # Pre-trade check — adjust daily candle timestamps (midnight UTC) to
        # market-open time so the market-hours check passes during backtest.
        check_dt = fill_candle.timestamp
        if check_dt.hour == 0 and check_dt.minute == 0:
            check_dt = datetime.combine(check_dt.date(), _US_MARKET_OPEN_UTC)
        result = checker.check(
            order_value=position_value,
            portfolio_equity=portfolio.equity,
            available_cash=portfolio.cash,
            open_position_count=len(portfolio.positions),
            dt=check_dt,
        )
        if not result.passed:
            if self._decision_journal is not None:
                self._journal_decision(
                    action=FinalAction.SKIP,
                    timestamp=fill_candle.timestamp,
                    symbol=symbol,
                    segment_id=segment_id,
                    broker=broker,
                    history=history,
                    signal=signal,
                    skip_reason="pre_trade_check_failed",
                    pre_trade_passed=False,
                    pre_trade_violations=result.violations,
                    position_value=position_value,
                )
            return

        # Compute quantity at fill price
        fill_price = fill_candle.open
        if fill_price <= 0:
            self._journal_skip(
                timestamp=fill_candle.timestamp,
                symbol=symbol,
                segment_id=segment_id,
                broker=broker,
                history=history,
                skip_reason="fill_price_zero",
            )
            return
        quantity = (position_value / fill_price).to_integral_value(rounding=ROUND_DOWN)
        if quantity <= 0:
            self._journal_skip(
                timestamp=fill_candle.timestamp,
                symbol=symbol,
                segment_id=segment_id,
                broker=broker,
                history=history,
                skip_reason="quantity_zero",
            )
            return

        # Pre-compute ATR stop-loss — reject trade if stop cannot be computed
        stop_price = compute_atr_stop_loss(
            entry_price=fill_price,
            candles=history,
            atr_multiplier=self._atr_multiplier,
        )
        if stop_price is None:
            self._journal_skip(
                timestamp=fill_candle.timestamp,
                symbol=symbol,
                segment_id=segment_id,
                broker=broker,
                history=history,
                skip_reason="no_stop_loss_data",
            )
            return

        order = OrderRequest(symbol=symbol, side="BUY", quantity=quantity)
        order_result = broker.submit_order(order, fill_candle)

        if order_result.filled and order_result.fill_price is not None:
            entry_prices[symbol] = order_result.fill_price
            if entry_bars is not None:
                entry_bars[symbol] = bar_index
            if entry_strategies is not None and signal is not None:
                entry_strategies[symbol] = signal.strategy_name

            # Journal the successful BUY (with stop-loss price)
            if self._decision_journal is not None:
                self._journal_decision(
                    action=FinalAction.BUY,
                    timestamp=fill_candle.timestamp,
                    symbol=symbol,
                    segment_id=segment_id,
                    broker=broker,
                    history=history,
                    signal=signal,
                    pre_trade_passed=True,
                    position_value=position_value,
                    quantity=quantity,
                    fill_price=order_result.fill_price,
                    stop_loss_price=stop_price,
                )

            # Deduct entry transaction costs from cash
            if self._transaction_costs is not None:
                cost = self._transaction_costs.total_cost(
                    order_result.fill_price, order_result.quantity
                )
                broker.deduct_fees(cost)

            # Set stop-loss based on mode
            atr_value = (
                (order_result.fill_price - stop_price) / self._atr_multiplier
                if self._atr_multiplier > 0
                else Decimal(0)
            )

            if self._stop_loss_mode == "chandelier":
                # Chandelier mode: compute initial stop from highest_high - mult * ATR
                segment_mult = get_chandelier_multiplier(segment_id)
                ch_stop = compute_chandelier_stop(history, atr_period=22, multiplier=segment_mult)
                initial_stop = ch_stop if ch_stop is not None else stop_price
                if chandelier_stops is not None:
                    chandelier_stops[symbol] = initial_stop
                # Use a fixed stop in the broker (chandelier ratchet is managed
                # externally in the main loop via chandelier_stops dict)
                broker.set_stop_loss(symbol, initial_stop)
            else:
                # Default trailing stop mode
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
        segment_id: str = "",
        signal: Signal | None = None,
        history: list[Candle] | None = None,
        entry_bars: dict[str, int] | None = None,
        entry_strategies: dict[str, str] | None = None,
        chandelier_stops: dict[str, Decimal] | None = None,
    ) -> None:
        """Process a SELL signal: sell all held quantity."""
        portfolio = broker.get_portfolio()
        held = portfolio.positions.get(symbol, Decimal(0))
        if held <= 0:
            self._journal_skip(
                timestamp=fill_candle.timestamp,
                symbol=symbol,
                segment_id=segment_id,
                broker=broker,
                history=history,
                skip_reason="no_position_held",
            )
            return

        order = OrderRequest(symbol=symbol, side="SELL", quantity=held)
        order_result = broker.submit_order(order, fill_candle)

        if order_result.filled and order_result.fill_price is not None:
            entry = entry_prices.pop(symbol, order_result.fill_price)
            if entry_bars is not None:
                entry_bars.pop(symbol, None)
            if entry_strategies is not None:
                entry_strategies.pop(symbol, None)
            if chandelier_stops is not None:
                chandelier_stops.pop(symbol, None)
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

            # Journal the successful SELL
            if self._decision_journal is not None:
                self._journal_decision(
                    action=FinalAction.SELL,
                    timestamp=fill_candle.timestamp,
                    symbol=symbol,
                    segment_id=segment_id,
                    broker=broker,
                    history=history,
                    signal=signal,
                    quantity=order_result.quantity,
                    fill_price=order_result.fill_price,
                )

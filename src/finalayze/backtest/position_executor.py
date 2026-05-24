"""Backtest position executor -- BUY / SELL / close logic.

Extracts the order-execution methods from ``BacktestEngine`` so that
``engine.py`` focuses on the main run-loop orchestration.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from finalayze.backtest.config import resolve_stop_atr_multiplier
from finalayze.backtest.decision_journal import FinalAction
from finalayze.core.schemas import Signal, TradeResult
from finalayze.execution.broker_base import OrderRequest
from finalayze.risk.chandelier_exit import get_chandelier_multiplier
from finalayze.risk.kelly import TradeRecord
from finalayze.risk.position_sizer import (
    compute_position_size,
    compute_realized_vol,
)
from finalayze.risk.position_sizing_pipeline import SizingContext
from finalayze.risk.pre_trade_check import CheckContext
from finalayze.risk.stop_loss import compute_atr_stop_loss

if TYPE_CHECKING:
    from datetime import time

    from finalayze.backtest.costs import TransactionCosts
    from finalayze.backtest.journal import BacktestJournal
    from finalayze.core.schemas import Candle
    from finalayze.execution.simulated_broker import SimulatedBroker
    from finalayze.ml.meta_labeler import MetaLabeler
    from finalayze.risk.kelly import RollingKelly
    from finalayze.risk.position_sizing_pipeline import PositionSizingPipeline
    from finalayze.risk.pre_trade_check import PreTradeChecker

logger = structlog.get_logger(__name__)

# Default Half-Kelly parameters (used when no RollingKelly is provided)
_DEFAULT_WIN_RATE = Decimal("0.5")
_DEFAULT_AVG_WIN_RATIO = Decimal("1.5")

# Default cap on order size as a fraction of the fill candle's volume.
# 5% of ADV is a standard liquidity-aware sizing cap used by institutional
# execution desks; larger orders get split over multiple bars.
_DEFAULT_MAX_ORDER_VOLUME_PCT = Decimal("0.05")


class BacktestPositionExecutor:
    """Handles BUY / SELL order execution and trade recording.

    This class encapsulates the position-management logic that was previously
    inlined in ``BacktestEngine._handle_buy``, ``_handle_sell``,
    ``_close_position``, and ``_record_trade``.

    It does **not** own the mutable position-tracking dicts (``entry_prices``,
    ``entry_bars``, etc.) -- those are passed in from the engine's run loop.
    """

    def __init__(
        self,
        *,
        journal: BacktestJournal,
        rolling_kelly: RollingKelly | None,
        kelly_fraction: Decimal,
        max_position_pct: Decimal,
        max_positions_per_segment: int,
        sizing_pipeline: PositionSizingPipeline | None,
        transaction_costs: TransactionCosts | None,
        trail_activation_atr: Decimal,
        trail_distance_atr: Decimal,
        stop_loss_mode: str,
        exclude_periods: tuple[tuple[str, str], ...],
        profit_target_atr: Decimal,
        target_vol: Decimal | None,
        meta_labeler: MetaLabeler | None,
        correlation_cache: dict[tuple[str, str], float],
        portfolio_returns: list[float],
        us_market_open_utc: time,
        moex_market_open_utc: time,
        max_order_volume_pct: Decimal = _DEFAULT_MAX_ORDER_VOLUME_PCT,
    ) -> None:
        self._journal = journal
        self._rolling_kelly = rolling_kelly
        self._kelly_fraction = kelly_fraction
        self._max_position_pct = max_position_pct
        self._max_positions_per_segment = max_positions_per_segment
        self._sizing_pipeline = sizing_pipeline
        self._transaction_costs = transaction_costs
        self._trail_activation_atr = trail_activation_atr
        self._trail_distance_atr = trail_distance_atr
        self._stop_loss_mode = stop_loss_mode
        self._exclude_periods = exclude_periods
        self._profit_target_atr = profit_target_atr
        self._target_vol = target_vol
        self._meta_labeler = meta_labeler
        self._correlation_cache = correlation_cache
        self._portfolio_returns = portfolio_returns
        self._us_market_open_utc = us_market_open_utc
        self._moex_market_open_utc = moex_market_open_utc
        self._max_order_volume_pct = max_order_volume_pct

    # ------------------------------------------------------------------
    # Mutable property setters -- the engine updates these between runs
    # ------------------------------------------------------------------

    def set_sizing_pipeline(self, pipeline: PositionSizingPipeline) -> None:
        """Update the sizing pipeline (rebuilt per run)."""
        self._sizing_pipeline = pipeline

    def set_correlation_cache(self, cache: dict[tuple[str, str], float]) -> None:
        """Update the correlation cache (rebuilt periodically)."""
        self._correlation_cache = cache

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradeResult) -> None:
        """Record a completed trade in the Rolling Kelly estimator."""
        if self._rolling_kelly is not None:
            self._rolling_kelly.update(TradeRecord(pnl=trade.pnl, pnl_pct=trade.pnl_pct))

    def close_position(
        self,
        *,
        symbol: str,
        exit_price: Decimal,
        quantity: Decimal,
        entry_prices: dict[str, Decimal],
        entry_bars: dict[str, int],
        entry_strategies: dict[str, str],
        chandelier_stops: dict[str, Decimal],
        bar_index: int,
        trades: list[TradeResult],
    ) -> TradeResult:
        """Close a position and record the trade.

        Mutates *entry_prices*, *entry_bars*, *entry_strategies*, and
        *chandelier_stops* by popping the key for *symbol*.  The resulting
        ``TradeResult`` is appended to *trades* and recorded in the Rolling
        Kelly estimator.
        """
        entry = entry_prices.pop(symbol, exit_price)
        entry_bar = entry_bars.pop(symbol, bar_index)
        entry_strategies.pop(symbol, None)
        chandelier_stops.pop(symbol, None)

        pnl = (exit_price - entry) * quantity
        if self._transaction_costs is not None:
            pnl -= self._transaction_costs.total_cost(exit_price, quantity)
        pnl_pct = (exit_price - entry) / entry if entry != 0 else Decimal(0)

        trade = TradeResult(
            signal_id=uuid4(),
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            entry_price=entry,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_bars=bar_index - entry_bar,
        )
        trades.append(trade)
        self.record_trade(trade)
        return trade

    # ------------------------------------------------------------------
    # BUY execution
    # ------------------------------------------------------------------

    def handle_buy(  # noqa: PLR0911, PLR0912, PLR0915
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
        from finalayze.backtest.risk_evaluator import BacktestRiskEvaluator  # noqa: PLC0415

        # Skip if a position is already open for this symbol
        if broker.has_position(symbol):
            self._journal.record_skip(
                timestamp=fill_candle.timestamp,
                symbol=symbol,
                segment_id=segment_id,
                broker=broker,
                history=history,
                skip_reason="position_already_open",
            )
            return

        # Check segment position cap
        segment_count = sum(1 for _s, qty in broker.get_positions().items() if qty > 0)
        if segment_count >= self._max_positions_per_segment:
            self._journal.record_skip(
                timestamp=fill_candle.timestamp,
                symbol=symbol,
                segment_id=segment_id,
                broker=broker,
                history=history,
                skip_reason="segment_position_cap",
            )
            return

        portfolio = broker.get_portfolio()

        # Compute position size via the unified sizing pipeline
        if self._rolling_kelly is not None:
            kelly_frac = self._rolling_kelly.optimal_fraction()
            base_position = portfolio.equity * kelly_frac
        else:
            base_position = compute_position_size(
                win_rate=_DEFAULT_WIN_RATE,
                avg_win_ratio=_DEFAULT_AVG_WIN_RATIO,
                equity=portfolio.equity,
                kelly_fraction=self._kelly_fraction,
                max_position_pct=self._max_position_pct,
            )

        asset_vol = compute_realized_vol(history) or Decimal("0.20")
        # Currency-aware min position: original thresholds scaled down for small portfolios
        if segment_id.startswith("ru_"):
            min_pos = min(Decimal(5000), max(Decimal(1000), portfolio.equity * Decimal("0.02")))
        else:
            min_pos = min(Decimal(500), max(Decimal(100), portfolio.equity * Decimal("0.005")))

        # Compute ML confidence from MetaLabeler if available
        ml_confidence: float | None = None
        if self._meta_labeler is not None and signal is not None:
            ml_confidence = self._meta_labeler.predict(signal, signal.features)

        context = SizingContext(
            equity=portfolio.equity,
            base_position=base_position,
            max_position_pct=self._max_position_pct,
            min_position_size=min_pos,
            asset_vol=asset_vol,
            target_vol=self._target_vol or Decimal("0.15"),
            regime_scale=Decimal(str(regime_position_scale or 1.0)),
            correlation_scale=Decimal("1.0"),
            returns_history=tuple(self._portfolio_returns),
            ml_confidence=ml_confidence,
        )
        if self._sizing_pipeline is None:
            return
        position_value = self._sizing_pipeline.compute(context)

        # Apply confidence scaling from signal (post-pipeline multiplier)
        if signal is not None:
            confidence_scale = Decimal(str(0.5 + signal.confidence * 0.5))  # [0.5x, 1.0x]
            position_value = position_value * confidence_scale

        if position_value <= 0:
            self._journal.record_skip(
                timestamp=fill_candle.timestamp,
                symbol=symbol,
                segment_id=segment_id,
                broker=broker,
                history=history,
                skip_reason="position_value_zero",
            )
            return

        # Pre-trade check -- adjust daily candle timestamps (midnight UTC) to
        # market-open time so the market-hours check passes during backtest.
        from datetime import datetime  # noqa: PLC0415

        check_dt = fill_candle.timestamp
        if check_dt.hour == 0 and check_dt.minute == 0:
            if segment_id.startswith("ru_"):
                check_dt = datetime.combine(check_dt.date(), self._moex_market_open_utc)
            else:
                check_dt = datetime.combine(check_dt.date(), self._us_market_open_utc)
        market_id = "moex" if segment_id.startswith("ru_") else "us"
        result = checker.check(
            CheckContext(
                order_value=position_value,
                portfolio_equity=portfolio.equity,
                available_cash=portfolio.cash,
                open_position_count=len(portfolio.positions),
                dt=check_dt,
                market_id=market_id,
                symbol=symbol,
                open_positions=list(broker.get_positions().keys()),
                strategy_name=signal.strategy_name if signal is not None else None,
                sector_id=segment_id,
                sector_exposure_value=BacktestRiskEvaluator.compute_segment_exposure(
                    broker, segment_id
                ),
                correlations=self._correlation_cache or None,
            )
        )
        if not result.passed:
            if self._journal.has_journal:
                self._journal.record_decision(
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
            self._journal.record_skip(
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
            self._journal.record_skip(
                timestamp=fill_candle.timestamp,
                symbol=symbol,
                segment_id=segment_id,
                broker=broker,
                history=history,
                skip_reason="quantity_zero",
            )
            return

        # Liquidity cap: never exceed max_order_volume_pct of the fill bar's volume.
        # Large orders relative to ADV are unrealistic at the open price; we clamp
        # rather than reject so strategies still participate, just at reduced size.
        if self._max_order_volume_pct > 0 and fill_candle.volume > 0:
            volume_cap = (
                Decimal(fill_candle.volume) * self._max_order_volume_pct
            ).to_integral_value(rounding=ROUND_DOWN)
            if quantity > volume_cap:
                logger.info(
                    "backtest_order_volume_capped",
                    symbol=symbol,
                    requested=str(quantity),
                    capped=str(volume_cap),
                    bar_volume=fill_candle.volume,
                    cap_pct=str(self._max_order_volume_pct),
                )
                quantity = volume_cap
            if quantity <= 0:
                self._journal.record_skip(
                    timestamp=fill_candle.timestamp,
                    symbol=symbol,
                    segment_id=segment_id,
                    broker=broker,
                    history=history,
                    skip_reason="volume_cap_zero",
                )
                return

        # Pre-compute ATR stop-loss -- use strategy-specific multiplier
        strategy_name = signal.strategy_name if signal is not None else ""
        stop_atr_mult = resolve_stop_atr_multiplier(strategy_name, segment_id=segment_id)
        stop_price = compute_atr_stop_loss(
            entry_price=fill_price,
            candles=history,
            atr_multiplier=stop_atr_mult,
            exclude_periods=self._exclude_periods,
        )
        if stop_price is None:
            self._journal.record_skip(
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
            if self._journal.has_journal:
                self._journal.record_decision(
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
            # Use the strategy-specific multiplier to recover correct ATR value
            atr_value = (
                (order_result.fill_price - stop_price) / stop_atr_mult
                if stop_atr_mult > 0
                else Decimal(0)
            )

            if self._stop_loss_mode == "chandelier":
                # Chandelier mode: use ATR-based stop as initial (guaranteed below
                # entry price), then let chandelier ratchet take over on subsequent bars.
                segment_mult = get_chandelier_multiplier(segment_id)
                initial_stop = compute_atr_stop_loss(
                    entry_price=order_result.fill_price,
                    candles=history,
                    atr_period=22,
                    atr_multiplier=Decimal(str(segment_mult)),
                    exclude_periods=self._exclude_periods,
                )
                if initial_stop is not None:
                    if chandelier_stops is not None:
                        chandelier_stops[symbol] = initial_stop
                    broker.set_stop_loss(symbol, initial_stop)
                else:
                    if chandelier_stops is not None:
                        chandelier_stops[symbol] = stop_price
                    broker.set_stop_loss(symbol, stop_price)
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

    # ------------------------------------------------------------------
    # SELL execution
    # ------------------------------------------------------------------

    def handle_sell(
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
        bar_index: int = 0,
    ) -> None:
        """Process a SELL signal: sell all held quantity."""
        portfolio = broker.get_portfolio()
        held = portfolio.positions.get(symbol, Decimal(0))
        if held <= 0:
            self._journal.record_skip(
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
            # Use empty dicts as fallback when optional tracking dicts are None
            _eb = entry_bars if entry_bars is not None else {}
            _es = entry_strategies if entry_strategies is not None else {}
            _cs = chandelier_stops if chandelier_stops is not None else {}
            self.close_position(
                symbol=symbol,
                exit_price=order_result.fill_price,
                quantity=order_result.quantity,
                entry_prices=entry_prices,
                entry_bars=_eb,
                entry_strategies=_es,
                chandelier_stops=_cs,
                bar_index=bar_index,
                trades=trades,
            )

            # Journal the successful SELL
            if self._journal.has_journal:
                self._journal.record_decision(
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

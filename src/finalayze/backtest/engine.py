"""Backtest engine -- iterates candles and runs a strategy with risk management.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, time
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from finalayze.backtest.config import (
    BacktestConfig,
    resolve_max_hold_bars,
    resolve_stop_atr_multiplier,
)
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
from finalayze.risk.kelly import RollingKelly, TradeRecord
from finalayze.risk.position_sizer import (
    compute_position_size,
    compute_realized_vol,
)
from finalayze.risk.position_sizing_pipeline import (
    BrentGateStep,
    CBRRegimeStep,
    CopulaStep,
    EVTStep,
    HardCapsStep,
    KellyStep,
    MetaLabelStep,
    PositionSizingPipeline,
    RegimeStep,
    RubOilRegimeStep,
    SectorAllocationStep,
    SizingContext,
    VolTargetStep,
)
from finalayze.risk.pre_trade_check import PreTradeChecker
from finalayze.risk.stop_loss import compute_atr_stop_loss, filter_candles_by_exclusion

if TYPE_CHECKING:
    from finalayze.backtest.costs import TransactionCosts
    from finalayze.risk.circuit_breaker import CircuitBreaker
    from finalayze.risk.loss_limits import LossLimitTracker
    from finalayze.risk.regime import RegimeProvider
    from finalayze.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)

# Sentinel value for "no entry bar recorded".  Used instead of a bare ``-2``
# so grep can find every reference and the meaning is self-documenting.
_NO_ENTRY_BAR = -2

# 15% intraday drop forces stop even on grace bar.  Quant-validated: 10% is
# too tight for earnings gaps; 15% corresponds to a 3+ sigma daily move.
_CATASTROPHIC_DROP_PCT = Decimal("0.15")

# Default Half-Kelly parameters (used when no RollingKelly is provided)
_DEFAULT_WIN_RATE = Decimal("0.5")
_DEFAULT_AVG_WIN_RATIO = Decimal("1.5")

# Default market open time (US 9:30 ET = 14:30 UTC) used to adjust daily
# candle timestamps so the pre-trade market-hours check passes during backtest.
_US_MARKET_OPEN_UTC = time(14, 30, tzinfo=UTC)

# MOEX market open time (10:00 MSK = 07:00 UTC) used to adjust daily
# candle timestamps for pre-trade market-hours check during backtest.
_MOEX_MARKET_OPEN_UTC = time(7, 0, tzinfo=UTC)


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
        # Use RollingKelly only when explicitly provided.  Default 1% cold-start
        # is too small for small per-symbol allocations (e.g. 50K RUB), causing the
        # sizing pipeline to zero all positions and creating a permanent deadlock.
        self._rolling_kelly = cfg.rolling_kelly
        self._loss_limits = cfg.loss_limits
        self._target_vol = cfg.target_vol
        self._decision_journal = cfg.decision_journal
        self._profit_target_atr = cfg.profit_target_atr
        self._max_hold_bars = cfg.max_hold_bars
        self._stop_loss_mode = cfg.stop_loss_mode
        self._exclude_periods = cfg.exclude_periods
        self._regime_provider = regime_provider
        self._meta_labeler = cfg.meta_labeler
        # Propagate market context to strategy if it supports it (duck typing)
        if cfg.market_context is not None and hasattr(self._strategy, "set_market_context"):
            self._strategy.set_market_context(cfg.market_context)
        # Sizing pipeline is built per-run (needs segment_id for MOEX steps)
        self._sizing_pipeline: PositionSizingPipeline | None = None
        self._max_positions_per_segment = cfg.max_positions_per_segment
        self._correlation_cache: dict[tuple[str, str], float] = {}
        self._correlation_update_interval: int = 50
        self._portfolio_returns: list[float] = []
        self._last_run_summary: dict[str, object] = {}

    @property
    def last_run_summary(self) -> dict[str, object]:
        """Per-symbol strategy activity summary from the most recent run() call."""
        return dict(self._last_run_summary)

    def _build_sizing_pipeline(self, segment_id: str) -> PositionSizingPipeline:
        """Build position sizing pipeline with optional EVT/Copula/MOEX steps.

        MOEX-specific steps (RubOilRegimeStep, BrentGateStep) are inserted after
        RegimeStep and before Copula/EVT/MetaLabel/HardCaps. They require segment_id
        which is only available at run() time.

        Pipeline order: Kelly -> VolTarget -> Regime -> [RubOilRegime] -> [BrentGate]
            -> [Copula] -> [EVT] -> MetaLabel -> HardCaps
        """
        cfg = self._config
        steps: list[object] = [KellyStep(), VolTargetStep(), RegimeStep()]
        # MOEX regime steps (Phase 9: Strategy Wiring)
        if cfg.rub_oil_regime_signal is not None:
            steps.append(RubOilRegimeStep(cfg.rub_oil_regime_signal, segment_id))
        if cfg.brent_rub_price > 0:
            steps.append(BrentGateStep(cfg.brent_rub_price, segment_id))
        # Phase 10: Macro regime steps
        if cfg.yield_slope_bps != 0.0 or segment_id.startswith("ru_"):
            steps.append(CBRRegimeStep(cfg.yield_slope_bps, segment_id))
        if cfg.cbr_direction:
            steps.append(SectorAllocationStep(cfg.brent_rub_price, cfg.cbr_direction, segment_id))
        if cfg.use_copula_scaling:
            steps.append(CopulaStep())
        if cfg.use_evt_sizing:
            steps.append(EVTStep())
        steps.append(MetaLabelStep())
        steps.append(HardCapsStep())
        return PositionSizingPipeline(steps=steps)  # type: ignore[arg-type]

    def _build_broker(
        self,
        symbol: str,
        candles: list[Candle],
    ) -> SimulatedBroker:
        """Build a SimulatedBroker, optionally with market impact model."""
        adv_dict: dict[str, float] = {}
        dvol_dict: dict[str, float] = {}
        if self._config.use_impact_model and len(candles) >= 20:  # noqa: PLR2004
            import math  # noqa: PLC0415

            volumes = [float(c.volume) for c in candles[-252:]]
            adv_dict[symbol] = sum(volumes) / len(volumes) if volumes else 1e6
            closes = [float(c.close) for c in candles[-252:]]
            if len(closes) >= 2:  # noqa: PLR2004
                log_rets = [
                    math.log(closes[j] / closes[j - 1])
                    for j in range(1, len(closes))
                    if closes[j - 1] > 0
                ]
                dvol_dict[symbol] = (
                    (sum(r**2 for r in log_rets) / len(log_rets)) ** 0.5 if log_rets else 0.02
                )
            else:
                dvol_dict[symbol] = 0.02
        return SimulatedBroker(
            initial_cash=self._initial_cash,
            use_impact_model=self._config.use_impact_model,
            adv=adv_dict,
            daily_vol=dvol_dict,
            impact_coeff=self._config.impact_coeff,
            max_impact_bps=self._config.max_impact_bps,
        )

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
        broker = self._build_broker(symbol, candles)
        # Build sizing pipeline per-run (MOEX steps need segment_id)
        self._sizing_pipeline = self._build_sizing_pipeline(segment_id)

        trades: list[TradeResult] = []
        snapshots: list[PortfolioState] = []
        entry_prices: dict[str, Decimal] = {}
        entry_bars: dict[str, int] = {}
        entry_strategies: dict[str, str] = {}
        chandelier_stops: dict[str, Decimal] = {}

        # Strategy reasoning counters
        strategy_signal_counts: dict[str, int] = defaultdict(int)
        strategy_none_counts: dict[str, int] = defaultdict(int)
        combined_above_threshold = 0
        trades_opened = 0

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
                history_for_atr = (
                    filter_candles_by_exclusion(history_so_far, self._exclude_periods)
                    if self._exclude_periods
                    else history_so_far
                )
                segment_mult = get_chandelier_multiplier(segment_id)
                candidate = compute_chandelier_stop(
                    history_for_atr,
                    atr_period=22,
                    multiplier=segment_mult,
                )
                if candidate is not None:
                    new_stop = max(chandelier_stops[symbol], candidate)
                    chandelier_stops[symbol] = new_stop
                    # Update broker stop state to match
                    broker.update_stop_loss(symbol, new_stop)

            # (c) Check stop-losses after prices are updated
            # Skip stop check on the fill candle (entry bar + 1) to avoid
            # intraday lows below stop on the same candle used for the fill.
            # Exception: catastrophic drops (>= 15%) override the grace bar.
            entry_bar_for_sym = entry_bars.get(symbol, _NO_ENTRY_BAR)
            is_grace_bar = entry_bar_for_sym + 1 == i
            if is_grace_bar:
                entry_p = entry_prices.get(symbol)
                if entry_p is not None and candle.low < entry_p * (1 - _CATASTROPHIC_DROP_PCT):
                    stop_results = broker.check_stop_losses(candle)
                else:
                    stop_results = []
            else:
                stop_results = broker.check_stop_losses(candle)
            stop_filled = False
            for sr in stop_results:
                if sr.filled and sr.fill_price is not None:
                    stop_filled = True
                    self._close_position(
                        symbol=sr.symbol,
                        exit_price=sr.fill_price,
                        quantity=sr.quantity,
                        entry_prices=entry_prices,
                        entry_bars=entry_bars,
                        entry_strategies=entry_strategies,
                        chandelier_stops=chandelier_stops,
                        bar_index=i,
                        trades=trades,
                    )

            # After stop-loss exit, skip to next bar (don't re-enter same bar)
            if stop_filled:
                snapshots.append(broker.get_portfolio())
                continue

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
                            self._close_position(
                                symbol=open_sym,
                                exit_price=order_result.fill_price,
                                quantity=order_result.quantity,
                                entry_prices=entry_prices,
                                entry_bars=entry_bars,
                                entry_strategies=entry_strategies,
                                chandelier_stops=chandelier_stops,
                                bar_index=i,
                                trades=trades,
                            )
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
                                self._close_position(
                                    symbol=symbol,
                                    exit_price=order_result.fill_price,
                                    quantity=order_result.quantity,
                                    entry_prices=entry_prices,
                                    entry_bars=entry_bars,
                                    entry_strategies=entry_strategies,
                                    chandelier_stops=chandelier_stops,
                                    bar_index=i,
                                    trades=trades,
                                )
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
            effective_max_hold = self._resolve_hold_bars(symbol, entry_strategies, segment_id)
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
                            self._close_position(
                                symbol=symbol,
                                exit_price=order_result.fill_price,
                                quantity=order_result.quantity,
                                entry_prices=entry_prices,
                                entry_bars=entry_bars,
                                entry_strategies=entry_strategies,
                                chandelier_stops=chandelier_stops,
                                bar_index=i,
                                trades=trades,
                            )
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

            # (e2) Track per-strategy signal counts for summary
            if isinstance(self._strategy, JournalingStrategyCombiner):
                for sname, ssig in self._strategy.last_signals.items():
                    if ssig is not None:
                        strategy_signal_counts[sname] += 1
                    else:
                        strategy_none_counts[sname] += 1
            if signal is not None:
                combined_above_threshold += 1

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

                    trades_opened += 1
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
                        bar_index=i,
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

            # Track portfolio returns for EVT sizing
            if len(snapshots) >= 2:  # noqa: PLR2004
                prev_equity = float(snapshots[-2].equity)
                curr_equity = float(snapshots[-1].equity)
                if prev_equity > 0:
                    self._portfolio_returns.append((curr_equity - prev_equity) / prev_equity)

        # Close any remaining open positions at the last candle's close price
        if candles:
            last_candle = candles[-1]
            _last_bar = len(candles) - 1
            for open_symbol, qty in broker.get_positions().items():
                self._close_position(
                    symbol=open_symbol,
                    exit_price=last_candle.close,
                    quantity=qty,
                    entry_prices=entry_prices,
                    entry_bars=entry_bars,
                    entry_strategies=entry_strategies,
                    chandelier_stops=chandelier_stops,
                    bar_index=_last_bar,
                    trades=trades,
                )

        # Log per-symbol strategy activity summary
        self._last_run_summary = {
            "bars_processed": len(candles),
            "trades_total": len(trades),
            "trades_opened": trades_opened,
            "combined_above_threshold": combined_above_threshold,
            "strategy_signals": dict(strategy_signal_counts),
            "strategy_nones": dict(strategy_none_counts),
        }
        logger.info(
            "backtest_symbol_summary",
            symbol=symbol,
            segment=segment_id,
            **self._last_run_summary,
        )

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
        # For portfolio mode, use first symbol's candles for impact estimates
        _first_sym = symbols[0]
        _first_candles = candles_by_symbol.get(_first_sym, [])
        broker = self._build_broker(_first_sym, _first_candles)
        # Build sizing pipeline per-run (MOEX steps need segment_id)
        self._sizing_pipeline = self._build_sizing_pipeline(segment_id)

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

            # Update correlation cache every N bars (portfolio mode only)
            if ts_index % self._correlation_update_interval == 0 and len(symbols) > 1:
                recent_candles: dict[str, list[Candle]] = {}
                for sym in symbols:
                    sym_candles_list = candles_by_symbol.get(sym, [])
                    if sym in candle_index and ts in candle_index[sym]:
                        ci = candle_index[sym][ts]
                        recent_candles[sym] = sym_candles_list[: ci + 1]
                self._correlation_cache = self._compute_correlations(recent_candles)

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
                    history_for_atr = (
                        filter_candles_by_exclusion(history_so_far, self._exclude_periods)
                        if self._exclude_periods
                        else history_so_far
                    )
                    segment_mult = get_chandelier_multiplier(segment_id)
                    candidate = compute_chandelier_stop(
                        history_for_atr, atr_period=22, multiplier=segment_mult
                    )
                    if candidate is not None:
                        new_stop = max(chandelier_stops[sym], candidate)
                        chandelier_stops[sym] = new_stop
                        broker.update_stop_loss(sym, new_stop)

            # Check stop-losses for all symbols (with grace bar + catastrophic override)
            stop_filled_symbols: set[str] = set()
            for sym in symbols:
                if sym not in candle_index or ts not in candle_index[sym]:
                    continue
                sym_candles = candles_by_symbol.get(sym, [])
                idx = candle_index[sym][ts]
                candle = sym_candles[idx]

                # Grace bar: skip stop on the fill candle, unless catastrophic drop
                is_grace = entry_bars.get(sym, _NO_ENTRY_BAR) + 1 == bar_counts.get(sym, 0)
                if is_grace:
                    entry_p = entry_prices.get(sym)
                    if entry_p is not None and candle.low < entry_p * (1 - _CATASTROPHIC_DROP_PCT):
                        stop_results = broker.check_stop_losses(candle)
                    else:
                        stop_results = []
                else:
                    stop_results = broker.check_stop_losses(candle)

                for sr in stop_results:
                    if sr.filled and sr.fill_price is not None:
                        stop_filled_symbols.add(sr.symbol)
                        self._close_position(
                            symbol=sr.symbol,
                            exit_price=sr.fill_price,
                            quantity=sr.quantity,
                            entry_prices=entry_prices,
                            entry_bars=entry_bars,
                            entry_strategies=entry_strategies,
                            chandelier_stops=chandelier_stops,
                            bar_index=bar_counts.get(sr.symbol, 0),
                            trades=trades,
                        )

            # Check profit target and time exit for all symbols
            for sym in symbols:
                if sym in stop_filled_symbols:
                    continue
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
                                    self._close_position(
                                        symbol=sym,
                                        exit_price=order_result.fill_price,
                                        quantity=order_result.quantity,
                                        entry_prices=entry_prices,
                                        entry_bars=entry_bars,
                                        entry_strategies=entry_strategies,
                                        chandelier_stops=chandelier_stops,
                                        bar_index=bar_counts.get(sym, 0),
                                        trades=trades,
                                    )
                                continue

                # Time-based exit check
                effective_max_hold = self._resolve_hold_bars(sym, entry_strategies, segment_id)
                if effective_max_hold > 0 and sym in entry_bars and idx + 1 < len(sym_candles):
                    bars_since_entry = bar_counts.get(sym, 0) - entry_bars.get(sym, 0)
                    if bars_since_entry >= effective_max_hold:
                        fill_candle = sym_candles[idx + 1]
                        held = broker.get_positions().get(sym, Decimal(0))
                        if held > 0:
                            order = OrderRequest(symbol=sym, side="SELL", quantity=held)
                            order_result = broker.submit_order(order, fill_candle)
                            if order_result.filled and order_result.fill_price is not None:
                                self._close_position(
                                    symbol=sym,
                                    exit_price=order_result.fill_price,
                                    quantity=order_result.quantity,
                                    entry_prices=entry_prices,
                                    entry_bars=entry_bars,
                                    entry_strategies=entry_strategies,
                                    chandelier_stops=chandelier_stops,
                                    bar_index=bar_counts.get(sym, 0),
                                    trades=trades,
                                )
                            continue

            # Generate signals for each symbol (skip those stopped out this bar)
            for sym in symbols:
                if sym in stop_filled_symbols:
                    continue
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
                            segment_id=segment_id,
                            signal=signal,
                            history=history,
                            entry_bars=entry_bars,
                            entry_strategies=entry_strategies,
                            chandelier_stops=chandelier_stops,
                            bar_index=bar_counts.get(sym, 0),
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
                self._close_position(
                    symbol=sym,
                    exit_price=sym_candles[-1].close,
                    quantity=qty,
                    entry_prices=entry_prices,
                    entry_bars=entry_bars,
                    entry_strategies=entry_strategies,
                    chandelier_stops=chandelier_stops,
                    bar_index=bar_counts.get(sym, 0),
                    trades=trades,
                )

        return trades, snapshots

    def _resolve_hold_bars(
        self,
        symbol: str,
        entry_strategies: dict[str, str],
        segment_id: str = "",
    ) -> int:
        """Resolve the effective max hold bars for a given symbol's position.

        Uses the strategy name that opened the position to look up
        per-strategy hold limits when ``max_hold_bars`` is a dict.
        MOEX segments (``ru_*``) get a 1.3x uplift.
        """
        strategy_name = entry_strategies.get(symbol, "")
        return resolve_max_hold_bars(self._max_hold_bars, strategy_name, segment_id=segment_id)

    def _record_trade(self, trade: TradeResult) -> None:
        """Record a completed trade in the Rolling Kelly estimator."""
        if self._rolling_kelly is not None:
            self._rolling_kelly.update(TradeRecord(pnl=trade.pnl, pnl_pct=trade.pnl_pct))

    def _close_position(
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

        Args:
            symbol: Ticker being closed.
            exit_price: Price at which the position is exited.
            quantity: Number of shares/contracts to close.
            entry_prices: Mutable map of open-entry prices; symbol is popped.
            entry_bars: Mutable map of the bar index at which entry occurred;
                symbol is popped.
            entry_strategies: Mutable map of the strategy name that opened the
                position; symbol is popped.
            chandelier_stops: Mutable map of chandelier stop prices; symbol
                is popped.
            bar_index: Current bar index (used to compute hold_bars).
            trades: Mutable list; the new TradeResult is appended.

        Returns:
            The created ``TradeResult``.
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
        self._record_trade(trade)
        return trade

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

        # Capture enriched features and model probas from combiner
        strategy_features: dict[str, float] | None = None
        model_probas: dict[str, float] | None = None
        if isinstance(self._strategy, JournalingStrategyCombiner):
            feats = self._strategy.last_features
            if feats:
                strategy_features = feats
            model_probas = self._strategy.last_model_probas

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
                strategy_features=strategy_features,
                model_probas=model_probas,
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

    def _compute_segment_exposure(
        self,
        broker: SimulatedBroker,
        segment_id: str,  # noqa: ARG002
    ) -> Decimal:
        """Compute the total position value for a segment (for concentration check).

        In single-symbol mode, all positions belong to the same segment.
        In portfolio mode, the engine only trades one segment at a time.
        So current equity in positions approximates segment exposure.
        """
        portfolio = broker.get_portfolio()
        position_value = portfolio.equity - portfolio.cash
        return max(position_value, Decimal(0))

    @staticmethod
    def _compute_correlations(
        candles_by_symbol: dict[str, list[Candle]],
        lookback: int = 60,
    ) -> dict[tuple[str, str], float]:
        """Compute trailing pairwise correlations for open positions.

        Delegates to :func:`finalayze.risk.correlation.compute_correlation_matrix`
        which uses pure-Python Pearson correlation (no numpy, no NaN risk).
        """
        from finalayze.risk.correlation import compute_correlation_matrix  # noqa: PLC0415

        return compute_correlation_matrix(candles_by_symbol, window=lookback)

    def _handle_buy(  # noqa: PLR0911, PLR0912, PLR0915
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

        # Check segment position cap
        segment_count = sum(1 for _s, qty in broker.get_positions().items() if qty > 0)
        if segment_count >= self._max_positions_per_segment:
            self._journal_skip(
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
        position_value = self._sizing_pipeline.compute(context)

        # Apply confidence scaling from signal (post-pipeline multiplier)
        if signal is not None:
            confidence_scale = Decimal(str(0.5 + signal.confidence * 0.5))  # [0.5x, 1.0x]
            position_value = position_value * confidence_scale

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
            if segment_id.startswith("ru_"):
                check_dt = datetime.combine(check_dt.date(), _MOEX_MARKET_OPEN_UTC)
            else:
                check_dt = datetime.combine(check_dt.date(), _US_MARKET_OPEN_UTC)
        market_id = "moex" if segment_id.startswith("ru_") else "us"
        result = checker.check(
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
            sector_exposure_value=self._compute_segment_exposure(broker, segment_id),
            correlations=self._correlation_cache or None,
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

        # Pre-compute ATR stop-loss — use strategy-specific multiplier
        strategy_name = signal.strategy_name if signal is not None else ""
        stop_atr_mult = resolve_stop_atr_multiplier(strategy_name, segment_id=segment_id)
        stop_price = compute_atr_stop_loss(
            entry_price=fill_price,
            candles=history,
            atr_multiplier=stop_atr_mult,
            exclude_periods=self._exclude_periods,
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
        bar_index: int = 0,
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
            # Use empty dicts as fallback when optional tracking dicts are None
            _eb = entry_bars if entry_bars is not None else {}
            _es = entry_strategies if entry_strategies is not None else {}
            _cs = chandelier_stops if chandelier_stops is not None else {}
            self._close_position(
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

"""Backtest engine -- iterates candles and runs a strategy with risk management.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from finalayze.backtest.config import (
    BacktestConfig,
    resolve_max_hold_bars,
)
from finalayze.backtest.journal import BacktestJournal
from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
from finalayze.backtest.position_executor import BacktestPositionExecutor
from finalayze.backtest.risk_evaluator import BacktestRiskEvaluator
from finalayze.core.schemas import (
    Candle,
    PortfolioState,
    SignalDirection,
    TradeResult,
)
from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.simulated_broker import SimulatedBroker
from finalayze.risk.chandelier_exit import compute_chandelier_stop, get_chandelier_multiplier
from finalayze.risk.position_sizing_pipeline import (
    BrentGateStep,
    CBRRegimeStep,
    CopulaStep,
    CpiRiskOffStep,
    EVTStep,
    HardCapsStep,
    MetaLabelStep,
    PositionSizingPipeline,
    RegimeStep,
    RubOilRegimeStep,
    SectorAllocationStep,
    VolTargetStep,
)
from finalayze.risk.pre_trade_check import PreTradeChecker
from finalayze.risk.stop_loss import filter_candles_by_exclusion
from finalayze.risk.stops import CATASTROPHIC_DROP_PCT

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import date

    from finalayze.backtest.costs import TransactionCosts
    from finalayze.backtest.decision_journal import DecisionJournal, FinalAction
    from finalayze.core.schemas import Signal
    from finalayze.execution.deposit_broker import DepositSimulatedBroker
    from finalayze.risk.circuit_breaker import CircuitBreaker
    from finalayze.risk.kelly import RollingKelly
    from finalayze.risk.loss_limits import LossLimitTracker
    from finalayze.risk.regime import RegimeProvider
    from finalayze.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)

# Sentinel value for "no entry bar recorded".  Used instead of a bare ``-2``
# so grep can find every reference and the meaning is self-documenting.
_NO_ENTRY_BAR = -2

# S3.1: catastrophic-drop override threshold is defined in risk.stops so the
# live PositionTracker can apply the same gating. Local alias keeps the
# existing call sites untouched.
_CATASTROPHIC_DROP_PCT = CATASTROPHIC_DROP_PCT

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
        # D-09 / LIQ-07 (Phase 66): a per-segment concurrent-position cap, when supplied,
        # overrides the portfolio-wide ``max_positions`` for the SHARED ``run_portfolio`` broker
        # (and only there -- it is silently ineffective in the per-symbol ``run`` path where each
        # symbol owns its own broker). ``None`` preserves prior behaviour for every existing caller.
        self._max_concurrent_positions = cfg.max_concurrent_positions
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

        # -- Composed components --
        self._journal = BacktestJournal(
            decision_journal=cfg.decision_journal,
            strategy=strategy,
        )

    @property
    def last_run_summary(self) -> dict[str, object]:
        """Per-symbol strategy activity summary from the most recent run() call."""
        return dict(self._last_run_summary)

    def _build_sizing_pipeline(self, segment_id: str) -> PositionSizingPipeline:
        """Build position sizing pipeline with optional EVT/Copula/MOEX steps.

        MOEX-specific steps (RubOilRegimeStep, BrentGateStep) are inserted after
        RegimeStep and before Copula/EVT/MetaLabel/HardCaps. They require segment_id
        which is only available at run() time.

        Pipeline order: VolTarget -> Regime -> [RubOilRegime] -> [BrentGate]
            -> [CBRRegime] -> [CpiRiskOff] -> [SectorAllocation] -> [Copula] -> [EVT]
            -> MetaLabel -> HardCaps
        (Kelly sizing is pre-applied to SizingContext.base_position upstream.)
        """
        cfg = self._config
        steps: list[object] = [VolTargetStep(), RegimeStep()]
        # MOEX regime steps (Phase 9: Strategy Wiring)
        if cfg.rub_oil_regime_signal is not None:
            steps.append(RubOilRegimeStep(cfg.rub_oil_regime_signal, segment_id))  # type: ignore[arg-type]
        if cfg.brent_rub_price > 0:
            steps.append(BrentGateStep(cfg.brent_rub_price, segment_id))
        # Phase 10: Macro regime steps
        if cfg.yield_slope_bps != 0.0 or segment_id.startswith("ru_"):
            steps.append(CBRRegimeStep(cfg.yield_slope_bps, segment_id))
        # Phase 60 (INTG-03): CPI risk-off overlay. ru_-gated; resolves CPI per-bar
        # from SizingContext.bar_date (no value baked at build time -> no look-ahead).
        if cfg.cpi_yoy_fraction != 0.0 or segment_id.startswith("ru_"):
            steps.append(CpiRiskOffStep(segment_id, cfg.cpi_yoy_fraction))
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

    def _build_executor(self) -> BacktestPositionExecutor:
        """Build a position executor with current engine state."""
        return BacktestPositionExecutor(
            journal=self._journal,
            rolling_kelly=self._rolling_kelly,
            kelly_fraction=self._kelly_fraction,
            max_position_pct=self._max_position_pct,
            max_positions_per_segment=self._max_positions_per_segment,
            sizing_pipeline=self._sizing_pipeline,
            transaction_costs=self._transaction_costs,
            trail_activation_atr=self._trail_activation_atr,
            trail_distance_atr=self._trail_distance_atr,
            stop_loss_mode=self._stop_loss_mode,
            exclude_periods=self._exclude_periods,
            profit_target_atr=self._profit_target_atr,
            target_vol=self._target_vol,
            meta_labeler=self._meta_labeler,
            correlation_cache=self._correlation_cache,
            portfolio_returns=self._portfolio_returns,
            us_market_open_utc=_US_MARKET_OPEN_UTC,
            moex_market_open_utc=_MOEX_MARKET_OPEN_UTC,
            max_order_volume_pct=self._config.max_order_volume_pct,
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

        # Build executor with current state
        executor = self._build_executor()

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
                    executor.close_position(
                        symbol=sr.symbol,
                        exit_price=sr.fill_price,
                        quantity=sr.quantity,
                        entry_prices=entry_prices,
                        entry_bars=entry_bars,
                        entry_strategies=entry_strategies,
                        chandelier_stops=chandelier_stops,
                        bar_index=i,
                        trades=trades,
                        exit_reason="stop",
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
                            executor.close_position(
                                symbol=open_sym,
                                exit_price=order_result.fill_price,
                                quantity=order_result.quantity,
                                entry_prices=entry_prices,
                                entry_bars=entry_bars,
                                entry_strategies=entry_strategies,
                                chandelier_stops=chandelier_stops,
                                bar_index=i,
                                trades=trades,
                                exit_reason="force_close",
                            )
                    snapshots.append(broker.get_portfolio())
                    continue

                # L2+: suppress new entries
                if cb_level in ("halted", "liquidate"):
                    self._journal.record_skip(
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
                    self._journal.record_skip(
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
                                executor.close_position(
                                    symbol=symbol,
                                    exit_price=order_result.fill_price,
                                    quantity=order_result.quantity,
                                    entry_prices=entry_prices,
                                    entry_bars=entry_bars,
                                    entry_strategies=entry_strategies,
                                    chandelier_stops=chandelier_stops,
                                    bar_index=i,
                                    trades=trades,
                                    exit_reason="profit_target",
                                )
                                self._journal.record_skip(
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
                            executor.close_position(
                                symbol=symbol,
                                exit_price=order_result.fill_price,
                                quantity=order_result.quantity,
                                entry_prices=entry_prices,
                                entry_bars=entry_bars,
                                entry_strategies=entry_strategies,
                                chandelier_stops=chandelier_stops,
                                bar_index=i,
                                trades=trades,
                                exit_reason="time",
                            )
                            self._journal.record_skip(
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
                        self._journal.record_skip(
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
                    executor.handle_buy(
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
                            float(regime_state.position_scale) if regime_state is not None else None
                        ),
                        entry_strategies=entry_strategies,
                        chandelier_stops=chandelier_stops,
                    )

                elif signal.direction == SignalDirection.SELL:
                    executor.handle_sell(
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
                self._journal.record_skip(
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

        # S5.3: end-of-data positions are left OPEN by default.  Closing
        # them at the final candle's close systematically inflated Sharpe
        # (synthetic exit at mid with no spread/slippage = costless realised
        # PnL).  Equity snapshots already carry MTM via the broker, so the
        # equity curve / Sharpe / max-DD remain honest without the forced
        # close.  Set ``force_close_at_end=True`` on BacktestConfig to keep
        # the legacy behaviour (e.g. for tooling that needs fully realised PnL).
        unclosed_at_end = 0
        if candles:
            open_positions = list(broker.get_positions().items())
            if self._config.force_close_at_end:
                last_candle = candles[-1]
                _last_bar = len(candles) - 1
                for open_symbol, qty in open_positions:
                    executor.close_position(
                        symbol=open_symbol,
                        exit_price=last_candle.close,
                        quantity=qty,
                        entry_prices=entry_prices,
                        entry_bars=entry_bars,
                        entry_strategies=entry_strategies,
                        chandelier_stops=chandelier_stops,
                        bar_index=_last_bar,
                        trades=trades,
                        exit_reason="force_close",
                    )
            else:
                unclosed_at_end = sum(1 for _, qty in open_positions if qty > 0)
                if unclosed_at_end:
                    logger.info(
                        "backtest_unclosed_at_end",
                        symbol=symbol,
                        count=unclosed_at_end,
                    )

        # Log per-symbol strategy activity summary
        self._last_run_summary = {
            "bars_processed": len(candles),
            "trades_total": len(trades),
            "trades_opened": trades_opened,
            "combined_above_threshold": combined_above_threshold,
            "strategy_signals": dict(strategy_signal_counts),
            "strategy_nones": dict(strategy_none_counts),
            "unclosed_at_end": unclosed_at_end,
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
        *,
        eligible_at: Callable[[datetime], set[str]] | None = None,
        dividend_schedule: dict[tuple[str, date], Decimal] | None = None,
        deposit_ladder: DepositSimulatedBroker | None = None,
    ) -> tuple[list[TradeResult], list[PortfolioState]]:
        """Run a portfolio-level backtest over multiple symbols.

        Iterates through a unified timeline, generating signals for each symbol
        on each bar, managing shared capital across all positions.

        Args:
            symbols: List of ticker symbols to trade.
            segment_id: Market segment identifier.
            candles_by_symbol: Candle data keyed by symbol.
            dividend_schedule: Optional total-return dividend index (ACCT-01 / D-16),
                ``{(symbol, ex_date): gross_per_share}`` from
                ``backtest.dividend_schedule.load_dividend_schedule``. When supplied, held
                positions accrue net-of-NDFL dividends to cash per bar (after the price update,
                before valuation) via ``broker.process_dividends`` -- the trade-PnL formula is
                untouched and the credited cash flows into the equity curve through
                ``get_portfolio``. When ``None`` (the default), the per-bar income hook is a
                no-op and behaviour is UNCHANGED -- this preserves every existing
                ``run_portfolio`` caller and test (byte-identical, D-16).
            deposit_ladder: Optional ``DepositSimulatedBroker`` deposit sleeve (DEP-01 / ACCT-02).
                When supplied, ``deposit_ladder.accrue(bar_date)`` credits its own daily-compounded
                net-of-NDFL interest each bar. When ``None`` (the default), no deposit interest is
                accrued and behaviour is UNCHANGED -- preserving every existing caller and test
                (byte-identical, D-16).
            eligible_at: Optional CARDINAL D-05 as-of universe gate. When supplied, the eligible
                universe is recomputed at each QUARTERLY rebalance bar ``T`` via ``eligible_at(T)``
                -- which the caller backs with ``markets.liquidity.eligible_universe_as_of`` so the
                set is derived from ONLY the candles dated ``timestamp <= T`` (zero look-ahead,
                survivorship-safe). New entries are SKIPPED for symbols not in the current eligible
                set; the set is carried forward between rebalances. EXISTING positions for a name
                that drops out of the eligible set are NOT force-liquidated -- they are managed and
                exited normally (stop/profit/time/SELL); the gate only blocks NEW entries. When
                ``None`` (the default), every passed symbol is always eligible and behaviour is
                UNCHANGED -- this preserves every existing ``run_portfolio`` caller and test.

        Returns:
            A tuple of (trades, portfolio_snapshots).
        """
        if not symbols or not candles_by_symbol:
            return [], []

        # D-09 / LIQ-07: a per-segment concurrent-position cap (if configured) overrides the
        # portfolio-wide ``max_positions`` for the SHARED ``PreTradeChecker``. One checker + one
        # broker means ``MaxPositionsCheck`` is portfolio-wide here (unlike the per-symbol ``run``
        # path where each symbol's broker makes the cap silently ineffective -- PATTERNS Pitfall 4).
        max_positions = (
            self._max_concurrent_positions
            if self._max_concurrent_positions is not None
            else self._max_positions
        )
        checker = PreTradeChecker(
            max_position_pct=self._max_position_pct,
            max_positions_per_market=max_positions,
        )
        # For portfolio mode, use first symbol's candles for impact estimates
        _first_sym = symbols[0]
        _first_candles = candles_by_symbol.get(_first_sym, [])
        broker = self._build_broker(_first_sym, _first_candles)
        # Build sizing pipeline per-run (MOEX steps need segment_id)
        self._sizing_pipeline = self._build_sizing_pipeline(segment_id)

        # Build executor with current state
        executor = self._build_executor()

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

        # CARDINAL D-05 as-of universe gate state. ``current_eligible`` is recomputed only at
        # quarterly rebalance boundaries (index-reconstitution cadence, D-06) and carried forward
        # between them. ``None`` means "no gate" -> every symbol is always eligible (unchanged
        # behaviour). ``_quarter_key`` detects a quarter boundary on the unified timeline.
        current_eligible: set[str] | None = None
        last_rebalance_quarter: tuple[int, int] | None = None

        def _quarter_key(when: datetime) -> tuple[int, int]:
            return (when.year, (when.month - 1) // 3)

        ts_index = 0
        for ts in all_timestamps:
            broker.set_timestamp(ts)

            # D-05 / D-06: at the first bar and at every new quarter, re-derive the eligible
            # universe AS-OF this bar from candles <= ts (the callback enforces the cutoff).
            if eligible_at is not None:
                this_quarter = _quarter_key(ts)
                if last_rebalance_quarter is None or this_quarter != last_rebalance_quarter:
                    current_eligible = eligible_at(ts)
                    last_rebalance_quarter = this_quarter

            # Update prices for all symbols that have data at this timestamp
            for sym in symbols:
                sym_candles = candles_by_symbol.get(sym, [])
                if sym in candle_index and ts in candle_index[sym]:
                    idx = candle_index[sym][ts]
                    broker.update_prices(sym_candles[idx])
                    bar_counts[sym] = bar_counts.get(sym, 0) + 1

            # Total-return income credit (mirrors bond_engine.py:214). No-op when both inputs
            # are None -> all existing run_portfolio callers stay byte-identical (D-16). Credits
            # net-of-NDFL income to cash via the broker; the trade-PnL formula is untouched and
            # equity auto-updates in get_portfolio(). After update_prices, before the snapshot
            # append below -> the credit lands in THIS bar's equity and nowhere earlier (D-17).
            if dividend_schedule is not None:
                broker.process_dividends(ts.date(), dividend_schedule)
            if deposit_ladder is not None:
                deposit_ladder.accrue(ts.date())

            # Update correlation cache every N bars (portfolio mode only)
            if ts_index % self._correlation_update_interval == 0 and len(symbols) > 1:
                recent_candles: dict[str, list[Candle]] = {}
                for sym in symbols:
                    sym_candles_list = candles_by_symbol.get(sym, [])
                    if sym in candle_index and ts in candle_index[sym]:
                        ci = candle_index[sym][ts]
                        recent_candles[sym] = sym_candles_list[: ci + 1]
                self._correlation_cache = BacktestRiskEvaluator.compute_correlations(recent_candles)
                executor.set_correlation_cache(self._correlation_cache)

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
                        executor.close_position(
                            symbol=sr.symbol,
                            exit_price=sr.fill_price,
                            quantity=sr.quantity,
                            entry_prices=entry_prices,
                            entry_bars=entry_bars,
                            entry_strategies=entry_strategies,
                            chandelier_stops=chandelier_stops,
                            bar_index=bar_counts.get(sr.symbol, 0),
                            trades=trades,
                            exit_reason="stop",
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
                                    executor.close_position(
                                        symbol=sym,
                                        exit_price=order_result.fill_price,
                                        quantity=order_result.quantity,
                                        entry_prices=entry_prices,
                                        entry_bars=entry_bars,
                                        entry_strategies=entry_strategies,
                                        chandelier_stops=chandelier_stops,
                                        bar_index=bar_counts.get(sym, 0),
                                        trades=trades,
                                        exit_reason="profit_target",
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
                                executor.close_position(
                                    symbol=sym,
                                    exit_price=order_result.fill_price,
                                    quantity=order_result.quantity,
                                    entry_prices=entry_prices,
                                    entry_bars=entry_bars,
                                    entry_strategies=entry_strategies,
                                    chandelier_stops=chandelier_stops,
                                    bar_index=bar_counts.get(sym, 0),
                                    trades=trades,
                                    exit_reason="time",
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
                        # CARDINAL D-05: skip NEW entries for symbols not in the current as-of
                        # eligible universe. ``current_eligible is None`` means no gate (unchanged).
                        # Existing positions for a now-ineligible name are untouched here -- they
                        # are managed/exited by the stop/profit/time/SELL paths above, never
                        # force-liquidated on de-listing from the eligible set.
                        if current_eligible is not None and sym not in current_eligible:
                            continue

                        # Skip BUY if regime blocks new longs
                        if regime_state is not None and not regime_state.allow_new_longs:
                            continue

                        executor.handle_buy(
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
                                float(regime_state.position_scale)
                                if regime_state is not None
                                else None
                            ),
                            entry_strategies=entry_strategies,
                            chandelier_stops=chandelier_stops,
                        )
                    elif signal.direction == SignalDirection.SELL:
                        executor.handle_sell(
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

        # S5.3: end-of-data positions are left OPEN by default.  Same
        # rationale as the single-symbol path: forced close at last bar's
        # close inflates Sharpe by pretending we can exit at mid without
        # spread / slippage.  Equity snapshots already carry the MTM via
        # broker.  ``force_close_at_end=True`` recovers legacy behaviour.
        if candles_by_symbol and self._config.force_close_at_end:
            for sym in symbols:
                sym_candles = candles_by_symbol.get(sym, [])
                if not sym_candles:
                    continue
                qty = broker.get_positions().get(sym, Decimal(0))
                if qty <= 0:
                    continue
                executor.close_position(
                    symbol=sym,
                    exit_price=sym_candles[-1].close,
                    quantity=qty,
                    entry_prices=entry_prices,
                    entry_bars=entry_bars,
                    entry_strategies=entry_strategies,
                    chandelier_stops=chandelier_stops,
                    bar_index=bar_counts.get(sym, 0),
                    trades=trades,
                    exit_reason="force_close",
                )
        elif candles_by_symbol:
            unclosed = sum(1 for q in broker.get_positions().values() if q > 0)
            if unclosed:
                logger.info("portfolio_backtest_unclosed_at_end", count=unclosed)

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

    # ------------------------------------------------------------------
    # Backward-compatibility shims -- methods that external code or tests
    # may reference directly on BacktestEngine.  They now delegate to
    # the composed components.
    # ------------------------------------------------------------------

    def _record_trade(self, trade: TradeResult) -> None:
        """Record a completed trade in the Rolling Kelly estimator.

        .. deprecated:: Delegate to ``BacktestPositionExecutor.record_trade``.
        """
        executor = self._build_executor()
        executor.record_trade(trade)

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
        exit_reason: str | None = None,
    ) -> TradeResult:
        """Close a position and record the trade.

        .. deprecated:: Delegate to ``BacktestPositionExecutor.close_position``.
        """
        executor = self._build_executor()
        return executor.close_position(
            symbol=symbol,
            exit_price=exit_price,
            quantity=quantity,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=bar_index,
            trades=trades,
            exit_reason=exit_reason,
        )

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
        """Record a decision in the journal.

        .. deprecated:: Delegate to ``BacktestJournal.record_decision``.
        """
        self._journal.record_decision(
            action=action,
            timestamp=timestamp,
            symbol=symbol,
            segment_id=segment_id,
            broker=broker,
            history=history,
            signal=signal,
            skip_reason=skip_reason,
            pre_trade_passed=pre_trade_passed,
            pre_trade_violations=pre_trade_violations,
            position_value=position_value,
            quantity=quantity,
            fill_price=fill_price,
            stop_loss_price=stop_loss_price,
            cb_level=cb_level,
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
        """Convenience wrapper for journaling a SKIP decision.

        .. deprecated:: Delegate to ``BacktestJournal.record_skip``.
        """
        self._journal.record_skip(
            timestamp=timestamp,
            symbol=symbol,
            segment_id=segment_id,
            broker=broker,
            history=history,
            skip_reason=skip_reason,
            cb_level=cb_level,
        )

    @staticmethod
    def _compute_segment_exposure(
        broker: SimulatedBroker,
        segment_id: str,
    ) -> Decimal:
        """Compute the total position value for a segment.

        .. deprecated:: Delegate to ``BacktestRiskEvaluator.compute_segment_exposure``.
        """
        return BacktestRiskEvaluator.compute_segment_exposure(broker, segment_id)

    @staticmethod
    def _compute_correlations(
        candles_by_symbol: dict[str, list[Candle]],
        lookback: int = 60,
    ) -> dict[tuple[str, str], float]:
        """Compute trailing pairwise correlations.

        .. deprecated:: Delegate to ``BacktestRiskEvaluator.compute_correlations``.
        """
        return BacktestRiskEvaluator.compute_correlations(candles_by_symbol, lookback)

    def _handle_buy(
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
        """Process a BUY signal.

        .. deprecated:: Delegate to ``BacktestPositionExecutor.handle_buy``.
        """
        executor = self._build_executor()
        executor.handle_buy(
            broker,
            checker,
            fill_candle,
            symbol,
            history,
            entry_prices,
            segment_id=segment_id,
            signal=signal,
            entry_bars=entry_bars,
            bar_index=bar_index,
            regime_position_scale=regime_position_scale,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
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
        """Process a SELL signal.

        .. deprecated:: Delegate to ``BacktestPositionExecutor.handle_sell``.
        """
        executor = self._build_executor()
        executor.handle_sell(
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
            bar_index=bar_index,
        )

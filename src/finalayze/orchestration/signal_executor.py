"""Signal generation, order building, and execution orchestration (Phase 1.7).

Extracted from TradingLoop to isolate:
  - Signal generation from candles + sentiment
  - Pre-trade validation checks
  - Position sizing pipeline assembly and computation
  - Order building (side, quantity) from signal
  - Order submission and fill handling (stop-loss setup, Kelly updates)

This class is the CORE of the trading system. It orchestrates the flow from
a signal to an executed trade, handling all intermediate steps including
circuit breaker gating, sizing, pre-trade checks, and stop-loss setup.

Thread safety: uses injected dependencies (PositionTracker, BrokerRouter, etc.)
which handle their own locking. No shared mutable state on SignalExecutor itself.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

from finalayze.core.schemas import SignalDirection

if TYPE_CHECKING:
    from config.settings import Settings

    from finalayze.api.alerts import TelegramAlerter
    from finalayze.api.metrics import MetricsCollector
    from finalayze.core.schemas import Candle, PortfolioState, Signal
    from finalayze.data.macro_cache import MacroCacheService
    from finalayze.execution.broker_base import OrderRequest
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.markets.instruments import InstrumentRegistry
    from finalayze.ml.registry import MLModelRegistry
    from finalayze.monitoring.health_monitor import HealthMonitor
    from finalayze.monitoring.sandbox_monitor import SandboxMonitorService
    from finalayze.orchestration.db_persistence import TradingPersistence
    from finalayze.orchestration.position_manager import PositionTracker
    from finalayze.orchestration.sentiment_manager import SentimentManager
    from finalayze.risk.circuit_breaker import CircuitLevel
    from finalayze.risk.loss_limits import LossLimitTracker
    from finalayze.risk.position_sizing_pipeline import PositionSizingPipeline
    from finalayze.risk.pre_trade_check import PreTradeChecker
    from finalayze.risk.regime import RegimeState
    from finalayze.strategies.combiner import StrategyCombiner

# ── Constants ──────────────────────────────────────────────────────────────
_CANDLE_LOOKBACK = 210  # SMA-200 needs 200 bars + buffer; dual_momentum needs 126
_CAUTION_SIZE_FACTOR = Decimal("0.5")  # halve position size at CAUTION
_MIN_CONFIDENCE_BOOST = 1.2  # raise required confidence 20% at CAUTION
_MIN_ALERT_CONFIDENCE = 0.5  # ALRT-02 D-13: skip noise below this threshold
_ZERO = Decimal(0)
_STALENESS_THRESHOLD_HOURS: float = 72.0  # 3x daily; covers weekends + calendar-aware holidays
_ATR_MULTIPLIER_US = Decimal("2.0")
_ATR_MULTIPLIER_MOEX = Decimal("2.5")
_MARKET_CURRENCY: dict[str, str] = {"us": "USD", "moex": "RUB"}

_log = structlog.get_logger(__name__)


class SignalExecutor:
    """Orchestrates signal generation, order building, and execution.

    Core responsibilities:
      1. Generate signal from candles + sentiment
      2. Check stop-losses on open positions
      3. Pre-trade validation (PDT, exposure, circuit breakers, etc.)
      4. Build order via position sizing pipeline
      5. Submit order + set/clear stop-loss on fill

    Cycle state (counts, last prices) is maintained per TradingLoop cycle:
      - _last_prices: cache during strategy cycle (populated by candle fetch)
      - _segment_min_confidence: cache preset values (lazy-loaded)

    Returned dict from process_instrument includes:
      - signals_generated: count for this instrument
      - orders_submitted: count for this instrument
      - orders_filled: count for this instrument
      - errors_caught: count for this instrument
      - dropped_no_bars, dropped_below_threshold, dropped_pre_trade: counts
    """

    def __init__(
        self,
        strategy: StrategyCombiner,
        broker_router: BrokerRouter,
        position_tracker: PositionTracker,
        sentiment_mgr: SentimentManager,
        persistence: TradingPersistence | None,
        pre_trade_checker: PreTradeChecker,
        loss_limit_tracker: LossLimitTracker,
        macro_cache: MacroCacheService | None,
        health_monitor: HealthMonitor | None,
        sandbox_monitor: SandboxMonitorService | None,
        metrics: type[MetricsCollector] | None,
        alerter: TelegramAlerter,
        registry: InstrumentRegistry,
        ml_registry: MLModelRegistry | None,
        settings: Settings,
    ) -> None:
        """Initialize SignalExecutor with all required dependencies.

        Args:
            strategy: StrategyCombiner for signal generation
            broker_router: BrokerRouter for order submission
            position_tracker: PositionTracker for stop-loss + entry tracking
            sentiment_mgr: SentimentManager for sentiment scores
            persistence: TradingPersistence for fire-and-forget writes
            pre_trade_checker: PreTradeChecker for order validation
            loss_limit_tracker: LossLimitTracker for daily loss limits
            macro_cache: Optional MacroCacheService for regime/macro data
            health_monitor: Optional HealthMonitor for feed timestamps
            sandbox_monitor: Optional SandboxMonitorService for slippage
            metrics: Optional MetricsCollector for Prometheus metrics
            alerter: TelegramAlerter for alerts
            registry: InstrumentRegistry for instrument lookups
            ml_registry: Optional MLModelRegistry for ML confidence
            settings: Settings for risk limits, Kelly fraction, etc.
        """
        from finalayze.execution.broker_base import OrderRequest  # noqa: PLC0415
        from finalayze.risk.circuit_breaker import CircuitLevel  # noqa: PLC0415

        # Store class references for runtime use without module-level imports
        self._OrderRequest = OrderRequest
        self._CircuitLevel = CircuitLevel

        self._strategy = strategy
        self._broker_router = broker_router
        self._position_tracker = position_tracker
        self._sentiment_mgr = sentiment_mgr
        self._persistence = persistence
        self._pre_trade_checker = pre_trade_checker
        self._loss_limit_tracker = loss_limit_tracker
        self._macro_cache = macro_cache
        self._health_monitor = health_monitor
        self._sandbox_monitor = sandbox_monitor
        self._metrics = metrics
        self._alerter = alerter
        self._registry = registry
        self._ml_registry = ml_registry
        self._settings = settings

        # Per-instrument last price cache: symbol -> Decimal (built during strategy cycle)
        self._last_prices: dict[str, Decimal] = {}

        # Segment min_combined_confidence cache: seg_id -> float
        self._segment_min_confidence: dict[str, float] = {}

    def process_instrument(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        instrument: object,
        market_id: str,
        level: CircuitLevel,
        fetcher: object,
        now: datetime,
        equity: Decimal,
        cash: Decimal,
        portfolio: PortfolioState | None,
    ) -> dict[str, Any]:
        """Process one instrument: fetch candles, generate signal, submit order.

        This is the CORE trading flow. It handles:
          1. Candle fetch and validation
          2. Stop-loss checks
          3. Signal generation
          4. Pre-trade validation
          5. Order building
          6. Order submission + fill handling

        Returns a dict with cycle stats:
          - signals_generated: 0 or 1
          - orders_submitted: 0 or 1
          - orders_filled: 0 or 1
          - errors_caught: count of exceptions
          - dropped_no_bars: 1 if no valid candles
          - dropped_below_threshold: 1 if signal below threshold
          - dropped_pre_trade: 1 if pre-trade check failed
        """
        from finalayze.core.exceptions import InstrumentNotFoundError  # noqa: PLC0415

        stats: dict[str, int] = {
            "signals_generated": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "errors_caught": 0,
            "dropped_no_bars": 0,
            "dropped_below_threshold": 0,
            "dropped_pre_trade": 0,
        }

        # Skip instruments without FIGI (delisted shares, bonds handled by bond_cycle)
        figi = getattr(instrument, "figi", None)
        if not figi:
            _log.debug("skip_no_figi", symbol=getattr(instrument, "symbol", "?"))
            return stats

        seg_id = getattr(instrument, "segment_id", "") or "us_tech"
        symbol = getattr(instrument, "symbol", "?")

        try:
            # Convert limit (bar count) to date range for fetcher API
            end = now
            start = end - timedelta(days=_CANDLE_LOOKBACK * 2)  # ~2x for weekends/holidays
            candles: list[Candle] = fetcher.fetch_candles(  # type: ignore[attr-defined]
                symbol=symbol,
                start=start,
                end=end,
            )
        except InstrumentNotFoundError:
            _log.debug("skip_instrument_not_found", symbol=symbol)
            return stats
        except Exception:
            _log.exception("process_instrument: failed to fetch candles for %s", symbol)
            stats["errors_caught"] = 1
            return stats

        # DATA-01: Validate candles through DataNormalizer before any processing
        from finalayze.data.normalizer import DataNormalizer  # noqa: PLC0415

        normalizer = DataNormalizer(market_id=market_id, source="live")
        candles = normalizer.normalize_batch(candles)
        if not candles:
            _log.warning("all_candles_invalid", symbol=symbol, market=market_id)
            stats["dropped_no_bars"] = 1
            return stats

        # DATA-02: Skip instrument if latest candle is stale
        from finalayze.orchestration.trading_loop import TradingLoop  # noqa: PLC0415

        if TradingLoop._is_candle_stale(candles[-1].timestamp, _STALENESS_THRESHOLD_HOURS):
            _log.warning(
                "candle_data_stale",
                symbol=symbol,
                latest_ts=candles[-1].timestamp.isoformat(),
                threshold_hours=_STALENESS_THRESHOLD_HOURS,
            )
            return stats

        # Update health monitor feed timestamp on successful fetch
        if candles and self._health_monitor is not None:
            self._health_monitor.update_feed_timestamp(now)

        # Cache last price for per-position sector exposure calculation (SIZE-02)
        if candles:
            self._last_prices[symbol] = Decimal(str(candles[-1].close))

        # #157/#182: Check stop-losses against latest candle price
        if candles:
            current_price = candles[-1].close
            self._position_tracker.check_stop_losses(market_id, symbol, current_price)

        # PARITY-04: Skip signal generation for symbols stopped out this cycle
        if symbol in self._position_tracker.exited_symbols:
            _log.debug("skip_reentry_guard", symbol=symbol)
            return stats

        sentiment_score = self._sentiment_mgr.get_sentiment(seg_id, symbol)

        broker = self._broker_router.route(market_id)
        has_open_position = broker.has_position(symbol)

        # Retroactive stop: position open but no stop state (e.g. after container restart
        # with no DB snapshot). Compute ATR stop from current candles and register it
        # so check_stop_losses starts protecting the position from next cycle onward.
        if has_open_position and candles and not self._position_tracker.has_stop(symbol):
            from finalayze.execution.simulated_broker import (  # noqa: PLC0415
                StopLossState,
            )
            from finalayze.risk.stop_loss import compute_atr_stop_loss  # noqa: PLC0415

            is_moex = market_id == "moex"
            mult = _ATR_MULTIPLIER_MOEX if is_moex else _ATR_MULTIPLIER_US
            cur = Decimal(str(candles[-1].close))
            entry = self._position_tracker._entry_prices.get(symbol, cur)
            natural_stop = compute_atr_stop_loss(entry, candles, atr_multiplier=mult)
            if natural_stop is not None and mult > _ZERO:
                atr_val = (entry - natural_stop) / mult
                if cur >= natural_stop:
                    stop_price = natural_stop
                    trail_activated = False
                    highest = max(entry, cur)
                else:
                    # Already below natural stop — grace: 0.5 ATR below current
                    stop_price = max(cur - Decimal("0.5") * atr_val, _ZERO)
                    trail_activated = True
                    highest = cur
                strategy = self._position_tracker._entry_strategy.get(symbol, "retroactive")
                stop_state = StopLossState(
                    initial_stop=stop_price,
                    current_stop=stop_price,
                    highest_price=highest,
                    trail_activated=trail_activated,
                    activation_atr=Decimal("1.0"),
                    trail_atr=Decimal("1.5"),
                    entry_price=entry,
                    atr_value=atr_val,
                )
                self._position_tracker.register_entry(
                    symbol, entry, strategy, stop_state, market_id=market_id
                )
                _log.warning(
                    "stop_retroactive_set",
                    symbol=symbol,
                    stop_price=float(stop_price),
                    entry_price=float(entry),
                    trail_activated=trail_activated,
                    market=market_id,
                )

        signal = self._strategy.generate_signal(
            symbol,
            candles,
            seg_id,
            sentiment_score=sentiment_score,
            has_open_position=has_open_position,
        )
        if signal is None:
            _log.info("signal_dropped_below_threshold", symbol=symbol, segment=seg_id)
            stats["dropped_below_threshold"] = 1
            return stats

        # Skip BUY when position already open — prevent infinite accumulation
        if has_open_position and signal.direction == SignalDirection.BUY:
            _log.debug(
                "signal_skip_already_positioned",
                symbol=symbol,
                direction="BUY",
            )
            return stats

        stats["signals_generated"] = 1

        # Fire-and-forget signal persistence (PERSIST-02)
        if self._persistence is not None:
            self._persistence.persist_signal(signal)

        if self._metrics:
            self._metrics.record_signal(
                market=market_id,
                strategy=signal.strategy_name,
                direction=signal.direction.value,
            )

        _log.info(
            "signal_generated",
            symbol=symbol,
            direction=signal.direction.value,
            strategy=signal.strategy_name,
            confidence=round(signal.confidence, 3),
            sentiment=round(sentiment_score, 3),
            segment=seg_id,
            has_position=has_open_position,
            reasoning=signal.reasoning,
            features={k: round(v, 4) for k, v in signal.features.items()} or None,
        )

        # Use cached portfolio or return if unavailable
        if portfolio is None:
            return stats

        # #162: Use RollingKelly for position sizing
        from finalayze.risk.kelly import RollingKelly  # noqa: PLC0415

        kelly_sizer = self._position_tracker._kelly_sizer
        if isinstance(kelly_sizer, RollingKelly):
            kelly_fraction = kelly_sizer.optimal_fraction()
        else:
            kelly_fraction = Decimal(str(getattr(self._settings, "kelly_fraction", 0.5)))

        _log.debug(
            "kelly_sizing",
            symbol=symbol,
            kelly_fraction=float(kelly_fraction),
            equity=float(equity),
            cash=float(cash),
        )
        order = self._build_order(
            signal,
            level,
            equity,
            cash,
            candles,
            symbol,
            kelly_fraction,
            portfolio=portfolio,
            seg_id=seg_id,
        )
        if order is None:
            _log.info(
                "order_sizing_zero",
                symbol=symbol,
                direction=signal.direction.value,
                strategy=signal.strategy_name,
                reason="position size rounded to zero",
            )
            return stats

        # #141: Run PreTradeChecker before submitting
        order_value = order.quantity * (candles[-1].close if candles else _ZERO)
        open_position_count = len([q for q in portfolio.positions.values() if q > _ZERO])

        # 6A.4: Aggregate invested value across ALL markets for cross-market exposure
        from finalayze.markets.currency import CurrencyConverter  # noqa: PLC0415

        fx = CurrencyConverter(base_currency="USD")
        total_equity = self._compute_total_equity_base(fx)
        total_invested = _ZERO
        for m_id in (
            self._pre_trade_checker._symbol_limits.keys()
            if hasattr(self._pre_trade_checker, "_symbol_limits")
            else []
        ):
            m_equity = self._get_market_equity(m_id)
            if m_equity is None:
                continue
            m_broker = self._broker_router.route(m_id)
            m_portfolio = m_broker.get_portfolio()
            m_invested = max(m_equity - m_portfolio.cash, _ZERO)
            currency = _MARKET_CURRENCY.get(m_id, "USD")
            total_invested += fx.to_base(m_invested, currency)

        order_currency = _MARKET_CURRENCY.get(market_id, "USD")
        order_value_base = fx.to_base(order_value, order_currency)
        prospective_invested = total_invested + order_value_base
        cross_exposure: Decimal = (
            prospective_invested / total_equity if total_equity > _ZERO else _ZERO
        )
        try:
            _raw_max_exp = getattr(self._settings, "max_cross_market_exposure_pct", 0.80)
            max_exposure = Decimal(str(float(_raw_max_exp)))
        except (TypeError, ValueError):
            max_exposure = Decimal("0.80")

        # 6A.7: Detect day trades for PDT compliance
        is_day_trade = self._is_day_trade(order.symbol, order.side, market_id)

        # 6A.2: Compute sector exposure for concentration check (SIZE-02 fix)
        sector_exposure = _ZERO
        for pos_symbol, qty in portfolio.positions.items():
            if qty > _ZERO:
                # Use each position's own last known price, not current instrument's candle
                pos_price = self._get_last_price(pos_symbol)
                sector_exposure += qty * pos_price
        # Only pass if we have segment context
        seg_exposure: Decimal | None = sector_exposure if seg_id else None

        # PARITY-03: Gather all pre-trade check parameters
        # Check 9: stop_loss_price from trailing stop state (Plan 01)
        stop_loss_price = self._position_tracker.get_stop_loss_price(symbol)

        # Check 10: has_pending_order via broker
        has_pending = self._has_pending_order(symbol, market_id)

        # Check 12: regime_state from macro cache
        regime_state = self._get_regime_state()

        # Check 13: strategy_name from the signal
        strategy_name = signal.strategy_name

        # Check 14: open positions and correlations
        open_positions = [s for s, q in portfolio.positions.items() if q > _ZERO]
        correlations = self._get_correlations(open_positions)

        pre_result = self._pre_trade_checker.check(
            order_value=order_value,
            portfolio_equity=portfolio.equity,
            available_cash=portfolio.cash,
            open_position_count=open_position_count,
            market_id=market_id,
            dt=now,
            circuit_breaker_level=self._get_circuit_breaker_level(market_id),
            stop_loss_price=stop_loss_price,
            require_stop_loss=self._position_tracker.has_stop(symbol),
            has_pending_order=has_pending,
            symbol=symbol,
            cross_market_exposure_pct=cross_exposure,
            max_cross_market_exposure_pct=max_exposure,
            is_day_trade=is_day_trade,
            sector_exposure_value=seg_exposure,
            sector_id=seg_id,
            regime_state=regime_state,
            strategy_name=strategy_name,
            open_positions=open_positions,
            correlations=correlations,
        )

        if not pre_result.passed:
            _log.info(
                "pre_trade_rejected",
                symbol=symbol,
                direction=signal.direction.value,
                strategy=signal.strategy_name,
                violations=pre_result.violations,
            )
            stats["dropped_pre_trade"] = 1
            return stats

        # ALRT-02 (D-11/D-12/D-13/D-14): fire signal alert AFTER pre-trade pass,
        # BEFORE submit. Best-effort — never crashes the cycle.
        self._fire_signal_alert(
            signal=signal,
            market_id=market_id,
            symbol=symbol,
            broker=broker,
        )

        price = candles[-1].close if candles else _ZERO
        _log.info(
            "order_submitted",
            symbol=symbol,
            direction=order.side,
            quantity=int(order.quantity),
            price=float(price),
            value_rub=float(order.quantity * price),
            kelly=float(kelly_fraction),
            equity=float(equity),
            strategy=signal.strategy_name,
            market=market_id,
        )
        result = self._submit_order(
            order, market_id, candles=candles, strategy_name=signal.strategy_name
        )
        stats["orders_submitted"] = 1
        if result and result.get("filled"):
            stats["orders_filled"] = 1

        # 6A.7: Record day trade after successful order submission
        if is_day_trade:
            from finalayze.risk.pre_trade_check import PDTTracker  # noqa: PLC0415

            # Get PDT tracker from pre_trade_checker
            if hasattr(self._pre_trade_checker, "_pdt_tracker"):
                pdt_tracker = self._pre_trade_checker._pdt_tracker
                if isinstance(pdt_tracker, PDTTracker):
                    pdt_tracker.record_day_trade(now.date())

        return stats

    def _build_sizing_pipeline(self, segment_id: str) -> PositionSizingPipeline:
        """Build position sizing pipeline matching backtest engine step order.

        Pipeline order: VolTarget -> Regime -> [RubOilRegime] -> [BrentGate]
            -> [CBRRegime] -> [SectorAllocation] -> [Copula] -> [EVT] -> MetaLabel -> HardCaps
        (Kelly sizing is pre-applied to SizingContext.base_position upstream.)
        """
        from finalayze.risk.position_sizing_pipeline import (  # noqa: PLC0415
            BrentGateStep,
            CBRRegimeStep,
            CopulaStep,
            EVTStep,
            HardCapsStep,
            MetaLabelStep,
            PositionSizingPipeline,
            RegimeStep,
            RubOilRegimeStep,
            SectorAllocationStep,
            VolTargetStep,
        )

        steps: list[object] = [VolTargetStep(), RegimeStep()]

        # Add MOEX-specific steps when macro_cache provides data
        if self._macro_cache is not None and segment_id.startswith("ru_"):
            rub_oil_signal = getattr(self._macro_cache, "rub_oil_regime_signal", None)
            if rub_oil_signal is not None:
                steps.append(RubOilRegimeStep(rub_oil_signal, segment_id))
            brent_rub = getattr(self._macro_cache, "brent_rub_price", 0.0)
            if brent_rub > 0:
                steps.append(BrentGateStep(brent_rub, segment_id))
            yield_slope = getattr(self._macro_cache, "yield_slope_bps", 0.0)
            steps.append(CBRRegimeStep(yield_slope, segment_id))
            cbr_dir = getattr(self._macro_cache, "cbr_direction", "")
            if cbr_dir:
                steps.append(SectorAllocationStep(brent_rub, cbr_dir, segment_id))

        steps.append(CopulaStep())
        steps.append(EVTStep())
        steps.append(MetaLabelStep())
        steps.append(HardCapsStep())
        return PositionSizingPipeline(steps=steps)  # type: ignore[arg-type]

    @staticmethod
    def _compute_asset_vol(candles: list[Candle]) -> Decimal:
        """Compute annualized volatility from candle close prices."""
        if len(candles) < 2:  # noqa: PLR2004
            return Decimal("0.20")  # fallback
        import math  # noqa: PLC0415

        closes = [float(c.close) for c in candles]
        log_rets = [
            math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0
        ]
        if not log_rets:
            return Decimal("0.20")
        var = sum(r**2 for r in log_rets) / len(log_rets)
        annual_vol = math.sqrt(var * 252)
        return Decimal(str(round(annual_vol, 4)))

    def _get_regime_scale(self) -> Decimal:
        """Get current regime scale factor. 1.0 = risk-on, lower = risk-off."""
        if self._macro_cache is not None:
            regime = getattr(self._macro_cache, "regime_scale", None)
            if regime is not None:
                return Decimal(str(regime))
        return Decimal("1.0")

    def _has_pending_order(self, symbol: str, market_id: str) -> bool:
        """Check if broker has a pending (unfilled) order for symbol."""
        try:
            broker = self._broker_router.route(market_id)
            if hasattr(broker, "get_pending_orders"):
                pending = broker.get_pending_orders()
                return any(o.symbol == symbol for o in pending)
        except Exception:
            _log.debug("pending_order_check_failed", symbol=symbol)
        return False

    def _get_regime_state(self) -> RegimeState | None:
        """Get current regime state from macro cache."""
        if self._macro_cache is not None:
            return getattr(self._macro_cache, "regime_state", None)
        return None

    def _get_correlations(
        self,
        open_positions: list[str],  # noqa: ARG002
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise correlations for open positions.

        For live trading, correlation computation requires historical returns
        which we don't track yet. Return empty dict for now (check 14 passes through).
        TODO: Wire returns history for live correlation computation in future phase.
        """
        return {}

    def _build_order(
        self,
        signal: Signal,
        level: CircuitLevel,
        portfolio_equity: Decimal,
        available_cash: Decimal,
        candles: list[Candle],
        symbol: str,
        kelly_fraction: Decimal,
        *,
        portfolio: PortfolioState | None = None,
        seg_id: str = "us_tech",
    ) -> OrderRequest | None:
        """Build an order from signal, using PositionSizingPipeline for BUY orders.

        PARITY-01: BUY orders go through the same multi-step sizing pipeline as backtest.
        SIZE-01: SELL orders use actual held position quantity.
        SIZE-03: CAUTION threshold uses segment preset min_combined_confidence * 1.2.
        """
        from finalayze.risk.position_sizing_pipeline import SizingContext  # noqa: PLC0415

        side: Literal["BUY", "SELL"] = "BUY" if signal.direction == SignalDirection.BUY else "SELL"

        # SIZE-01: SELL orders use actual held quantity, skip pipeline sizing
        if signal.direction == SignalDirection.SELL:
            held = portfolio.positions.get(symbol, _ZERO) if portfolio is not None else _ZERO
            if held <= _ZERO:
                return None
            return self._OrderRequest(symbol=symbol, side=side, quantity=held)

        # SIZE-03: CAUTION threshold from segment preset (not hardcoded 0.5)
        if level == self._CircuitLevel.CAUTION:
            preset_conf = self._get_segment_min_confidence(seg_id)
            min_conf = preset_conf * _MIN_CONFIDENCE_BOOST
            if signal.confidence < min_conf:
                return None

        # PARITY-01: Build sizing pipeline and context (matching backtest engine)
        pipeline = self._build_sizing_pipeline(seg_id)
        asset_vol = self._compute_asset_vol(candles)
        regime_scale = self._get_regime_scale()
        ml_confidence = signal.features.get("ml_confidence") if signal.features else None

        _limits = self._settings.effective_risk_limits()
        min_pos = max(portfolio_equity * Decimal("0.005"), Decimal(500))

        context = SizingContext(
            equity=portfolio_equity,
            base_position=kelly_fraction * portfolio_equity,
            max_position_pct=Decimal(str(_limits.max_position_pct)),
            min_position_size=min_pos,
            asset_vol=asset_vol,
            target_vol=Decimal(str(getattr(self._settings, "target_vol", 0.15))),
            regime_scale=regime_scale,
            correlation_scale=Decimal("1.0"),
            returns_history=(),
            ml_confidence=ml_confidence,
        )

        order_value = pipeline.compute(context)
        if order_value <= _ZERO:
            return None

        # Cap by available cash
        order_value = min(order_value, available_cash)

        # CAUTION reduction (on top of pipeline)
        if level == self._CircuitLevel.CAUTION:
            order_value = order_value * _CAUTION_SIZE_FACTOR

        qty = (order_value / Decimal(str(candles[-1].close))) if candles else _ZERO
        qty = qty.quantize(Decimal(1))
        if qty <= _ZERO:
            return None

        return self._OrderRequest(symbol=symbol, side=side, quantity=qty)

    def _get_last_price(self, symbol: str) -> Decimal:
        """Return cached last price for a symbol, or _ZERO if unknown (SIZE-02)."""
        return self._last_prices.get(symbol, _ZERO)

    def _get_segment_min_confidence(self, seg_id: str) -> float:
        """Load min_combined_confidence from segment preset YAML (SIZE-03).

        Caches result to avoid re-reading YAML on every call.
        Falls back to 0.5 if preset not found.
        """
        if seg_id in self._segment_min_confidence:
            return self._segment_min_confidence[seg_id]

        import yaml  # noqa: PLC0415

        presets_dir = Path(__file__).parent.parent / "strategies" / "presets"
        path = presets_dir / f"{seg_id}.yaml"
        default_conf = 0.5
        try:
            with path.open() as f:
                config = yaml.safe_load(f)
            if isinstance(config, dict):
                result = float(config.get("min_combined_confidence", default_conf))
            else:
                result = default_conf
        except (FileNotFoundError, OSError):
            _log.warning("segment_preset_not_found", seg_id=seg_id, path=str(path))
            result = default_conf

        self._segment_min_confidence[seg_id] = result
        return result

    def get_entry_strategies(self) -> dict[str, str]:
        """Return a snapshot of {symbol: strategy_name} for currently open positions.

        Used by PresetApplicator to check position ownership before disabling a
        strategy via auto-apply.  Returns a copy so callers cannot mutate internal state.
        """
        return self._position_tracker.get_entry_strategies()

    def _extract_strategy_contribs(
        self,
        signal: Signal,
    ) -> list[tuple[str, float]]:
        """Return [(name, confidence)] sorted descending by confidence.

        Reads the per-strategy ``{name}_confidence`` keys that
        ``StrategyCombiner`` writes onto ``signal.features``. Excludes the ADX
        routing keys (``adx_*_confidence``) which are not contributing
        strategies. ALRT-02 D-14: caller (TelegramAlerter.on_signal_generated)
        truncates to top-3 + "(+N more)".
        """
        contribs: list[tuple[str, float]] = []
        for key, val in (signal.features or {}).items():
            if key.endswith("_confidence") and not key.startswith("adx_"):
                name = key[: -len("_confidence")]
                contribs.append((name, float(val)))
        contribs.sort(key=lambda t: -t[1])
        return contribs

    def _fire_signal_alert(
        self,
        *,
        signal: Signal,
        market_id: str,
        symbol: str,
        broker: object,
    ) -> None:
        """ALRT-02 (D-11/D-12/D-13/D-14): fire on_signal_generated with NEW/ADD/FLIP.

        Called from ``process_instrument`` AFTER pre-trade validation passes,
        BEFORE the order is submitted (D-12). Skips when:

        - signal.confidence < _MIN_ALERT_CONFIDENCE (D-13 noise gate)
        - self._alerter is None
        - broker.get_positions raises (broker outage — alert is best-effort)

        Position context (D-11):
        - qty == 0          => NEW
        - qty * direction same sign => ADD
        - qty * direction opposite  => FLIP

        A Telegram outage NEVER crashes the cycle (logged via _log.exception).
        """
        if signal.confidence < _MIN_ALERT_CONFIDENCE or self._alerter is None:
            return
        try:
            broker_positions = broker.get_positions()  # type: ignore[attr-defined]
        except Exception:
            _log.exception("signal_alert_get_positions_failed", symbol=symbol)
            return
        current_qty = broker_positions.get(symbol, _ZERO)
        if current_qty == _ZERO:
            position_context = "NEW"
        elif (current_qty > _ZERO and signal.direction == SignalDirection.BUY) or (
            current_qty < _ZERO and signal.direction == SignalDirection.SELL
        ):
            position_context = "ADD"
        else:
            position_context = "FLIP"
        strategy_contribs = self._extract_strategy_contribs(signal)
        try:
            self._alerter.on_signal_generated(
                symbol=symbol,
                market_id=market_id,
                side=signal.direction.value,
                confidence=float(signal.confidence),
                strategy_breakdown=strategy_contribs,
                position_context=position_context,
            )
        except Exception:
            _log.exception("signal_alert_fire_failed", symbol=symbol)

    def _submit_order(  # noqa: PLR0912
        self,
        order: OrderRequest,
        market_id: str,
        candles: list[Candle] | None = None,
        strategy_name: str = "",
    ) -> dict[str, Any]:
        """Submit order, set stop-loss on BUY fill, clear on SELL fill.

        Returns dict with fill information:
          - filled: bool
          - quantity: Decimal or None
          - fill_price: Decimal or None
        """
        from finalayze.risk.stop_loss import compute_atr_stop_loss  # noqa: PLC0415

        result_dict: dict[str, Any] = {
            "filled": False,
            "quantity": None,
            "fill_price": None,
        }

        try:
            result = self._broker_router.submit(order, market_id=market_id)
            if result.filled:
                result_dict["filled"] = True
                result_dict["quantity"] = result.quantity
                result_dict["fill_price"] = result.fill_price

                _log.info(
                    "order_executed",
                    symbol=order.symbol,
                    side=order.side,
                    qty=float(result.quantity),
                    fill_price=float(result.fill_price) if result.fill_price else None,
                    market=market_id,
                )
                self._alerter.on_trade_filled(result, market_id, broker=market_id)

                # Fire-and-forget order persistence (PERSIST-01)
                if self._persistence is not None:
                    self._persistence.persist_order(order, result, market_id)

                # Compute slippage in bps
                expected_price = candles[-1].close if candles else None
                if (
                    result.fill_price is not None
                    and expected_price is not None
                    and expected_price > 0
                ):
                    slippage_bps = float(
                        (result.fill_price - expected_price) / expected_price * 10000
                    )
                else:
                    slippage_bps = 0.0

                if self._sandbox_monitor is not None:
                    self._sandbox_monitor.record_slippage(slippage_bps)

                if self._metrics:
                    self._metrics.record_trade(
                        market=market_id,
                        side=order.side.lower(),
                        slippage_bps=slippage_bps,
                        fill_latency_seconds=0.0,
                    )
                # Track position ownership for PresetApplicator (APPLY-03)
                if order.side == "BUY":
                    self._position_tracker._entry_strategy[order.symbol] = strategy_name
                    # Wire stop-loss on BUY fill + track entry price for Kelly
                    if candles and result.fill_price is not None:
                        from finalayze.execution.simulated_broker import (  # noqa: PLC0415
                            StopLossState,
                        )

                        is_moex = market_id == "moex"
                        multiplier = _ATR_MULTIPLIER_MOEX if is_moex else _ATR_MULTIPLIER_US
                        stop = compute_atr_stop_loss(
                            result.fill_price, candles, atr_multiplier=multiplier
                        )
                        if stop is not None and multiplier > _ZERO:
                            # Derive ATR: stop = entry - mult * atr => atr = (entry - stop) / mult
                            atr_val = (result.fill_price - stop) / multiplier
                            stop_state = StopLossState(
                                initial_stop=stop,
                                current_stop=stop,
                                highest_price=result.fill_price,
                                trail_activated=False,
                                activation_atr=Decimal("1.0"),
                                trail_atr=Decimal("1.5"),
                                entry_price=result.fill_price,
                                atr_value=atr_val,
                            )
                            self._position_tracker.register_entry(
                                order.symbol,
                                result.fill_price,
                                strategy_name,
                                stop_state,
                                market_id=market_id,
                            )
                        else:
                            # If stop-loss computation failed, still track entry price
                            self._position_tracker._entry_prices[order.symbol] = result.fill_price
                # Update Kelly on SELL fill + clear stop-loss
                elif order.side == "SELL":
                    if result.fill_price is not None:
                        self._position_tracker._update_kelly(order.symbol, result.fill_price)
                    self._position_tracker.register_exit(order.symbol)
            else:
                _log.warning(
                    "order_rejected",
                    symbol=order.symbol,
                    side=order.side,
                    reason=result.reason,
                    market=market_id,
                )
                self._alerter.on_trade_rejected(order, result.reason)
                if self._metrics:
                    self._metrics.record_rejection(
                        market=market_id, reason=result.reason or "unknown"
                    )
        except Exception:
            _log.exception("_submit_order: order submission failed for %s", order.symbol)

        return result_dict

    def _is_day_trade(self, symbol: str, side: str, market_id: str) -> bool:
        """Return True if this order would open+close a position same day.

        A SELL of a position opened today constitutes a day trade.
        Simplified heuristic: a SELL order for a symbol with an existing
        position is flagged as a potential day trade. PDT is US-only.
        """
        if market_id != "us":
            return False
        broker = self._broker_router.route(market_id)
        return side == "SELL" and broker.has_position(symbol)

    def _get_circuit_breaker_level(self, market_id: str) -> Any:  # noqa: ARG002
        """Get current circuit breaker level for market.

        Imported from TradingLoop which has _circuit_breakers dict.
        Returns CircuitLevel or None if market not found.
        """
        # This should be passed in from TradingLoop or from a shared circuit_breakers dict
        # For now, return None (check will skip)
        return None

    def _get_market_equity(self, market_id: str) -> Decimal | None:
        """Return current portfolio equity for market, or None on failure.

        Injected from TradingLoop's _get_market_equity method.
        """
        try:
            broker = self._broker_router.route(market_id)
            portfolio = broker.get_portfolio()
            return Decimal(str(portfolio.equity))
        except Exception:
            return None

    def _compute_total_equity_base(self, fx: object) -> Decimal:
        """Sum equities across all markets, converting to base currency (USD)."""
        total = _ZERO
        for m_id in ["us", "moex"]:  # Inject known markets or use a parameter
            equity = self._get_market_equity(m_id)
            if equity is None:
                continue
            currency = _MARKET_CURRENCY.get(m_id, "USD")
            total += fx.to_base(equity, currency)  # type: ignore[attr-defined]
        return total

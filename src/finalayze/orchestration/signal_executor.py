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

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

from finalayze.core.schemas import SignalDirection
from finalayze.orchestration.cycle_stats import CycleStats
from finalayze.risk.exposure import ExposureCalculator
from finalayze.risk.pre_trade_check import CheckContext

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
# S1.4: BUY-fill stop multiplier is resolved via risk.stops.resolve_stop_atr_multiplier
# (shared with backtest path). Previously hardcoded _ATR_MULTIPLIER_US/_MOEX here.
_MARKET_CURRENCY: dict[str, str] = {"us": "USD", "moex": "RUB"}

_log = structlog.get_logger(__name__)


# ── Stage hand-off contexts (Phase 3) ──────────────────────────────────────
# Immutable payloads that flow between the three process_instrument stages.
# Each stage either returns one of these (proceed) or a CycleStats (early exit).


@dataclass(frozen=True, slots=True)
class _SignalContext:
    signal: Signal
    candles: list[Candle]
    seg_id: str
    symbol: str
    sentiment_score: float
    has_open_position: bool
    broker: Any  # routed broker for downstream alert/submit


@dataclass(frozen=True, slots=True)
class _OrderContext:
    signal: Signal
    order: OrderRequest
    candles: list[Candle]
    seg_id: str
    symbol: str
    broker: Any
    is_day_trade: bool
    kelly_fraction: Decimal


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

    process_instrument returns a CycleStats (see ``cycle_stats.py``) with
    per-instrument counters; TradingLoop aggregates these across the cycle.
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

    def process_instrument(
        self,
        instrument: object,
        market_id: str,
        level: CircuitLevel,
        fetcher: object,
        now: datetime,
        equity: Decimal,
        cash: Decimal,
        portfolio: PortfolioState | None,
    ) -> CycleStats:
        """Process one instrument through the pipeline.

        Stages:
          1a. _prepare_candles: fetch, normalize, staleness-check.
          1b. process_from_candles: stop checks + signal generation (public seam).
          2.  _stage_validate_and_size: sizing pipeline + pre-trade checks.
          3.  _stage_submit_and_record: alert + submit + PDT recording.

        TradingLoop aggregates the returned CycleStats across the cycle via
        the __add__ operator.
        """
        figi = getattr(instrument, "figi", None)
        if not figi:
            _log.debug("skip_no_figi", symbol=getattr(instrument, "symbol", "?"))
            return CycleStats()

        symbol = getattr(instrument, "symbol", "?")
        seg_id = getattr(instrument, "segment_id", "") or "us_tech"

        candles_result = self._prepare_candles(symbol, fetcher, now, market_id)
        if isinstance(candles_result, CycleStats):
            return candles_result

        sig_or_stats = self.process_from_candles(candles_result, symbol, seg_id, market_id)
        if isinstance(sig_or_stats, CycleStats):
            return sig_or_stats

        ord_or_stats = self._stage_validate_and_size(
            ctx=sig_or_stats,
            level=level,
            equity=equity,
            cash=cash,
            portfolio=portfolio,
            market_id=market_id,
            now=now,
        )
        if isinstance(ord_or_stats, CycleStats):
            return ord_or_stats

        return self._stage_submit_and_record(
            ctx=ord_or_stats,
            market_id=market_id,
            equity=equity,
            now=now,
        )

    def _prepare_candles(
        self,
        symbol: str,
        fetcher: object,
        now: datetime,
        market_id: str,
    ) -> list[Candle] | CycleStats:
        """Fetch, normalize, and staleness-check candles for one symbol.

        Returns the validated candle list on success, or a CycleStats early-exit
        value on any failure (not-found, fetch error, empty after normalize, stale).
        Side-effects: updates _health_monitor feed timestamp and _last_prices cache.
        """
        from finalayze.core.exceptions import InstrumentNotFoundError  # noqa: PLC0415

        try:
            end = now
            start = end - timedelta(days=_CANDLE_LOOKBACK * 2)  # ~2x for weekends/holidays
            candles: list[Candle] = fetcher.fetch_candles(  # type: ignore[attr-defined]
                symbol=symbol,
                start=start,
                end=end,
            )
        except InstrumentNotFoundError:
            _log.debug("skip_instrument_not_found", symbol=symbol)
            return CycleStats()
        except Exception:
            _log.exception("process_instrument: failed to fetch candles for %s", symbol)
            return CycleStats.error_caught()

        # DATA-01: Validate candles through DataNormalizer before any processing
        from finalayze.data.normalizer import DataNormalizer  # noqa: PLC0415

        normalizer = DataNormalizer(market_id=market_id, source="live")
        candles = normalizer.normalize_batch(candles)
        if not candles:
            _log.warning("all_candles_invalid", symbol=symbol, market=market_id)
            return CycleStats.no_bars()

        # DATA-02: Skip instrument if latest candle is stale
        from finalayze.orchestration.trading_loop import TradingLoop  # noqa: PLC0415

        if TradingLoop._is_candle_stale(candles[-1].timestamp, _STALENESS_THRESHOLD_HOURS):
            _log.warning(
                "candle_data_stale",
                symbol=symbol,
                latest_ts=candles[-1].timestamp.isoformat(),
                threshold_hours=_STALENESS_THRESHOLD_HOURS,
            )
            return CycleStats()

        if self._health_monitor is not None:
            self._health_monitor.update_feed_timestamp(now)
        self._last_prices[symbol] = Decimal(str(candles[-1].close))

        return candles

    def process_from_candles(
        self,
        candles: list[Candle],
        symbol: str,
        seg_id: str,
        market_id: str,
    ) -> _SignalContext | CycleStats:
        """Stop checks + signal generation from pre-validated candles.

        Public seam: callers can inject candles directly, bypassing fetch and
        normalisation. Useful for testing signal-threshold logic without wiring
        up a fetcher, DataNormalizer, or staleness check.

        Returns a _SignalContext (proceed to Stage 2) or a CycleStats early-exit.
        """
        # #157/#182: Check stop-losses against latest candle price
        self._position_tracker.check_stop_losses(market_id, symbol, candles[-1].close)

        # PARITY-04: Skip signal generation for symbols stopped out this cycle
        if symbol in self._position_tracker.exited_symbols:
            _log.debug("skip_reentry_guard", symbol=symbol)
            return CycleStats()

        sentiment_score = self._sentiment_mgr.get_sentiment(seg_id, symbol)
        broker = self._broker_router.route(market_id)
        has_open_position = broker.has_position(symbol)

        # Retroactive stop for orphaned positions (e.g. after container restart).
        if has_open_position:
            self._position_tracker.maybe_register_retroactive_stop(symbol, candles, market_id)

        signal = self._strategy.generate_signal(
            symbol,
            candles,
            seg_id,
            sentiment_score=sentiment_score,
            has_open_position=has_open_position,
        )
        if signal is None:
            _log.info("signal_dropped_below_threshold", symbol=symbol, segment=seg_id)
            return CycleStats.signal_dropped_threshold()

        if has_open_position and signal.direction == SignalDirection.BUY:
            _log.debug("signal_skip_already_positioned", symbol=symbol, direction="BUY")
            return CycleStats()

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
            strategy_payload={k: round(v, 4) for k, v in signal.strategy_payload.items()} or None,
            contributions={k: round(v, 4) for k, v in signal.contributions.items()} or None,
        )

        return _SignalContext(
            signal=signal,
            candles=candles,
            seg_id=seg_id,
            symbol=symbol,
            sentiment_score=sentiment_score,
            has_open_position=has_open_position,
            broker=broker,
        )

    def _stage_validate_and_size(
        self,
        *,
        ctx: _SignalContext,
        level: CircuitLevel,
        equity: Decimal,
        cash: Decimal,
        portfolio: PortfolioState | None,
        market_id: str,
        now: datetime,
    ) -> _OrderContext | CycleStats:
        """Stage 2: portfolio gate, sizing, exposure, pre-trade check."""
        if portfolio is None:
            return CycleStats.signal_generated()

        # #162: Kelly fraction from RollingKelly when available
        from finalayze.risk.kelly import RollingKelly  # noqa: PLC0415

        kelly_sizer = self._position_tracker._kelly_sizer
        kelly_fraction = (
            kelly_sizer.optimal_fraction()
            if isinstance(kelly_sizer, RollingKelly)
            else Decimal(str(getattr(self._settings, "kelly_fraction", 0.5)))
        )
        _log.debug(
            "kelly_sizing",
            symbol=ctx.symbol,
            kelly_fraction=float(kelly_fraction),
            equity=float(equity),
            cash=float(cash),
        )

        order = self._build_order(
            ctx.signal,
            level,
            equity,
            cash,
            ctx.candles,
            ctx.symbol,
            kelly_fraction,
            portfolio=portfolio,
            seg_id=ctx.seg_id,
        )
        if order is None:
            _log.info(
                "order_sizing_zero",
                symbol=ctx.symbol,
                direction=ctx.signal.direction.value,
                strategy=ctx.signal.strategy_name,
                reason="position size rounded to zero",
            )
            return CycleStats.signal_generated()

        order_value = order.quantity * (ctx.candles[-1].close if ctx.candles else _ZERO)
        open_position_count = len([q for q in portfolio.positions.values() if q > _ZERO])

        # 6A.4: Cross-market exposure
        symbol_limit_markets = (
            self._pre_trade_checker._symbol_limits.keys()
            if hasattr(self._pre_trade_checker, "_symbol_limits")
            else []
        )
        cross_exposure, max_exposure = ExposureCalculator(
            broker_router=self._broker_router,
            symbol_limit_markets=symbol_limit_markets,
            settings=self._settings,
            get_market_equity=self._get_market_equity,
        ).compute(market_id=market_id, order_value=order_value)

        # 6A.7: Detect day trades (also needed for post-fill PDT recording)
        is_day_trade = self._is_day_trade(order.symbol, order.side, market_id)

        pre_result = self._run_pre_trade_check(
            signal=ctx.signal,
            order_value=order_value,
            portfolio=portfolio,
            open_position_count=open_position_count,
            market_id=market_id,
            symbol=ctx.symbol,
            seg_id=ctx.seg_id,
            now=now,
            cross_exposure=cross_exposure,
            max_exposure=max_exposure,
            is_day_trade=is_day_trade,
        )

        if not pre_result.passed:
            _log.info(
                "pre_trade_rejected",
                symbol=ctx.symbol,
                direction=ctx.signal.direction.value,
                strategy=ctx.signal.strategy_name,
                violations=pre_result.violations,
            )
            return CycleStats.pre_trade_rejected()

        return _OrderContext(
            signal=ctx.signal,
            order=order,
            candles=ctx.candles,
            seg_id=ctx.seg_id,
            symbol=ctx.symbol,
            broker=ctx.broker,
            is_day_trade=is_day_trade,
            kelly_fraction=kelly_fraction,
        )

    def _stage_submit_and_record(
        self,
        *,
        ctx: _OrderContext,
        market_id: str,
        equity: Decimal,
        now: datetime,
    ) -> CycleStats:
        """Stage 3: alert + submit + PDT recording."""
        # ALRT-02 (D-11/D-12/D-13/D-14): fire signal alert AFTER pre-trade pass,
        # BEFORE submit. Best-effort — never crashes the cycle.
        self._fire_signal_alert(
            signal=ctx.signal,
            market_id=market_id,
            symbol=ctx.symbol,
            broker=ctx.broker,
        )

        price = ctx.candles[-1].close if ctx.candles else _ZERO
        _log.info(
            "order_submitted",
            symbol=ctx.symbol,
            direction=ctx.order.side,
            quantity=int(ctx.order.quantity),
            price=float(price),
            value_rub=float(ctx.order.quantity * price),
            kelly=float(ctx.kelly_fraction),
            equity=float(equity),
            strategy=ctx.signal.strategy_name,
            market=market_id,
        )
        result = self._submit_order(
            ctx.order, market_id, candles=ctx.candles, strategy_name=ctx.signal.strategy_name
        )
        filled = bool(result and result.get("filled"))

        # 6A.7: Record day trade after successful order submission
        if ctx.is_day_trade:
            from finalayze.risk.pre_trade_check import PDTTracker  # noqa: PLC0415

            if hasattr(self._pre_trade_checker, "_pdt_tracker"):
                pdt_tracker = self._pre_trade_checker._pdt_tracker
                if isinstance(pdt_tracker, PDTTracker):
                    pdt_tracker.record_day_trade(now.date())

        return CycleStats.order_submitted(filled=filled)

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
        ml_confidence = signal.metadata.ml_confidence

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

        Reads ``signal.contributions`` (per-strategy confidence written by
        ``StrategyCombiner``). ALRT-02 D-14: caller
        (TelegramAlerter.on_signal_generated) truncates to top-3 + "(+N more)".
        """
        contribs = [(name, float(val)) for name, val in signal.contributions.items()]
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
                        from finalayze.risk.stops import (  # noqa: PLC0415
                            resolve_stop_atr_multiplier,
                        )

                        multiplier = resolve_stop_atr_multiplier(strategy_name, market_id=market_id)
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

    def _compute_sector_exposure(self, portfolio: PortfolioState, seg_id: str) -> Decimal | None:
        """Sum value of all open positions using each one's last known price (SIZE-02).

        Returns None when no segment context is supplied — caller signals
        ``sector_exposure_value=None`` to the pre-trade check.
        """
        if not seg_id:
            return None
        total = _ZERO
        for pos_symbol, qty in portfolio.positions.items():
            if qty > _ZERO:
                total += qty * self._get_last_price(pos_symbol)
        return total

    def _run_pre_trade_check(
        self,
        *,
        signal: Signal,
        order_value: Decimal,
        portfolio: PortfolioState,
        open_position_count: int,
        market_id: str,
        symbol: str,
        seg_id: str,
        now: datetime,
        cross_exposure: Decimal,
        max_exposure: Decimal,
        is_day_trade: bool,
    ) -> Any:
        """Gather the 14 pre-trade fields and invoke the checker.

        Centralises the parameter assembly that previously lived inline in
        ``process_instrument``. Each field is fetched from the appropriate
        sub-component; the call site stays focused on flow control.
        """
        open_positions = [s for s, q in portfolio.positions.items() if q > _ZERO]
        ctx = CheckContext(
            order_value=order_value,
            portfolio_equity=portfolio.equity,
            available_cash=portfolio.cash,
            open_position_count=open_position_count,
            market_id=market_id,
            dt=now,
            circuit_breaker_level=self._get_circuit_breaker_level(market_id),
            stop_loss_price=self._position_tracker.get_stop_loss_price(symbol),
            require_stop_loss=self._position_tracker.has_stop(symbol),
            has_pending_order=self._has_pending_order(symbol, market_id),
            symbol=symbol,
            cross_market_exposure_pct=cross_exposure,
            max_cross_market_exposure_pct=max_exposure,
            is_day_trade=is_day_trade,
            sector_exposure_value=self._compute_sector_exposure(portfolio, seg_id),
            sector_id=seg_id,
            regime_state=self._get_regime_state(),
            strategy_name=signal.strategy_name,
            open_positions=open_positions,
            correlations=self._get_correlations(open_positions),
        )
        return self._pre_trade_checker.check(ctx)

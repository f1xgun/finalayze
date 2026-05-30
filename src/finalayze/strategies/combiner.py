"""Per-segment weighted strategy combiner (Layer 4)."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import yaml

from finalayze.core.schemas import (
    AdxRegime,
    Candle,
    EventType,
    MarketContext,
    Signal,
    SignalDirection,
    SignalMetadata,
)
from finalayze.strategies.adx import compute_adx
from finalayze.strategies.event_driven import EventDrivenStrategy
from finalayze.strategies.hrp import compute_hrp_weights

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"
_MIN_COMBINED_CONFIDENCE = Decimal("0.50")
_MIN_EXIT_CONFIDENCE = Decimal("0.25")
_BUY_SCORE = Decimal(1)
_SELL_SCORE = Decimal(-1)
_MAX_CONFIDENCE = Decimal("1.0")
_ZERO = Decimal(0)
_DEFAULT_WEIGHT = Decimal("1.0")

# ADX regime routing constants
_MOMENTUM_STRATEGIES = frozenset({"momentum", "dual_momentum"})
_MR_STRATEGIES = frozenset({"mean_reversion", "pairs", "ou_mean_reversion", "rsi2_connors"})

# Reinforcer-only strategies: can boost other signals but never create standalone trades.
# When only reinforcer strategies fire, the combined signal is suppressed.
_REINFORCER_STRATEGIES = frozenset({"ml_ensemble"})

# Event strategies: bypass ADX regime routing (always fire regardless of trend/MR regime).
# These strategies are calendar-driven, not momentum/MR, so ADX gating is irrelevant.
_EVENT_STRATEGIES = frozenset({"dividend_gap", "cbr_calendar"})

# Event confidence floor: when an event strategy fires, lower the threshold
# so that the signal is not diluted below this floor.
_EVENT_MIN_CONFIDENCE = Decimal("0.40")
_ADX_TREND_THRESHOLD = 35
_ADX_MR_THRESHOLD = 15

# HRP allocation constants
_HRP_MIN_HISTORY = 20

# Event types that trigger duplicate-signal suppression (EVNT-02). NONE is excluded.
# EARNINGS (Phase 60, INTG-01) is keyed here for clean dedup; event_driven stays
# ADX-neutral (NOT added to _EVENT_STRATEGIES) per RESEARCH A1 — revisit only if
# the proving run shows a trend/MR conflict.
_DEDUP_EVENT_TYPES: frozenset[EventType] = frozenset(
    {EventType.CBR, EventType.DIVIDEND, EventType.EARNINGS}
)


def _dedup_event_signals(
    signals_by_strategy: dict[str, tuple[Signal, Decimal]],
) -> set[str]:
    """Find strategy names to zero when duplicate CBR/dividend events detected.

    Returns set of strategy names whose weight should be zeroed.
    Per CONTEXT.md: same ticker + same cycle + same event_type -> zero lower-weight.
    """
    zeroed: set[str] = set()
    by_event: dict[EventType, list[tuple[str, Decimal]]] = {}
    for name, (sig, weight) in signals_by_strategy.items():
        event_type = sig.metadata.event_type
        if event_type in _DEDUP_EVENT_TYPES:
            by_event.setdefault(event_type, []).append((name, weight))

    for entries in by_event.values():
        if len(entries) < 2:  # noqa: PLR2004
            continue
        sorted_entries = sorted(entries, key=lambda e: e[1], reverse=True)
        for name, _ in sorted_entries[1:]:
            zeroed.add(name)

    return zeroed


class StrategyCombiner:
    """Combines multiple strategy signals using per-segment YAML weights."""

    def __init__(
        self,
        strategies: list[BaseStrategy],
        normalize_mode: str = "firing",
        allocation_mode: str = "static",
        market_context: MarketContext | None = None,
    ) -> None:
        self._strategies: dict[str, BaseStrategy] = {s.name: s for s in strategies}
        self._presets_dir = _PRESETS_DIR
        self._normalize_mode = normalize_mode
        self._allocation_mode = allocation_mode
        self._strategy_returns: dict[str, list[float]] = defaultdict(list)
        self._hrp_weights: dict[str, Decimal] | None = None
        self._adx_regimes: dict[str, str] = {}

        # Propagate market context to strategies that support it (duck typing)
        if market_context is not None:
            for strategy in strategies:
                if hasattr(strategy, "set_market_context"):
                    strategy.set_market_context(market_context)

    def record_strategy_return(self, strategy_name: str, ret: float) -> None:
        """Record a strategy return observation for HRP weight computation.

        Only accumulates data when allocation_mode is 'hrp'.
        """
        if self._allocation_mode != "hrp":
            return
        self._strategy_returns[strategy_name].append(ret)
        # Invalidate cached weights so they are recomputed next time
        self._hrp_weights = None

    def _has_hrp_weights(self) -> bool:
        """Return True if HRP weights can be computed (enough history)."""
        if self._allocation_mode != "hrp":
            return False
        if len(self._strategy_returns) < 2:  # noqa: PLR2004
            return False
        min_len = min(len(v) for v in self._strategy_returns.values())
        return min_len >= _HRP_MIN_HISTORY

    def _compute_hrp_overrides(self) -> dict[str, Decimal]:
        """Compute HRP weight overrides from recorded strategy returns."""
        if self._hrp_weights is not None:
            return self._hrp_weights
        names = sorted(self._strategy_returns.keys())
        min_len = min(len(self._strategy_returns[n]) for n in names)
        returns_matrix = [self._strategy_returns[n][:min_len] for n in names]
        raw_weights = compute_hrp_weights(returns_matrix, names)
        self._hrp_weights = {k: Decimal(str(v)) for k, v in raw_weights.items()}
        return self._hrp_weights

    @staticmethod
    def _resolve_weight(
        strategy_name: str,
        strategy_cfg: dict[str, object],
        weight_overrides: dict[str, Decimal] | None,
    ) -> Decimal:
        """Return the effective weight for a strategy."""
        if weight_overrides and strategy_name in weight_overrides:
            return weight_overrides[strategy_name]
        try:
            return Decimal(str(strategy_cfg.get("weight", "1.0")))
        except InvalidOperation:
            return _DEFAULT_WEIGHT

    def _compute_adx_regime(
        self,
        symbol: str,
        candles: list[Candle],
        config: dict[str, object],
    ) -> tuple[float | None, str]:
        """Compute ADX and determine regime: 'trend', 'mr', or 'ambiguous'.

        Simple threshold routing: ADX > trend_threshold is 'trend',
        ADX < mr_threshold is 'mr', otherwise 'ambiguous'. Per-symbol
        regime state is tracked for downstream consumers.

        Returns:
            Tuple of (adx_value, regime_label). adx_value is None when
            insufficient data or routing is disabled.
        """
        routing_cfg = config.get("regime_routing", {})
        if isinstance(routing_cfg, dict) and not routing_cfg.get("enabled", True):
            return None, "ambiguous"

        period = int(routing_cfg.get("adx_period", 14)) if isinstance(routing_cfg, dict) else 14
        trend_threshold = (
            int(routing_cfg.get("trend_threshold", _ADX_TREND_THRESHOLD))
            if isinstance(routing_cfg, dict)
            else _ADX_TREND_THRESHOLD
        )
        mr_threshold = (
            int(routing_cfg.get("mr_threshold", _ADX_MR_THRESHOLD))
            if isinstance(routing_cfg, dict)
            else _ADX_MR_THRESHOLD
        )

        closes = [float(c.close) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]

        adx_value = compute_adx(closes, highs, lows, period)
        if adx_value is None:
            return None, "ambiguous"  # fall back when insufficient data

        # Simple threshold routing (no hysteresis)
        if adx_value > trend_threshold:
            regime = "trend"
        elif adx_value < mr_threshold:
            regime = "mr"
        else:
            regime = "ambiguous"

        self._adx_regimes[symbol] = regime
        return adx_value, regime

    def _parse_config(self, config: dict[str, object]) -> tuple[dict[str, object], str, Decimal]:
        """Extract strategies config, normalize mode, and min confidence from config."""
        strategies_cfg_raw = config.get("strategies", {})
        strategies_cfg: dict[str, object] = (
            strategies_cfg_raw if isinstance(strategies_cfg_raw, dict) else {}
        )
        effective_normalize = str(config.get("normalize_mode", self._normalize_mode))
        try:
            effective_min_confidence = Decimal(
                str(config.get("min_combined_confidence", _MIN_COMBINED_CONFIDENCE))
            )
        except InvalidOperation:
            effective_min_confidence = _MIN_COMBINED_CONFIDENCE
        return strategies_cfg, effective_normalize, effective_min_confidence

    def _build_result(
        self,
        net: Decimal,
        *,
        metadata: SignalMetadata,
        contributions: dict[str, float],
        strategy_payload: dict[str, float],
        symbol: str,
        market_id: str,
        segment_id: str,
        dominant_strategy_name: str = "combined",
        signal_price: Decimal | None = None,
    ) -> Signal:
        """Create the combined Signal from net score and split payloads."""
        direction = SignalDirection.BUY if net > _ZERO else SignalDirection.SELL
        confidence = float(min(abs(net), _MAX_CONFIDENCE))
        strategy_count = len(contributions)
        return Signal(
            strategy_name=dominant_strategy_name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            metadata=metadata,
            contributions=contributions,
            strategy_payload=strategy_payload,
            reasoning=(
                f"Combined signal: net_score={float(net):.3f} from {strategy_count} strategies"
            ),
            signal_price=signal_price,
        )

    def _resolve_effective_overrides(
        self,
        weight_overrides: dict[str, Decimal] | None,
    ) -> tuple[dict[str, Decimal] | None, dict[str, Decimal] | None]:
        """Resolve weight overrides, preferring explicit overrides over HRP.

        Returns (effective_overrides, hrp_overrides) tuple.
        """
        hrp_overrides: dict[str, Decimal] | None = None
        if self._has_hrp_weights():
            hrp_overrides = self._compute_hrp_overrides()
        effective = weight_overrides if weight_overrides is not None else hrp_overrides
        return effective, hrp_overrides

    @staticmethod
    def _effective_threshold(
        config: dict[str, object],
        min_confidence: Decimal,
        has_open_position: bool,
        net: Decimal,
    ) -> Decimal:
        """Compute the effective confidence threshold, lowering for exit signals."""
        threshold = min_confidence
        if has_open_position and net < _ZERO:
            try:
                exit_conf = Decimal(str(config.get("min_exit_confidence", _MIN_EXIT_CONFIDENCE)))
            except InvalidOperation:
                exit_conf = _MIN_EXIT_CONFIDENCE
            threshold = min(min_confidence, exit_conf)
        return threshold

    def set_market_context(self, ctx: MarketContext) -> None:
        """Propagate MarketContext to strategies that support it (duck typing).

        This allows post-construction injection of benchmark/VIX data for
        cross-asset features — called by the backtest harness after the combiner
        is created but before ``generate_signal()`` runs.
        """
        for strategy in self._strategies.values():
            if hasattr(strategy, "set_market_context"):
                strategy.set_market_context(ctx)

    @property
    def presets_dir(self) -> Path:
        """Directory the combiner reads segment YAML presets from."""
        return self._presets_dir

    # ── Hooks for subclass extension (JournalingStrategyCombiner) ──────────

    def _on_generate_start(self, symbol: str, segment_id: str) -> None:
        """Hook: called at start of generate_signal, before strategy loop."""

    def _on_strategy_signal(
        self,
        name: str,
        strategy: BaseStrategy,
        signal: Signal | None,
        weight: Decimal,
    ) -> None:
        """Hook: called after each strategy fires (including None signals)."""

    def _on_normalized(self, net: float, strategy_payload: dict[str, float]) -> None:
        """Hook: called after normalization (or 0.0 on early return)."""

    def _on_final_signal(
        self,
        signal: Signal | None,
        strategy_payload: dict[str, float],
    ) -> None:
        """Hook: called with the final signal (or None if below threshold)."""

    # ── Core signal generation ──────────────────────────────────────────

    def generate_signal(  # noqa: PLR0912, PLR0915
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
        weight_overrides: dict[str, Decimal] | None = None,
        credibility: float = 1.0,
        event_type_code: float = 0.0,
    ) -> Signal | None:
        """Generate a combined signal by weighting enabled strategy signals.

        Uses ADX regime routing to gate strategy pools:
        - trend regime (ADX > _ADX_TREND_THRESHOLD, default 35): momentum only
        - mr regime (ADX < _ADX_MR_THRESHOLD, default 15): mean-reversion only
        - ambiguous (in-between): both pools fire, dominant pool wins
        Per-segment YAML can override the thresholds via ``adx_routing``.

        Args:
            weight_overrides: When provided, these weights are used instead of
                the YAML-configured weights for each named strategy.
            credibility: Source credibility [0.0, 1.0], threaded to event_driven only.
            event_type_code: Numeric event code for CBR/dividend dedup (0.0 = none).
        """
        self._on_generate_start(symbol, segment_id)

        config = self._load_config(segment_id)
        strategies_cfg, effective_normalize, effective_min_confidence = self._parse_config(config)
        effective_overrides, hrp_overrides = self._resolve_effective_overrides(weight_overrides)

        weighted_score = _ZERO
        total_weight = _ZERO
        total_enabled_weight = _ZERO
        data_ready_weight = _ZERO  # strategies registered + enabled (data-ready)
        contributions: dict[str, float] = {}  # {strategy_name: confidence}
        strategy_payload: dict[str, float] = {}  # display data (direction floats, hrp, etc.)
        firing_names: set[str] = set()
        dominant_strategy_name = "combined"
        dominant_contribution = _ZERO
        collected: dict[str, tuple[Signal, Decimal]] = {}

        # ADX regime routing
        adx_value, regime = self._compute_adx_regime(symbol, candles, config)
        adx_regime_enum: AdxRegime | None = (
            AdxRegime(regime) if regime in {r.value for r in AdxRegime} else None
        )

        # Per-pool score tracking for ambiguous regime dominant-pool-wins logic
        trend_score = _ZERO
        mr_score = _ZERO
        neutral_score = _ZERO
        trend_weight = _ZERO
        mr_weight = _ZERO
        neutral_weight = _ZERO
        trend_pool_fired = False
        mr_pool_fired = False

        for strategy_name, strategy_cfg in strategies_cfg.items():
            if not isinstance(strategy_cfg, dict):
                continue
            if not strategy_cfg.get("enabled", True):
                continue

            weight = self._resolve_weight(strategy_name, strategy_cfg, effective_overrides)
            total_enabled_weight += weight  # all enabled strategies in config
            strategy = self._strategies.get(strategy_name)
            if strategy is None:
                continue
            data_ready_weight += weight  # registered + enabled (actually called)

            # ADX regime gating: skip strategies that belong to the wrong pool
            is_trend = strategy_name in _MOMENTUM_STRATEGIES
            is_mr = strategy_name in _MR_STRATEGIES
            is_event = strategy_name in _EVENT_STRATEGIES

            if regime == "trend" and is_mr and not is_event:
                self._on_strategy_signal(strategy_name, strategy, None, weight)
                continue  # skip MR strategies in trending market
            if regime == "mr" and is_trend and not is_event:
                self._on_strategy_signal(strategy_name, strategy, None, weight)
                continue  # skip trend strategies in range-bound market

            if strategy_name == "event_driven" and isinstance(strategy, EventDrivenStrategy):
                signal = strategy.generate_signal(
                    symbol,
                    candles,
                    segment_id,
                    sentiment_score=sentiment_score,
                    has_open_position=has_open_position,
                    credibility=credibility,
                    event_type_code=event_type_code,
                )
            else:
                signal = strategy.generate_signal(
                    symbol,
                    candles,
                    segment_id,
                    sentiment_score=sentiment_score,
                    has_open_position=has_open_position,
                )
            self._on_strategy_signal(strategy_name, strategy, signal, weight)
            if signal is None:
                continue

            score = _BUY_SCORE if signal.direction == SignalDirection.BUY else _SELL_SCORE
            contribution = score * Decimal(str(signal.confidence)) * weight
            weighted_score += contribution
            total_weight += weight

            # Track per-pool scores for ambiguous regime
            if is_trend:
                trend_score += contribution
                trend_weight += weight
                trend_pool_fired = True
            elif is_mr:
                mr_score += contribution
                mr_weight += weight
                mr_pool_fired = True
            else:
                neutral_score += contribution
                neutral_weight += weight

            if abs(contribution) > abs(dominant_contribution):
                dominant_contribution = contribution
                dominant_strategy_name = strategy_name
            contributions[strategy_name] = signal.confidence
            firing_names.add(strategy_name)
            strategy_payload[f"{strategy_name}_direction"] = (
                1.0 if signal.direction == SignalDirection.BUY else -1.0
            )
            collected[strategy_name] = (signal, weight)

        # EVNT-02: CBR/dividend duplicate-signal suppression
        zeroed = _dedup_event_signals(collected)
        if zeroed:
            for zname in zeroed:
                zsig, zweight = collected[zname]
                zscore = _BUY_SCORE if zsig.direction == SignalDirection.BUY else _SELL_SCORE
                zcontrib = zscore * Decimal(str(zsig.confidence)) * zweight
                weighted_score -= zcontrib
                total_weight -= zweight

        # Ambiguous regime: dominant pool wins when both pools fired
        if regime == "ambiguous" and trend_pool_fired and mr_pool_fired:
            if abs(trend_score) >= abs(mr_score):
                # Keep trend + neutral, remove MR contributions
                weighted_score = trend_score + neutral_score
                total_weight = trend_weight + neutral_weight
            else:
                # Keep MR + neutral, remove trend contributions
                weighted_score = mr_score + neutral_score
                total_weight = mr_weight + neutral_weight

        if total_weight == _ZERO:
            self._on_normalized(0.0, strategy_payload)
            self._on_final_signal(None, strategy_payload)
            return None

        # Reinforcer-only check: if every firing strategy is a reinforcer, suppress the signal.
        if firing_names and firing_names <= _REINFORCER_STRATEGIES:
            self._on_normalized(0.0, strategy_payload)
            self._on_final_signal(None, strategy_payload)
            return None

        if effective_normalize == "total":
            denominator = total_enabled_weight
        elif effective_normalize == "active":
            denominator = data_ready_weight if data_ready_weight > _ZERO else total_enabled_weight
        else:  # "firing" (default)
            denominator = total_weight
        if denominator == _ZERO:
            self._on_normalized(0.0, strategy_payload)
            self._on_final_signal(None, strategy_payload)
            return None
        net = weighted_score / denominator

        # Add HRP weight features when using HRP allocation
        if hrp_overrides is not None:
            for sname, sweight in hrp_overrides.items():
                strategy_payload[f"hrp_weight_{sname}"] = float(sweight)

        self._on_normalized(float(net), strategy_payload)

        threshold = self._effective_threshold(
            config, effective_min_confidence, has_open_position, net
        )

        # Event strategy confidence floor: when an event strategy fires,
        # lower threshold so calendar-driven signals are not diluted.
        has_event_firing = bool(firing_names & _EVENT_STRATEGIES)
        if has_event_firing and threshold > _EVENT_MIN_CONFIDENCE:
            threshold = _EVENT_MIN_CONFIDENCE

        if abs(net) < threshold:
            # Log when strategies fired but combined score was below threshold
            firing = {
                name: {
                    "direction": (
                        "BUY" if strategy_payload.get(f"{name}_direction", 0) > 0 else "SELL"
                    ),
                    "confidence": contributions.get(name, 0.0),
                }
                for name in firing_names
            }
            if firing:
                logger.debug(
                    "signals_below_threshold",
                    symbol=symbol,
                    firing_strategies=firing,
                    net_score=float(net),
                    threshold=float(threshold),
                )
            self._on_final_signal(None, strategy_payload)
            return None

        metadata = SignalMetadata(
            adx_value=adx_value,
            adx_regime=adx_regime_enum,
        )
        result = self._build_result(
            net,
            metadata=metadata,
            contributions=contributions,
            strategy_payload=strategy_payload,
            symbol=symbol,
            market_id=candles[0].market_id,
            segment_id=segment_id,
            dominant_strategy_name=dominant_strategy_name,
            signal_price=Decimal(str(candles[-1].close)),
        )
        self._on_final_signal(result, strategy_payload)
        return result

    def _load_config(self, segment_id: str) -> dict[str, object]:
        """Load segment YAML preset, returning an empty dict if not found or malformed."""
        try:
            path = self._presets_dir / f"{segment_id}.yaml"
            with path.open() as f:
                result = yaml.safe_load(f)
            return dict(result) if isinstance(result, dict) else {}
        except (FileNotFoundError, OSError, yaml.YAMLError):
            return {}

    def invalidate_segment_cache(self, segment_id: str) -> None:
        """Invalidate any cached preset configuration for the given segment.

        Currently ``_load_config`` reads YAML from disk on every call, so there
        is no in-memory cache to clear.  This method exists as a forward-
        compatibility hook: callers (e.g. ``PresetApplicator``) should invoke it
        after modifying a preset file so that any future caching addition is
        automatically handled.

        Args:
            segment_id: The segment whose cached config should be discarded.
        """
        # No-op: _load_config reads from disk on every invocation.
        # If a cache is added in the future, clear it here.
        _log = structlog.get_logger()
        _log.debug("segment_cache_invalidated", segment_id=segment_id, cache_exists=False)

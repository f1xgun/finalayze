"""Drawdown monitors (Layer 4).

Contains:
- ``DrawdownMonitor``: Simple rolling peak-to-trough drawdown monitor
  with a single equity stream and configurable threshold.
- ``DrawdownStatus``: Frozen snapshot of multi-layer drawdown state.
- ``LayeredDrawdownMonitor``: Coordinates per-layer and portfolio-level
  drawdown monitoring for the multi-asset portfolio architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import structlog

from finalayze.risk.layer_circuit_breaker import (
    LAYER_CIRCUIT_CONFIGS,
    PORTFOLIO_L3_THRESHOLD,
    CircuitLevel,
    LayerCircuitBreaker,
    LayerCircuitConfig,
    PortfolioCircuitBreaker,
)

_ZERO = Decimal(0)
_ONE = Decimal(1)

logger = structlog.get_logger()

# Layers that get force-liquidated on portfolio breach (core is exempt)
_NON_CORE_LAYERS = frozenset({"strategic", "tactical", "short"})


# ---------------------------------------------------------------------------
# Simple drawdown monitor (pre-existing)
# ---------------------------------------------------------------------------


class DrawdownMonitor:
    """Monitors portfolio drawdown and triggers at threshold.

    A new equity peak updates the baseline.  Once triggered, the flag
    stays set until :meth:`reset` is called explicitly (e.g. at the
    start of a new backtest run).
    """

    _DEFAULT_THRESHOLD = Decimal("0.12")  # 12%

    def __init__(self, threshold: Decimal = _DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._peak_equity: Decimal = _ZERO
        self._current_equity: Decimal = _ZERO
        self._triggered = False

    def update(self, current_equity: Decimal) -> bool:
        """Update with current equity.

        Returns ``True`` if drawdown threshold is breached on this call.
        """
        self._current_equity = current_equity

        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self._triggered = False

        if self._peak_equity > _ZERO:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self._threshold:
                self._triggered = True
                return True
        return False

    @property
    def triggered(self) -> bool:
        """Whether the drawdown threshold has been breached."""
        return self._triggered

    @property
    def current_drawdown(self) -> Decimal:
        """Current drawdown percentage from peak (as a Decimal ratio)."""
        if self._peak_equity <= _ZERO:
            return _ZERO
        return (self._peak_equity - self._current_equity) / self._peak_equity

    def reset(self) -> None:
        """Reset for new backtest run."""
        self._peak_equity = _ZERO
        self._current_equity = _ZERO
        self._triggered = False


# ---------------------------------------------------------------------------
# Multi-layer drawdown coordinator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawdownStatus:
    """Snapshot of drawdown state across all layers and portfolio."""

    layer_levels: dict[str, CircuitLevel]  # layer_id -> current circuit level
    layer_drawdowns: dict[str, Decimal]  # layer_id -> current DD as decimal
    portfolio_dd: Decimal  # total portfolio DD
    portfolio_breach: bool  # True if portfolio L3 triggered
    layers_to_liquidate: list[str]  # layers that should be force-liquidated
    sizing_multipliers: dict[str, Decimal]  # layer_id -> effective multiplier


class LayeredDrawdownMonitor:
    """Coordinates per-layer and portfolio-level drawdown monitoring.

    Each layer has its own ``LayerCircuitBreaker`` with independent
    peak-to-trough tracking.  A single ``PortfolioCircuitBreaker`` monitors
    the sum of all layer equities against a 10% threshold.

    Usage in backtest loop::

        monitor = LayeredDrawdownMonitor()
        for bar in bars:
            layer_equities = {"core": ..., "strategic": ..., ...}
            status = monitor.update(layer_equities)
            if status.portfolio_breach:
                # liquidate strategic + tactical + short
                ...
            for layer_id, mult in status.sizing_multipliers.items():
                # use mult for position sizing
                ...
    """

    def __init__(
        self,
        layer_configs: dict[str, LayerCircuitConfig] | None = None,
        portfolio_threshold: Decimal = PORTFOLIO_L3_THRESHOLD,
    ) -> None:
        configs = layer_configs or LAYER_CIRCUIT_CONFIGS
        self._layer_breakers: dict[str, LayerCircuitBreaker] = {
            layer_id: LayerCircuitBreaker(config) for layer_id, config in configs.items()
        }
        self._portfolio_breaker = PortfolioCircuitBreaker(threshold=portfolio_threshold)
        # Own peak tracking for DD computation (avoids accessing private state)
        self._portfolio_peak: Decimal = _ZERO
        self._portfolio_current: Decimal = _ZERO

    def update(self, layer_equities: dict[str, Decimal]) -> DrawdownStatus:
        """Update all circuit breakers with current layer equities.

        Args:
            layer_equities: mapping of layer_id -> current equity value.
                Unknown layer IDs (not in the configured layer breakers) are
                logged as a warning and skipped.

        Returns:
            DrawdownStatus with current state across all layers.
        """
        # 1. Update each layer breaker with its equity (if provided)
        layer_levels: dict[str, CircuitLevel] = {}
        layer_drawdowns: dict[str, Decimal] = {}

        for layer_id, breaker in self._layer_breakers.items():
            if layer_id in layer_equities:
                breaker.update(layer_equities[layer_id])
            layer_levels[layer_id] = breaker.level
            layer_drawdowns[layer_id] = breaker.drawdown_pct

        # Log warnings for unknown layers
        for layer_id in layer_equities:
            if layer_id not in self._layer_breakers:
                logger.warning(
                    "unknown_layer_in_equities",
                    layer_id=layer_id,
                    known_layers=list(self._layer_breakers.keys()),
                )

        # 2. Compute total equity from known layers only
        total_equity = sum(
            (layer_equities[lid] for lid in self._layer_breakers if lid in layer_equities),
            _ZERO,
        )

        # 3. Update portfolio breaker and own peak tracking
        self._portfolio_current = total_equity
        self._portfolio_peak = max(self._portfolio_peak, total_equity)
        self._portfolio_breaker.update(total_equity)
        portfolio_breach = self._portfolio_breaker.is_triggered
        layers_to_liquidate = list(self._portfolio_breaker.layers_to_liquidate)

        # 4. Compute portfolio DD from own tracking
        portfolio_dd = self._portfolio_dd()

        # 5. Compute effective sizing multipliers
        sizing_multipliers: dict[str, Decimal] = {}
        for layer_id, breaker in self._layer_breakers.items():
            layer_mult = breaker.sizing_multiplier()

            if portfolio_breach and layer_id in _NON_CORE_LAYERS:
                # Non-core layers get zeroed on portfolio breach
                sizing_multipliers[layer_id] = _ZERO
            elif portfolio_breach and layer_id not in _NON_CORE_LAYERS:
                # Core (and any custom non-standard layers) keep layer multiplier
                sizing_multipliers[layer_id] = layer_mult
            else:
                # No portfolio breach: use layer-level multiplier
                sizing_multipliers[layer_id] = min(layer_mult, _ONE)

        return DrawdownStatus(
            layer_levels=layer_levels,
            layer_drawdowns=layer_drawdowns,
            portfolio_dd=portfolio_dd,
            portfolio_breach=portfolio_breach,
            layers_to_liquidate=layers_to_liquidate,
            sizing_multipliers=sizing_multipliers,
        )

    def reset(self) -> None:
        """Reset all circuit breakers and peak tracking."""
        for breaker in self._layer_breakers.values():
            breaker.reset()
        self._portfolio_breaker.reset()
        self._portfolio_peak = _ZERO
        self._portfolio_current = _ZERO

    @property
    def is_portfolio_breached(self) -> bool:
        """Whether the portfolio-level circuit breaker has been triggered."""
        return self._portfolio_breaker.is_triggered

    def _portfolio_dd(self) -> Decimal:
        """Compute the current portfolio drawdown as a decimal fraction."""
        if self._portfolio_peak <= _ZERO:
            return _ZERO
        return (self._portfolio_peak - self._portfolio_current) / self._portfolio_peak

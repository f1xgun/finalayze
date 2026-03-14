"""Bond cycle processor — orchestrates bond trading across portfolio layers (Layer 6).

NOTE: This module lives in core/ for import convenience but is architecturally
Layer 6 — it imports from strategies (L4), risk (L4), data (L2), execution (L5).
See docs/architecture/DEPENDENCY_LAYERS.md.

Processing order per layer (validated by risk review):
1. Aggregate bond breaker check
2. Per-layer circuit breaker check
3. Yield stop evaluation on existing positions → forced SELL signals
4. Execute SELL signals (frees DV01 budget)
5. Generate new strategy signals
6. DV01/EqualWeight sizing against updated budget
7. Bond pre-trade validation
8. Execute new BUY orders
9. Update LayerLedger
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from finalayze.core.alerts import TelegramAlerter
    from finalayze.core.layer_ledger import LayerLedger
    from finalayze.core.schemas import LayerConfig, PortfolioLayer, Signal
    from finalayze.data.fetchers.cbr import MacroSnapshot
    from finalayze.data.macro_cache import MacroCacheService
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.markets.instruments import InstrumentRegistry
    from finalayze.ml.registry import MLModelRegistry
    from finalayze.risk.dv01_sizing import DV01BudgetStep, EqualWeightBondSizer
    from finalayze.risk.layer_circuit_breaker import AggregateBondBreaker, BondLayerBreaker
    from finalayze.risk.yield_stop import YieldStop

_log = structlog.get_logger()

_BOND_MARKET_KEY = "moex_bonds"


@dataclass
class LayerResult:
    """Result of processing a single portfolio layer."""

    layer: PortfolioLayer
    signals: int = 0
    executed: int = 0
    exits: int = 0
    halted: bool = False
    error: bool = False


@dataclass
class BondCycleResult:
    """Result of a full bond trading cycle."""

    layer_results: list[LayerResult] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""

    def to_log_dict(self) -> dict[str, Any]:
        """Convert to dict for structlog."""
        processed = [r for r in self.layer_results if not r.halted and not r.error]
        halted = [r for r in self.layer_results if r.halted]
        errored = [r for r in self.layer_results if r.error]
        return {
            "skipped": self.skipped,
            "reason": self.reason,
            "layers_processed": len(processed),
            "layers_halted": len(halted),
            "layers_errored": len(errored),
            "total_signals": sum(r.signals for r in processed),
            "total_executed": sum(r.executed for r in processed),
            "total_exits": sum(r.exits for r in processed),
        }


class BondCycleProcessor:
    """Processes bond strategy signals across all 4 portfolio layers."""

    def __init__(
        self,
        layer_configs: dict[PortfolioLayer, LayerConfig],
        layer_ledgers: dict[PortfolioLayer, LayerLedger],
        layer_breakers: dict[PortfolioLayer, BondLayerBreaker],
        aggregate_breaker: AggregateBondBreaker,
        strategies: dict[PortfolioLayer, list[Any]],
        macro_cache: MacroCacheService,
        dv01_sizer: DV01BudgetStep,
        equal_weight_sizer: EqualWeightBondSizer,
        yield_stops: dict[PortfolioLayer, YieldStop],
        broker_router: BrokerRouter,
        instrument_registry: InstrumentRegistry,
        fetcher: Any,
        alerter: TelegramAlerter,
        ml_registry: MLModelRegistry | None = None,
    ) -> None:
        self._layer_configs = layer_configs
        self._layer_ledgers = layer_ledgers
        self._layer_breakers = layer_breakers
        self._aggregate_breaker = aggregate_breaker
        self._strategies = strategies
        self._macro_cache = macro_cache
        self._dv01_sizer = dv01_sizer
        self._equal_weight_sizer = equal_weight_sizer
        self._yield_stops = yield_stops
        self._broker_router = broker_router
        self._registry = instrument_registry
        self._fetcher = fetcher
        self._alerter = alerter
        self._ml_registry = ml_registry

    def run_cycle(self) -> BondCycleResult:
        """Execute one bond trading cycle across all layers. SYNC."""
        macro = self._macro_cache.get()
        if macro is None:
            _log.warning("bond_cycle_skipped", reason="no macro data")
            return BondCycleResult(skipped=True, reason="no macro data")

        if not self._aggregate_breaker.check():
            _log.warning("aggregate_bond_breaker_halted")
            self._alerter.send_alert("Bond portfolio HALTED — aggregate drawdown limit reached")
            return BondCycleResult(skipped=True, reason="aggregate breaker halted")

        results: list[LayerResult] = []
        for layer, config in self._layer_configs.items():
            try:
                if not self._layer_breakers[layer].check():
                    results.append(LayerResult(layer=layer, halted=True))
                    continue
                result = self._process_layer(layer, config, macro)
                results.append(result)
            except Exception:
                _log.exception("bond_layer_failed", layer=layer.value)
                results.append(LayerResult(layer=layer, error=True))

        return BondCycleResult(layer_results=results)

    def _process_layer(
        self,
        layer: PortfolioLayer,
        config: LayerConfig,
        macro: MacroSnapshot,
    ) -> LayerResult:
        """Process all bond instruments for a single layer."""
        from datetime import UTC, datetime, timedelta  # noqa: PLC0415

        bonds = self._get_layer_instruments(layer, config)
        ledger = self._layer_ledgers[layer]
        yield_stop = self._yield_stops[layer]

        # Step 1: Yield stop evaluation on existing positions
        exit_count = self._process_yield_stops(layer, ledger, yield_stop)

        # Step 2: Generate new strategy signals
        new_signals: list[Signal] = []
        # Fetch candles once per bond (90d window for technical indicators)
        now = datetime.now(tz=UTC)
        candle_start = now - timedelta(days=90)
        for bond in bonds:
            try:
                candles = self._fetcher.fetch_candles(
                    symbol=bond.symbol,
                    start=candle_start,
                    end=now,
                    timeframe="1d",
                )
            except Exception:
                _log.warning(
                    "bond_candle_fetch_failed",
                    symbol=bond.symbol,
                    layer=layer.value,
                )
                continue
            if not candles:
                continue
            for strategy in self._strategies.get(layer, []):
                signal = strategy.generate_signal(
                    symbol=bond.symbol,
                    candles=candles,
                    open_positions=dict(ledger.positions),
                    bar_idx=len(candles) - 1,
                    key_rate=macro.key_rate,
                    ruonia_7d_avg=macro.ruonia_7d_avg,
                    cpi_yoy=macro.cpi_yoy,
                    last_cbr_decision=macro.last_cbr_decision,
                )
                if signal is not None:
                    new_signals.append(signal)

        # Step 3: ML filter (no-op if ml_registry is None)
        new_signals = self._apply_ml_filter(new_signals, layer, macro)

        # Step 4: Size and execute
        executed = 0
        for signal in new_signals:
            if self._size_and_execute(signal, layer, ledger):
                executed += 1

        # Step 5: Log for training data
        self._log_signals(new_signals, layer, macro)

        return LayerResult(
            layer=layer,
            exits=exit_count,
            signals=len(new_signals),
            executed=executed,
        )

    def _get_layer_instruments(
        self,
        layer: PortfolioLayer,  # noqa: ARG002
        config: LayerConfig,
    ) -> list[Any]:
        """Get bond instruments appropriate for this layer."""
        all_bonds = self._registry.list_by_type("moex", "bond")
        allowed_types = config.allowed_instrument_types
        if "bond" not in allowed_types:
            return []
        return all_bonds

    def _process_yield_stops(
        self,
        layer: PortfolioLayer,  # noqa: ARG002
        ledger: LayerLedger,  # noqa: ARG002
        yield_stop: YieldStop,  # noqa: ARG002
    ) -> int:
        """Check yield stops on existing positions, execute exits. Returns count."""
        # In sandbox/live, yield stop requires current YTM which needs market data.
        # For now, log and return 0. Full implementation needs candle-to-YTM conversion.
        return 0

    def _apply_ml_filter(
        self,
        signals: list[Signal],
        layer: PortfolioLayer,  # noqa: ARG002
        macro: MacroSnapshot,  # noqa: ARG002
    ) -> list[Signal]:
        """Apply ML model to filter/adjust signals. No-op if ml_registry is None."""
        if self._ml_registry is None:
            return signals
        return signals

    def _size_and_execute(
        self,
        signal: Signal,
        layer: PortfolioLayer,  # noqa: ARG002
        ledger: LayerLedger,  # noqa: ARG002
    ) -> bool:
        """Size a signal and submit order. Returns True if executed."""
        # Placeholder — full implementation in subsequent task
        _log.info(
            "bond_signal_generated",
            symbol=signal.symbol,
            direction=signal.direction.value,
            confidence=signal.confidence,
            strategy=signal.strategy_name,
        )
        return False

    def _log_signals(
        self,
        signals: list[Signal],
        layer: PortfolioLayer,
        macro: MacroSnapshot,
    ) -> None:
        """Log signals with features and macro context for future ML training data."""
        for signal in signals:
            _log.info(
                "bond_signal",
                symbol=signal.symbol,
                layer=layer.value,
                direction=signal.direction.value,
                confidence=signal.confidence,
                strategy=signal.strategy_name,
                features=signal.features,
                key_rate=str(macro.key_rate),
                ruonia=str(macro.ruonia_7d_avg),
            )

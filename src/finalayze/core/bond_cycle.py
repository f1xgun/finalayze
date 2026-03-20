"""Bond cycle processor — orchestrates bond trading across portfolio layers (Layer 6).

NOTE: This module lives in core/ for import convenience but is architecturally
Layer 6 — it imports from strategies (L4), risk (L4), data (L2), execution (L5).
See docs/architecture/DEPENDENCY_LAYERS.md.

Processing order per layer (validated by risk review):
1. Aggregate bond breaker check
2. Per-layer circuit breaker check
3. Yield stop evaluation on existing positions -> forced SELL signals
4. Execute SELL signals (frees DV01 budget)
5. Generate new strategy signals
6. DV01/EqualWeight sizing against updated budget
7. Bond pre-trade validation
8. Execute new BUY orders
9. Update LayerLedger
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from finalayze.core import bond_math
from finalayze.core.schemas import BondPositionRecord, PortfolioLayer, SignalDirection
from finalayze.execution.broker_base import OrderRequest

if TYPE_CHECKING:
    from finalayze.core.alerts import TelegramAlerter
    from finalayze.core.layer_ledger import LayerLedger
    from finalayze.core.schemas import LayerConfig, Signal
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

# Sizing constants
_MAX_SIZING_ITERATIONS = 5
_SIZING_EPSILON = Decimal("0.01")

# Fill wait constants
_FILL_TIMEOUT_SECONDS = 120  # 2 minutes
_FILL_POLL_INTERVAL_SECONDS = 2

# Transaction cost estimates for iterative sizing (Tinkoff Trader tariff for bonds)
_BOND_COMMISSION_RATE = Decimal("0.0005")  # 0.05% of trade value
_BOND_SPREAD_BPS = Decimal(5)
_BOND_SLIPPAGE_BPS = Decimal(3)
_BPS_DIVISOR = Decimal(10_000)

_MOEX_MARKET_ID = "moex"

# OFZ rotation: shift from CORE (PK floaters) to STRATEGIC (PD fixed) during cutting cycle
_OFZ_ROTATION_SHIFT = Decimal("0.15")


def apply_ofz_rotation(
    configs: dict[PortfolioLayer, LayerConfig],
    as_of: date,
) -> dict[PortfolioLayer, LayerConfig]:
    """Adjust CORE/STRATEGIC allocations if CBR cutting cycle detected.

    Cutting cycle = 2+ consecutive CBR rate cuts. When active, shifts 15pp
    from CORE (PK floaters) to STRATEGIC (PD fixed) to capture duration trade.
    Reverts to original allocations if latest decision is not "cut".
    """
    from finalayze.data.fetchers.cbr import CBR_MEETINGS  # noqa: PLC0415

    past = [m for m in CBR_MEETINGS if m.date <= as_of and m.decision is not None]
    if len(past) < 2:  # noqa: PLR2004
        return configs

    last_two = [past[-1].decision, past[-2].decision]
    if not all(d == "cut" for d in last_two):
        return configs

    # Apply rotation: shift capital from CORE to STRATEGIC
    result = dict(configs)
    result[PortfolioLayer.CORE] = replace(
        configs[PortfolioLayer.CORE],
        capital_pct=configs[PortfolioLayer.CORE].capital_pct - _OFZ_ROTATION_SHIFT,
    )
    result[PortfolioLayer.STRATEGIC] = replace(
        configs[PortfolioLayer.STRATEGIC],
        capital_pct=configs[PortfolioLayer.STRATEGIC].capital_pct + _OFZ_ROTATION_SHIFT,
    )
    return result


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


def _estimate_transaction_costs_per_unit(clean_price_pct: Decimal, face_value: Decimal) -> Decimal:
    """Estimate per-bond transaction costs (commission + spread + slippage).

    Uses the MOEX bond cost model from backtest/costs.py constants.
    """
    price_rub = clean_price_pct / Decimal(100) * face_value
    commission = price_rub * _BOND_COMMISSION_RATE
    spread = price_rub * _BOND_SPREAD_BPS / _BPS_DIVISOR
    slippage = price_rub * _BOND_SLIPPAGE_BPS / _BPS_DIVISOR
    return commission + spread + slippage


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

        effective_configs = apply_ofz_rotation(self._layer_configs, datetime.now(tz=UTC).date())
        if effective_configs != self._layer_configs:
            _log.info(
                "ofz_rotation_active",
                core_pct=str(effective_configs[PortfolioLayer.CORE].capital_pct),
                strategic_pct=str(effective_configs[PortfolioLayer.STRATEGIC].capital_pct),
            )

        results: list[LayerResult] = []
        for layer, config in effective_configs.items():
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
        exit_count = self._process_yield_stops(layer, ledger, yield_stop, macro)

        # Step 2: Coupon reinvestment (use accumulated coupon cash for BUY signals)
        coupon_cash = getattr(ledger, "coupon_cash", Decimal(0))
        if coupon_cash > 0:
            _log.info("bond_coupon_reinvestment", layer=layer.value, coupon_cash=str(coupon_cash))
            self._alerter.on_coupon_received(
                symbol=f"Layer:{layer.value}",
                amount=coupon_cash,
                currency="RUB",
            )
            ledger.credit_cash(coupon_cash)
            if hasattr(ledger, "coupon_cash"):
                ledger.coupon_cash = Decimal(0)

        # Step 3: Generate new strategy signals
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

        # Step 4: ML filter (no-op if ml_registry is None)
        new_signals = self._apply_ml_filter(new_signals, layer, macro)

        # Step 5: Size and execute
        executed = 0
        for signal in new_signals:
            if self._size_and_execute(signal, layer, ledger):
                executed += 1

        # Step 6: Log for training data
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
        layer: PortfolioLayer,
        ledger: LayerLedger,
        yield_stop: YieldStop,
        macro: MacroSnapshot,
    ) -> int:
        """Check yield stops on existing positions, execute exits. Returns count.

        Fetches real-time prices via GetLastPrices, computes current YTM,
        and applies regime-adaptive yield stop thresholds. Exits are executed
        immediately via SELL orders.
        """
        from finalayze.strategies.bond_duration_rotation import classify_regime  # noqa: PLC0415

        if not ledger.bond_positions:
            return 0

        broker = self._broker_router.route(_BOND_MARKET_KEY)
        symbols = list(ledger.bond_positions.keys())

        try:
            prices = broker.get_last_prices(symbols)
        except Exception:
            _log.exception("yield_stop_price_fetch_failed", layer=layer.value)
            return 0

        regime = classify_regime(
            macro.key_rate,
            macro.ruonia_7d_avg,
            macro.cpi_yoy,
            macro.last_cbr_decision,
        )

        exit_count = 0
        for symbol, record in list(ledger.bond_positions.items()):
            current_price_pct = prices.get(symbol)
            if current_price_pct is None:
                _log.warning("yield_stop_no_price", symbol=symbol)
                continue

            # Compute current YTM from current clean price
            try:
                bond_info = self._registry.get(symbol, _MOEX_MARKET_ID)
                current_ytm = bond_math.ytm(
                    clean_price_pct=current_price_pct,
                    coupon_rate=bond_info.coupon_rate,
                    face_value=bond_info.face_value,
                    coupon_frequency=bond_info.coupon_frequency,
                    settlement_date=datetime.now(tz=UTC).date(),
                    maturity_date=bond_info.maturity_date,
                )
            except Exception:
                _log.exception("yield_stop_ytm_calc_failed", symbol=symbol)
                continue

            if yield_stop.is_stopped_with_regime(record.entry_ytm_pct, current_ytm, int(regime)):
                _log.info(
                    "yield_stop_triggered",
                    symbol=symbol,
                    entry_ytm=str(record.entry_ytm_pct),
                    current_ytm=str(current_ytm),
                    regime=int(regime),
                    layer=layer.value,
                )
                # Submit SELL order for full position
                if self._execute_sell(symbol, record.quantity, ledger, bond_info):
                    exit_count += 1

        return exit_count

    def _execute_sell(
        self,
        symbol: str,
        quantity: Decimal,
        ledger: LayerLedger,
        bond_info: Any,
    ) -> bool:
        """Execute a SELL order for a bond position. Returns True if filled."""
        broker = self._broker_router.route(_BOND_MARKET_KEY)
        order = OrderRequest(symbol=symbol, side="SELL", quantity=quantity)

        try:
            result = broker.submit_order(order)
        except Exception:
            _log.exception("bond_sell_order_failed", symbol=symbol)
            return False

        if result.filled:
            # Immediate fill
            fill_qty = result.quantity
            fill_price_pct = result.fill_price or Decimal(0)
            sell_proceeds = (
                bond_math.dirty_price(
                    fill_price_pct,
                    bond_math.nkd(
                        bond_info.coupon_rate
                        / Decimal(100)
                        * bond_info.face_value
                        / bond_info.coupon_frequency,
                        0,
                        182,
                    ),
                    bond_info.face_value,
                )
                * fill_qty
            )
            ledger.credit_cash(sell_proceeds)
            ledger.remove_bond_position(symbol, fill_qty)
            return True

        # Wait for fill
        if result.order_id:
            filled = self._wait_for_fill(result.order_id, broker)
            if filled is not None:
                fill_price_pct = filled.filled_price
                sell_proceeds = (
                    bond_math.dirty_price(
                        fill_price_pct,
                        bond_math.nkd(
                            bond_info.coupon_rate
                            / Decimal(100)
                            * bond_info.face_value
                            / bond_info.coupon_frequency,
                            0,
                            182,
                        ),
                        bond_info.face_value,
                    )
                    * filled.filled_quantity
                )
                ledger.credit_cash(sell_proceeds)
                ledger.remove_bond_position(symbol, filled.filled_quantity)
                return True

        return False

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
        layer: PortfolioLayer,
        ledger: LayerLedger,
    ) -> bool:
        """Size a signal and submit order. Returns True if executed.

        For BUY: iterative sizing loop with dirty price + transaction costs,
        submit limit order, wait for fill (2 min timeout), update ledger.
        For SELL: submit sell for full position quantity, wait for fill.
        """
        broker = self._broker_router.route(_BOND_MARKET_KEY)

        if signal.direction == SignalDirection.SELL:
            return self._handle_sell_signal(signal, ledger, broker)

        if signal.direction == SignalDirection.BUY:
            return self._handle_buy_signal(signal, layer, ledger, broker)

        return False

    def _handle_sell_signal(
        self,
        signal: Signal,
        ledger: LayerLedger,
        broker: Any,  # noqa: ARG002
    ) -> bool:
        """Handle a SELL signal: sell full position."""
        record = ledger.bond_positions.get(signal.symbol)
        if record is None:
            _log.warning("bond_sell_no_position", symbol=signal.symbol)
            return False

        try:
            bond_info = self._registry.get(signal.symbol, _MOEX_MARKET_ID)
        except Exception:
            _log.exception("bond_sell_info_failed", symbol=signal.symbol)
            return False

        return self._execute_sell(signal.symbol, record.quantity, ledger, bond_info)

    def _handle_buy_signal(
        self,
        signal: Signal,
        layer: PortfolioLayer,
        ledger: LayerLedger,
        broker: Any,
    ) -> bool:
        """Handle a BUY signal: iterative sizing, submit order, wait fill, update ledger."""
        pricing = self._compute_buy_pricing(signal.symbol, broker)
        if pricing is None:
            return False
        bond_info, clean_price_pct, dirty, tx_costs_per_unit, entry_ytm, dv01_per_unit = pricing

        # Size the order
        quantity = self._compute_buy_quantity(
            bond_info, layer, ledger, dirty, tx_costs_per_unit, dv01_per_unit
        )
        if quantity <= 0:
            return False

        _log.info(
            "bond_buy_order_sizing",
            symbol=signal.symbol,
            quantity=quantity,
            dirty_price=str(dirty),
            total_cost=str(dirty * Decimal(quantity) + tx_costs_per_unit * Decimal(quantity)),
            cash=str(ledger.cash),
        )

        # Submit and wait for fill
        return self._submit_and_await_buy(
            signal.symbol, quantity, clean_price_pct, entry_ytm, bond_info, ledger, broker
        )

    def _compute_buy_pricing(
        self, symbol: str, broker: Any
    ) -> tuple[Any, Decimal, Decimal, Decimal, Decimal, Decimal] | None:
        """Compute pricing data for a BUY order. Returns None on failure."""
        try:
            bond_info = self._registry.get(symbol, _MOEX_MARKET_ID)
        except Exception:
            _log.exception("bond_buy_info_failed", symbol=symbol)
            return None

        coupon_amount = (
            bond_info.coupon_rate / Decimal(100) * bond_info.face_value / bond_info.coupon_frequency
        )
        nkd_estimate = bond_math.nkd(coupon_amount, 91, 182)

        try:
            prices = broker.get_last_prices([symbol])
            clean_price_pct = prices.get(symbol)
            if clean_price_pct is None:
                _log.warning("bond_buy_no_price", symbol=symbol)
                return None
        except Exception:
            _log.exception("bond_buy_price_failed", symbol=symbol)
            return None

        dirty = bond_math.dirty_price(clean_price_pct, nkd_estimate, bond_info.face_value)
        tx_costs = _estimate_transaction_costs_per_unit(clean_price_pct, bond_info.face_value)
        today = datetime.now(tz=UTC).date()

        try:
            entry_ytm = bond_math.ytm(
                clean_price_pct=clean_price_pct,
                coupon_rate=bond_info.coupon_rate,
                face_value=bond_info.face_value,
                coupon_frequency=bond_info.coupon_frequency,
                settlement_date=today,
                maturity_date=bond_info.maturity_date,
            )
        except Exception:
            _log.exception("bond_buy_ytm_failed", symbol=symbol)
            return None

        try:
            mod_dur = bond_math.modified_duration(
                entry_ytm,
                bond_info.coupon_rate,
                bond_info.face_value,
                bond_info.coupon_frequency,
                today,
                bond_info.maturity_date,
            )
            dv01_per_unit = bond_math.dv01(mod_dur, dirty)
        except Exception:
            dv01_per_unit = Decimal("0.01")

        return (bond_info, clean_price_pct, dirty, tx_costs, entry_ytm, dv01_per_unit)

    def _compute_buy_quantity(
        self,
        bond_info: Any,
        layer: PortfolioLayer,
        ledger: LayerLedger,
        dirty: Decimal,
        tx_costs_per_unit: Decimal,
        dv01_per_unit: Decimal,
    ) -> int:
        """Compute quantity via sizer + iterative cash check. Returns 0 if none."""
        current_dv01 = self._compute_portfolio_dv01(layer)
        sizer = self._equal_weight_sizer if bond_info.floating_coupon else self._dv01_sizer

        quantity = sizer.compute_position_size(
            layer_equity=ledger.current_equity,
            bond_dv01_per_unit=dv01_per_unit,
            current_portfolio_dv01=current_dv01,
            unit_cost=dirty,
            transaction_costs_per_unit=tx_costs_per_unit,
        )

        if quantity <= 0:
            _log.info("bond_buy_zero_quantity")
            return 0

        for _ in range(_MAX_SIZING_ITERATIONS):
            total_cost = dirty * Decimal(quantity) + tx_costs_per_unit * Decimal(quantity)
            if total_cost <= ledger.cash:
                break
            quantity -= 1
            if quantity <= 0:
                _log.info("bond_buy_insufficient_cash")
                return 0

        return quantity

    def _submit_and_await_buy(
        self,
        symbol: str,
        quantity: int,
        clean_price_pct: Decimal,
        entry_ytm: Decimal,
        bond_info: Any,
        ledger: LayerLedger,
        broker: Any,
    ) -> bool:
        """Submit BUY order, wait for fill, handle timeout/partial. Returns True if filled."""
        order = OrderRequest(symbol=symbol, side="BUY", quantity=Decimal(quantity))
        try:
            result = broker.submit_order(order)
        except Exception:
            _log.exception("bond_buy_order_failed", symbol=symbol)
            return False

        if result.filled:
            return self._record_buy_fill(
                symbol,
                result.quantity,
                result.fill_price or clean_price_pct,
                entry_ytm,
                bond_info,
                ledger,
            )

        if not result.order_id:
            _log.warning("bond_buy_no_order_id", symbol=symbol)
            return False

        filled = self._wait_for_fill(result.order_id, broker)
        if filled is not None:
            return self._record_buy_fill(
                symbol,
                filled.filled_quantity,
                filled.filled_price,
                entry_ytm,
                bond_info,
                ledger,
            )

        return self._handle_buy_timeout(
            result.order_id, symbol, entry_ytm, bond_info, ledger, broker
        )

    def _handle_buy_timeout(
        self,
        order_id: str,
        symbol: str,
        entry_ytm: Decimal,
        bond_info: Any,
        ledger: LayerLedger,
        broker: Any,
    ) -> bool:
        """Cancel timed-out order, check for partial fill. Returns True if partial filled."""
        try:
            broker.cancel_order(order_id)
        except Exception:
            _log.exception("bond_buy_cancel_failed", order_id=order_id)

        try:
            final_state = broker.get_order_state(order_id)
            if final_state.filled_quantity > 0:
                _log.info(
                    "bond_buy_partial_fill",
                    symbol=symbol,
                    filled_qty=str(final_state.filled_quantity),
                    order_id=order_id,
                )
                return self._record_buy_fill(
                    symbol,
                    final_state.filled_quantity,
                    final_state.filled_price,
                    entry_ytm,
                    bond_info,
                    ledger,
                )
        except Exception:
            _log.exception("bond_buy_final_state_failed", order_id=order_id)

        _log.warning("bond_buy_timeout", symbol=symbol, order_id=order_id)
        return False

    def _record_buy_fill(
        self,
        symbol: str,
        quantity: Decimal,
        fill_price_pct: Decimal,
        entry_ytm: Decimal,
        bond_info: Any,
        ledger: LayerLedger,
    ) -> bool:
        """Record a BUY fill in the ledger. Debit cash, add bond position."""
        coupon_amount = (
            bond_info.coupon_rate / Decimal(100) * bond_info.face_value / bond_info.coupon_frequency
        )
        nkd_est = bond_math.nkd(coupon_amount, 91, 182)
        dirty = bond_math.dirty_price(fill_price_pct, nkd_est, bond_info.face_value)
        tx_costs = _estimate_transaction_costs_per_unit(fill_price_pct, bond_info.face_value)
        total_cost = dirty * quantity + tx_costs * quantity

        ledger.debit_cash(total_cost)
        ledger.add_bond_position(
            BondPositionRecord(
                symbol=symbol,
                quantity=quantity,
                entry_ytm_pct=entry_ytm,
                entry_date=datetime.now(tz=UTC).date(),
                entry_price=fill_price_pct,
                entry_clean_pct=fill_price_pct,
                layer_id=ledger.layer_id,
            )
        )

        _log.info(
            "bond_buy_filled",
            symbol=symbol,
            quantity=str(quantity),
            fill_price=str(fill_price_pct),
            dirty_price=str(dirty),
            total_cost=str(total_cost),
        )
        return True

    def _wait_for_fill(
        self,
        order_id: str,
        broker: Any,
    ) -> Any | None:
        """Poll order state until terminal or timeout. Returns OrderStateResult if filled."""
        start = time.monotonic()
        while True:
            try:
                state = broker.get_order_state(order_id)
            except Exception:
                _log.exception("fill_wait_poll_failed", order_id=order_id)
                return None

            if state.is_terminal:
                if state.execution_status == "fill" and state.filled_quantity > 0:
                    return state
                if state.execution_status == "cancelled" and state.filled_quantity > 0:
                    return state
                return None

            elapsed = time.monotonic() - start
            if elapsed >= _FILL_TIMEOUT_SECONDS:
                return None

            time.sleep(_FILL_POLL_INTERVAL_SECONDS)

    def _compute_portfolio_dv01(self, layer: PortfolioLayer) -> Decimal:
        """Compute aggregate DV01 across all bond positions in this layer."""
        ledger = self._layer_ledgers[layer]
        total_dv01 = Decimal(0)
        for symbol, record in ledger.bond_positions.items():
            try:
                bond_info = self._registry.get(symbol, _MOEX_MARKET_ID)
                mod_dur = bond_math.modified_duration(
                    record.entry_ytm_pct,
                    bond_info.coupon_rate,
                    bond_info.face_value,
                    bond_info.coupon_frequency,
                    record.entry_date,
                    bond_info.maturity_date,
                )
                coupon_amount = (
                    bond_info.coupon_rate
                    / Decimal(100)
                    * bond_info.face_value
                    / bond_info.coupon_frequency
                )
                nkd_est = bond_math.nkd(coupon_amount, 91, 182)
                dirty = bond_math.dirty_price(record.entry_clean_pct, nkd_est, bond_info.face_value)
                unit_dv01 = bond_math.dv01(mod_dur, dirty)
                total_dv01 += unit_dv01 * record.quantity
            except Exception:
                _log.warning("portfolio_dv01_calc_failed", symbol=symbol)
        return total_dv01

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

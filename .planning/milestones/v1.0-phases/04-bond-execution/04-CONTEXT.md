# Phase 4: Bond Execution - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Complete BondCycleProcessor stubs (_size_and_execute, _process_yield_stops), wire YieldStop to live positions with regime-adaptive thresholds, register a separate "moex_bonds" TinkoffBroker in BrokerRouter, fix DV01BudgetStep to use dirty price for cash checks, implement LayerLedger persistence and startup reconciliation, and prove positive bond backtest PnL with walk-forward validation on OFZ instruments.

</domain>

<decisions>
## Implementation Decisions

### Order Submission Flow
- DV01BudgetStep must use dirty price (clean + NKD) AND transaction costs for cash sufficiency checks
- Iterative sizing to resolve the quantity/cost circular dependency: compute quantity, then costs, reduce quantity if needed, repeat until stable
- Order type: Claude's discretion (market vs limit) based on OFZ liquidity and existing TinkoffBroker patterns
- Wait for fill confirmation from T-Invest before updating LayerLedger (not optimistic)
- Fill timeout: 2 minutes, then cancel unfilled portion
- Partial fills: cancel remainder, keep partial fill, update ledger with filled quantity
- No retry on timeout — next cycle can try again
- Explicit coupon reinvestment step in _process_layer: use accumulated coupon cash specifically to buy bonds (not just accumulate as generic cash)

### Yield Stop Exit Logic
- Use regime-adaptive stops (is_stopped_with_regime) with current CBR regime from MacroSnapshot — consistent with BondBacktestEngine
- Price source for current YTM: real-time quote via T-Invest GetLastPrices() (not cached candles)
- Immediate exit: submit SELL as soon as yield stop triggers, before processing new BUY signals (matches BondCycleProcessor processing order)
- Wait for fill confirmation on exits too (consistent with BUY orders): 2 min timeout, cancel remainder, keep partial
- No cooldown after yield stop exit — allow immediate re-entry next cycle if signal is still BUY
- Entry YTM stored in LayerLedger via BondPositionRecord dataclass (symbol, quantity, entry_ytm_pct, entry_date, entry_price)
- New BondPositionRecord dataclass replaces the plain dict[str, Decimal] for bond layer positions

### Bond Broker Separation
- Shared gRPC channel between equity and bond TinkoffBroker instances (same AsyncClient, separate instances)
- Same TinkoffBroker class for bonds (no subclass) — T-Invest handles bonds and equities through same OrderService
- Same T-Invest account_id for both equities and bonds — capital allocation is virtual via LayerLedger
- BrokerRouter gets "moex_bonds" key pointing to the bond TinkoffBroker instance
- Startup reconciliation: query T-Invest GetPortfolio, diff against persisted ledger, fix discrepancies
- Unknown positions (found in broker but not in ledger): add to Core layer and send Telegram alert
- LayerLedger state persisted to TimescaleDB (consistent with MacroSnapshot persistence from Phase 3)

### Backtest PnL Target
- Run both strategies: BondCarryStrategy on OFZ-PK (Core/Strategic layers), BondDurationRotationStrategy on OFZ-PD (Tactical layer)
- Bond universe: all OFZs discoverable through Phase 3 pipeline (not hand-picked subset)
- Total return PnL: include coupon income net of 13% NDFL tax (not price PnL only)
- Minimum Sharpe > 0 (walk-forward out-of-sample)
- Minimum Profit Factor > 1.0
- Maximum drawdown <= 3% (matches AggregateBondBreaker threshold)
- Walk-forward split: Claude's discretion based on bond market regime frequency and data availability

### Claude's Discretion
- Order type selection (market vs limit) based on liquidity patterns
- Walk-forward train/test split period lengths
- Bond-specific logging verbosity and error messages
- Exact iterative sizing convergence criteria (max iterations, epsilon)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BondCycleProcessor` (`core/bond_cycle.py`): Full skeleton with stub methods — fill in `_size_and_execute()` and `_process_yield_stops()`
- `YieldStop` (`risk/yield_stop.py`): Fully implemented with `is_stopped_with_regime()` — ready to use
- `DV01BudgetStep` / `EqualWeightBondSizer` (`risk/dv01_sizing.py`): Need dirty price fix in `compute_position_size()`
- `BondBacktestEngine` (`backtest/bond_engine.py`): Complete engine with coupon tracking, NDFL, regime stops
- `BondCarryStrategy` (`strategies/bond_carry.py`): Maturity ladder with CBR regime gating
- `BondDurationRotationStrategy` (`strategies/bond_duration_rotation.py`): Tactical rotation based on CBR regime
- `BondSimulatedBroker` (`execution/bond_simulated_broker.py`): For backtesting
- `LayerLedger` (`core/layer_ledger.py`): Needs BondPositionRecord extension and DB persistence
- `BondLayerBreaker` / `AggregateBondBreaker` (`risk/layer_circuit_breaker.py`): Ready to use
- `BrokerRouter` (`execution/broker_router.py`): Add "moex_bonds" registration
- `TinkoffBroker` (`execution/tinkoff_broker.py`): Reuse for bond orders
- `run_bond_iteration.py`: Backtest script already wired

### Established Patterns
- BrokerRouter routes by market_id string — extend with "moex_bonds"
- TinkoffBroker uses RetryPolicy with backoff — reuse for bond orders
- MacroSnapshot persisted via async SQLAlchemy ORM — reuse pattern for LayerLedger
- Bond math functions (ytm, dirty_price, dv01, modified_duration) in bond_math.py — Layer 0, no I/O

### Integration Points
- `BondCycleProcessor.__init__()` already accepts all dependencies via DI
- `TradingLoop` in `core/trading_loop.py` has bond cycle placeholder
- `config/settings.py` needs bond broker configuration
- `core/schemas.py` may need BondPositionRecord schema
- Event bus for coupon event emission (wired in Phase 3)

</code_context>

<specifics>
## Specific Ideas

- T-Invest GetPortfolio returns both equity and bond positions — filter by instrument_type for reconciliation
- Bond lot sizes on MOEX are typically 1 bond (face value 1000 RUB) — simpler than equity lot sizing
- OFZ-PK carry should be naturally profitable (RUONIA spread ~1.3-1.6%) — the backtest bar is low
- 2022-2023 was a CBR hiking cycle (rate went from 9.5% to 16%) — challenging for OFZ-PD duration rotation
- AggregateBondBreaker 3% threshold is calibrated below sum of weighted per-layer maximums (3.8%)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-bond-execution*
*Context gathered: 2026-03-14*

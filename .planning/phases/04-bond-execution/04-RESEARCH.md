# Phase 4: Bond Execution - Research

**Researched:** 2026-03-14
**Domain:** Bond order execution, yield stop wiring, broker separation, DV01 dirty-price fix, backtest validation
**Confidence:** HIGH

## Summary

Phase 4 fills in the stub methods in `BondCycleProcessor` (`_size_and_execute`, `_process_yield_stops`), wires a separate `moex_bonds` TinkoffBroker into `BrokerRouter`, fixes `DV01BudgetStep` to use dirty price instead of face value for cash sufficiency, adds `BondPositionRecord` dataclass and `LayerLedger` DB persistence with startup reconciliation, and proves positive bond backtest PnL with walk-forward validation.

All building blocks exist from Phases 2-3: `YieldStop.is_stopped_with_regime()` is fully implemented, `TinkoffBroker` has `submit_order()`, `cancel_order()`, and `get_portfolio()`, the `BondBacktestEngine` handles coupon tracking with NDFL, and `run_bond_iteration.py` already runs end-to-end. The work is primarily integration and wiring -- connecting existing components into the live cycle processor, fixing the DV01 cash calculation, and adding persistence.

**Primary recommendation:** Implement in 3 plans: (1) DV01 dirty-price fix + BondPositionRecord + LayerLedger persistence, (2) _size_and_execute + _process_yield_stops + moex_bonds broker registration, (3) bond backtest walk-forward validation proving positive PnL.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- DV01BudgetStep must use dirty price (clean + NKD) AND transaction costs for cash sufficiency checks
- Iterative sizing to resolve the quantity/cost circular dependency: compute quantity, then costs, reduce quantity if needed, repeat until stable
- Wait for fill confirmation from T-Invest before updating LayerLedger (not optimistic)
- Fill timeout: 2 minutes, then cancel unfilled portion
- Partial fills: cancel remainder, keep partial fill, update ledger with filled quantity
- No retry on timeout -- next cycle can try again
- Explicit coupon reinvestment step in _process_layer: use accumulated coupon cash specifically to buy bonds
- Use regime-adaptive stops (is_stopped_with_regime) with current CBR regime from MacroSnapshot
- Price source for current YTM: real-time quote via T-Invest GetLastPrices() (not cached candles)
- Immediate exit: submit SELL as soon as yield stop triggers, before processing new BUY signals
- Wait for fill confirmation on exits too: 2 min timeout, cancel remainder, keep partial
- No cooldown after yield stop exit -- allow immediate re-entry next cycle if signal is still BUY
- Entry YTM stored in LayerLedger via BondPositionRecord dataclass
- New BondPositionRecord dataclass replaces the plain dict[str, Decimal] for bond layer positions
- Shared gRPC channel between equity and bond TinkoffBroker instances (same AsyncClient, separate instances)
- Same TinkoffBroker class for bonds (no subclass) -- T-Invest handles bonds and equities through same OrderService
- Same T-Invest account_id for both equities and bonds -- capital allocation is virtual via LayerLedger
- BrokerRouter gets "moex_bonds" key pointing to the bond TinkoffBroker instance
- Startup reconciliation: query T-Invest GetPortfolio, diff against persisted ledger, fix discrepancies
- Unknown positions (found in broker but not in ledger): add to Core layer and send Telegram alert
- LayerLedger state persisted to TimescaleDB (consistent with MacroSnapshot persistence from Phase 3)
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

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BEX-01 | `BondCycleProcessor._size_and_execute()` completes real order submission | TinkoffBroker.submit_order() exists; need iterative sizing loop with dirty price, fill wait, partial fill handling |
| BEX-02 | `YieldStop._process_yield_stops()` computes current YTM and exits positions | YieldStop.is_stopped_with_regime() ready; need GetLastPrices() for real-time YTM, SELL order submission |
| BEX-03 | Separate `moex_bonds` TinkoffBroker instance in BrokerRouter | BrokerRouter accepts dict[str, BrokerBase]; register "moex_bonds" key with shared AsyncClient |
| BEX-04 | DV01BudgetStep uses dirty price (not face_value) for cash calculations | Current code uses `face_value` on line 74 of dv01_sizing.py; must change to dirty_price |
| BEX-05 | Bond backtest shows positive carry PnL with walk-forward validation | BondBacktestEngine + run_bond_iteration.py exist; need walk-forward wrapper |
| BEX-06 | LayerLedger reconciliation on startup (sync with broker state) | TinkoffBroker.get_portfolio() returns FIGI-keyed positions; need diff logic + DB persistence |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| t-tech-investments | latest | T-Invest gRPC SDK for order submission, GetLastPrices, GetPortfolio | Only way to interact with T-Invest API |
| SQLAlchemy | 2.0 | Async ORM for LayerLedger persistence | Already used for MacroSnapshotModel, BondCandleModel |
| Pydantic | v2 | BondPositionRecord schema validation | Project convention for all schemas |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| structlog | existing | Structured logging for order lifecycle | All bond execution events |
| asyncio | stdlib | Async boundary for gRPC calls within sync context | TinkoffBroker pattern: asyncio.run() in sync methods |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Market orders | Limit orders | OFZ bonds have thin order books; market orders risk slippage but guarantee fill. Limit orders risk non-fill within 2min timeout. **Recommendation: use limit orders at best-ask for BUY, best-bid for SELL** -- OFZ order book is usually tight (1-3 bps spread on liquid issues) and this avoids adverse selection from market orders on thin books |

## Architecture Patterns

### Key Integration Points
```
core/
  bond_cycle.py      # BondCycleProcessor -- fill _size_and_execute, _process_yield_stops
  layer_ledger.py    # Add BondPositionRecord, DB persistence methods
  schemas.py         # Add BondPositionRecord dataclass
  models.py          # Add LayerLedgerModel ORM class
risk/
  dv01_sizing.py     # Fix cash sufficiency to use dirty price
  yield_stop.py      # Already complete -- no changes needed
execution/
  tinkoff_broker.py  # Add get_last_prices() method, extend submit_order return with order_id
  broker_router.py   # Register "moex_bonds" (no code changes needed, just wiring)
scripts/
  run_bond_iteration.py  # Add walk-forward wrapper for backtest validation
```

### Pattern 1: Iterative Sizing Loop
**What:** Resolve circular dependency: quantity depends on cost, cost depends on quantity.
**When to use:** Every bond BUY order in _size_and_execute.
**Example:**
```python
# Iterative sizing: compute qty, then cost, reduce if needed
MAX_SIZING_ITERATIONS = 5
SIZING_EPSILON = Decimal("0.01")  # convergence check

quantity = sizer.compute_position_size(
    layer_equity=ledger.cash,
    bond_dv01_per_unit=bond_dv01,
    current_portfolio_dv01=portfolio_dv01,
    face_value=dirty_px,  # FIX: use dirty price, not face_value
)

for _ in range(MAX_SIZING_ITERATIONS):
    cost = bond_total_cost(costs, clean_pct, face_value, Decimal(quantity), symbol)
    total_outlay = dirty_px * quantity + cost
    if total_outlay <= ledger.cash:
        break
    # Reduce quantity by 1 and retry
    quantity -= 1
    if quantity <= 0:
        break
```

### Pattern 2: Fill Wait with Timeout
**What:** Submit order, poll for fill, cancel after timeout.
**When to use:** Both BUY and SELL orders.
**Example:**
```python
import asyncio
import time

FILL_TIMEOUT_SECONDS = 120  # 2 minutes

result = broker.submit_order(order)
order_id = result.order_id  # Need to extend OrderResult

start = time.monotonic()
while time.monotonic() - start < FILL_TIMEOUT_SECONDS:
    state = broker.get_order_state(order_id)
    if state.is_terminal:
        break
    time.sleep(2)  # poll every 2 seconds

if not state.is_filled:
    broker.cancel_order(order_id)
    # Keep partial fill if any
    filled_qty = state.filled_quantity
```

### Pattern 3: LayerLedger DB Persistence (following MacroSnapshotModel pattern)
**What:** Persist ledger state to TimescaleDB using async SQLAlchemy.
**When to use:** After every successful order fill, on startup for reconciliation.
**Example:**
```python
class LayerLedgerModel(Base):
    __tablename__ = "layer_ledger"

    layer_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(30), primary_key=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    entry_ytm_pct: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    entry_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
```

### Anti-Patterns to Avoid
- **Optimistic ledger update:** Never update LayerLedger before confirming fill -- user decision explicitly requires fill confirmation first.
- **Shared broker instance for equity and bonds:** User decided separate TinkoffBroker instances with same AsyncClient (shared gRPC channel). Do NOT reuse the same TinkoffBroker object.
- **Face value in cash check:** The current DV01BudgetStep uses `face_value` in `max_by_position` calculation (line 74). This underestimates actual cost because dirty price > face_value when NKD is positive. Must use dirty price.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YTM calculation | Custom YTM solver | `core/bond_math.ytm()` | Newton-Raphson already implemented and tested |
| Dirty price math | Inline calculation | `core/bond_math.dirty_price()` | Handles clean_pct to RUB conversion correctly |
| DV01 computation | Manual approximation | `core/bond_math.dv01()` | Uses modified duration * dirty price * 0.0001 |
| Coupon income tracking | Manual counter | `BondBacktestEngine` pattern | Already handles NDFL tax, gross/net separation |
| Order retries | Custom retry loop | `RetryPolicy` from execution/retry.py | Already wired into TinkoffBroker |
| Regime classification | Hardcoded thresholds | `classify_regime()` from bond_duration_rotation.py | Used by both backtest engine and yield stops |

## Common Pitfalls

### Pitfall 1: Bond lot size is 1, but quantity is in lots
**What goes wrong:** TinkoffBroker rounds quantity to lot multiples. Bond lot_size on MOEX is typically 1 (1 bond = 1 lot), but some bonds may have different lot sizes.
**Why it happens:** The existing TinkoffBroker.submit_order() divides by lot_size. If lot_size=1, this is a no-op, but if someone registers a bond with lot_size != 1, the rounding could unexpectedly reduce quantity.
**How to avoid:** Verify lot_size for OFZ bonds in InstrumentRegistry. OFZ lot_size should be 1.
**Warning signs:** Quantity 0 returned from submit_order for small positions.

### Pitfall 2: GetLastPrices returns clean price as % of face
**What goes wrong:** Using the raw price from GetLastPrices as RUB amount instead of % of face value.
**Why it happens:** MOEX bond prices are quoted as percentage of face value (e.g., 85.50 means 855 RUB for a 1000 RUB bond). The same convention is used throughout bond_math.py functions.
**How to avoid:** Always pass GetLastPrices result through bond_math functions. The existing pipeline (BondBacktestEngine) already handles this correctly -- match its convention.
**Warning signs:** YTM calculations returning absurd values (>100% or negative).

### Pitfall 3: NKD (accrued interest) changes daily
**What goes wrong:** Using stale NKD from candle cache instead of computing current NKD for real-time orders.
**Why it happens:** NKD accrues linearly between coupon dates. A cached candle's NKD is for that candle's date, not today.
**How to avoid:** For live execution, either fetch NKD from T-Invest GetBondCoupons (which includes accumulated_coupon_value) or compute via bond_math.nkd() using current date and coupon schedule.
**Warning signs:** Cash sufficiency checks passing but actual order cost exceeding available cash.

### Pitfall 4: post_order returns immediately, fill is async
**What goes wrong:** Assuming post_order return means the order is filled. T-Invest post_order returns an execution report that may show ORDER_FILL_STATUS_PROGRESS.
**Why it happens:** The current TinkoffBroker.submit_order() reads `executed_order_price` from the response, assuming immediate fill. For OFZ bonds, market orders may not fill immediately.
**How to avoid:** After post_order, use get_order_state (or get_orders) to poll for terminal status. The user explicitly requires this: "Wait for fill confirmation."
**Warning signs:** executed_order_price of 0 in the response, or quantity mismatch.

### Pitfall 5: asyncio.run() inside async context
**What goes wrong:** TinkoffBroker uses `asyncio.run()` for sync-to-async bridging. If BondCycleProcessor is ever called from an async context (e.g., inside an event loop), asyncio.run() will raise RuntimeError.
**Why it happens:** The existing TinkoffBroker pattern uses asyncio.run() throughout. BondCycleProcessor.run_cycle() is sync.
**How to avoid:** BondCycleProcessor.run_cycle() is designed as sync (matching the equity cycle pattern). Keep it sync. The fill-wait polling loop should use time.sleep(), not asyncio.sleep().
**Warning signs:** "This event loop is already running" RuntimeError.

### Pitfall 6: Reconciliation must handle FIGI-to-symbol mapping
**What goes wrong:** TinkoffBroker.get_portfolio() returns positions keyed by FIGI, but LayerLedger uses symbol (ticker) keys.
**Why it happens:** T-Invest API uses FIGI as primary identifier; our system uses symbol/ticker.
**How to avoid:** Use InstrumentRegistry to map between FIGI and symbol during reconciliation. The registry already has this mapping from Phase 3 bond discovery.
**Warning signs:** Reconciliation finding "unknown" positions that are actually known bonds under different keys.

## Code Examples

### Current DV01BudgetStep Bug (line 74)
```python
# CURRENT (buggy):
max_by_position = int(layer_equity * self._max_single_position_pct / face_value)

# FIXED:
# face_value parameter should be dirty_price for cash sufficiency
# OR add a new parameter for actual unit cost
max_by_position = int(layer_equity * self._max_single_position_pct / unit_cost)
# where unit_cost = dirty_price(clean_pct, nkd, face_value)
```

### TinkoffBroker GetLastPrices (new method needed)
```python
async def _get_last_prices_async(self, figis: list[str]) -> object:
    """Fetch last prices for given FIGIs."""
    client = self._get_client()
    return await client.market_data.get_last_prices(
        figi=figis,
    )

def get_last_prices(self, symbols: list[str]) -> dict[str, Decimal]:
    """Get last prices for symbols. Returns symbol -> price (% of face for bonds)."""
    figis = [self._registry.get(s, "moex").figi for s in symbols]
    response = self._call(lambda: asyncio.run(self._get_last_prices_async(figis)))
    result = {}
    for lp in response.last_prices:
        price = self._quotation_to_decimal(lp.price)
        # Map FIGI back to symbol
        for sym in symbols:
            if self._registry.get(sym, "moex").figi == lp.figi:
                result[sym] = price
                break
    return result
```

### BondPositionRecord (new dataclass)
```python
@dataclass
class BondPositionRecord:
    """Track a bond position with entry metadata for yield stop evaluation."""
    symbol: str
    quantity: Decimal
    entry_ytm_pct: Decimal
    entry_date: date
    entry_price: Decimal  # dirty price at entry (RUB)
    entry_clean_pct: Decimal  # clean price as % of face at entry
    layer_id: str
```

### Startup Reconciliation Pattern
```python
def reconcile_on_startup(
    broker: TinkoffBroker,
    ledgers: dict[PortfolioLayer, LayerLedger],
    registry: InstrumentRegistry,
    alerter: TelegramAlerter,
) -> None:
    """Diff broker portfolio against persisted ledgers, fix discrepancies."""
    portfolio = broker.get_portfolio()
    broker_positions = portfolio.positions  # FIGI -> qty

    # Build ledger aggregate: FIGI -> (layer, qty)
    ledger_positions: dict[str, tuple[PortfolioLayer, Decimal]] = {}
    for layer, ledger in ledgers.items():
        for symbol, qty in ledger.positions.items():
            figi = registry.get(symbol, "moex").figi
            ledger_positions[figi] = (layer, qty)

    # Find discrepancies
    for figi, broker_qty in broker_positions.items():
        if figi not in ledger_positions:
            # Unknown position -- add to Core layer
            symbol = registry.figi_to_symbol(figi, "moex")
            ledgers[PortfolioLayer.CORE].add_position(symbol, broker_qty)
            alerter.send_alert(f"Unknown bond position found: {symbol} qty={broker_qty}")
        elif ledger_positions[figi][1] != broker_qty:
            # Quantity mismatch -- trust broker
            layer, _ = ledger_positions[figi]
            symbol = registry.figi_to_symbol(figi, "moex")
            ledgers[layer].positions[symbol] = broker_qty
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Face value for cash check | Dirty price for cash check | This phase | Prevents overspending on positions when NKD is high (up to 3% of face near coupon date) |
| dict[str, Decimal] positions | BondPositionRecord with entry metadata | This phase | Enables yield stop evaluation (needs entry_ytm_pct) |
| In-memory-only ledger | DB-persisted LayerLedger | This phase | Survives restarts, enables reconciliation |
| Single "moex" broker key | Separate "moex" + "moex_bonds" keys | This phase | Prevents bond orders from interfering with equity order state |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (latest via uv) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/test_bond_cycle.py tests/unit/test_dv01_sizing.py tests/unit/test_yield_stop.py tests/unit/test_layer_ledger.py -x` |
| Full suite command | `uv run pytest tests/ -x --timeout=300` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BEX-01 | _size_and_execute submits real order via TinkoffBroker | unit (mocked broker) | `uv run pytest tests/unit/test_bond_cycle.py -k "size_and_execute" -x` | Partially (test_bond_cycle.py exists, needs new tests) |
| BEX-02 | _process_yield_stops computes YTM and exits | unit (mocked prices) | `uv run pytest tests/unit/test_bond_cycle.py -k "yield_stop" -x` | Partially (test exists, needs yield stop wiring tests) |
| BEX-03 | moex_bonds broker registered in BrokerRouter | unit | `uv run pytest tests/unit/test_broker_router.py -k "moex_bonds" -x` | No, needs new test |
| BEX-04 | DV01BudgetStep uses dirty price | unit | `uv run pytest tests/unit/test_dv01_sizing.py -k "dirty" -x` | No, needs new test |
| BEX-05 | Bond backtest positive PnL walk-forward | integration | `uv run pytest tests/integration/test_bond_iteration_script.py -x` | Partially (script test exists) |
| BEX-06 | Startup reconciliation syncs ledger with broker | unit (mocked broker) | `uv run pytest tests/unit/test_layer_ledger.py -k "reconcil" -x` | No, needs new test |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_bond_cycle.py tests/unit/test_dv01_sizing.py tests/unit/test_layer_ledger.py tests/unit/test_broker_router.py -x`
- **Per wave merge:** `uv run pytest tests/ -x --timeout=300`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_bond_cycle.py` -- needs new tests for _size_and_execute with mocked broker, _process_yield_stops with mocked GetLastPrices, fill timeout, partial fills
- [ ] `tests/unit/test_dv01_sizing.py` -- needs test for dirty price parameter (currently tests use face_value)
- [ ] `tests/unit/test_layer_ledger.py` -- needs tests for BondPositionRecord, DB persistence, reconciliation
- [ ] `tests/unit/test_broker_router.py` -- needs test for "moex_bonds" key registration
- [ ] `tests/unit/test_bond_walk_forward.py` -- exists but may need walk-forward validation tests

## Open Questions

1. **Order type: market vs limit?**
   - What we know: OFZ bonds trade on MOEX with generally tight spreads (1-3 bps on liquid issues like SU26238). Market orders risk adverse selection on thin books. T-Invest supports both ORDER_TYPE_MARKET and ORDER_TYPE_LIMIT.
   - Recommendation: **Use limit orders at mid-market** (best_bid + best_ask) / 2 rounded to tick. This avoids slippage while still likely filling within the 2-minute timeout. If GetOrderBook is too expensive to call, use last price as limit. Fall back to market order if limit doesn't fill within 90 seconds.

2. **Walk-forward split periods?**
   - What we know: Bond market regimes change at CBR meetings (~8/year). 2022-2025 data has ~3 years. The CBR hiking cycle (2022-2023) is fundamentally different from the potential easing cycle (2024-2025).
   - Recommendation: **12-month train, 6-month test, rolling quarterly**. This gives 4-5 walk-forward folds on 2022-2025 data and ensures each fold sees at least 2 CBR meetings.

3. **GetLastPrices vs GetOrderBook for current price?**
   - What we know: GetLastPrices returns the last trade price. GetOrderBook returns full depth. For YTM calculation, last trade price is sufficient. For limit order placement, order book is better.
   - Recommendation: Use GetLastPrices for YTM in yield stops (cheaper, simpler). Use GetOrderBook only if implementing limit orders for order placement.

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `src/finalayze/core/bond_cycle.py` -- stub methods identified, DI pattern confirmed
- Codebase inspection: `src/finalayze/execution/tinkoff_broker.py` -- submit_order, cancel_order, get_portfolio all implemented
- Codebase inspection: `src/finalayze/risk/yield_stop.py` -- is_stopped_with_regime() fully implemented
- Codebase inspection: `src/finalayze/risk/dv01_sizing.py` -- face_value bug on line 74 confirmed
- Codebase inspection: `src/finalayze/backtest/bond_engine.py` -- complete engine with regime stops, coupon tracking
- Codebase inspection: `src/finalayze/core/models.py` -- MacroSnapshotModel persistence pattern confirmed

### Secondary (MEDIUM confidence)
- T-Invest SDK API: `client.market_data.get_last_prices()`, `client.orders.get_order_state()` -- based on SDK type stubs and training data about T-Invest gRPC API

### Tertiary (LOW confidence)
- OFZ order book spread estimates (1-3 bps) -- from general knowledge about Russian government bond markets, needs validation with live data

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, no new dependencies needed
- Architecture: HIGH -- all integration points visible in codebase, patterns established
- Pitfalls: HIGH -- identified from actual code inspection (DV01 bug, FIGI mapping, asyncio pattern)
- Backtest targets: MEDIUM -- OFZ carry should be naturally profitable but walk-forward results depend on data quality

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable domain, existing codebase)

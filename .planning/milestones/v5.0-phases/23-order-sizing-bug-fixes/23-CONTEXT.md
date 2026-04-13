# Phase 23: Order Sizing Bug Fixes - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix three critical order sizing bugs: (1) SELL orders computed via Kelly instead of actual position, (2) sector exposure uses current instrument's price for all positions, (3) CAUTION confidence threshold hardcoded to 0.6 instead of segment-specific.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure bug-fix phase.
Key constraints from audit:
- SIZE-01: SELL in _build_order() must use actual held quantity from broker portfolio, not kelly * equity / price
- SIZE-02: Sector exposure in _process_instrument() must query each position's own last price (from portfolio or cache)
- SIZE-03: CAUTION min_conf must be `preset.min_combined_confidence * _MIN_CONFIDENCE_BOOST` not `0.5 * 1.2`
- SimulatedBroker masks SIZE-01 via min(order.quantity, held) — live broker does not

</decisions>

<code_context>
## Existing Code Insights

### Key Files
- `src/finalayze/orchestration/trading_loop.py:1486-1514` — _build_order() with Kelly sizing for both BUY/SELL
- `src/finalayze/orchestration/trading_loop.py:1430-1437` — sector exposure using candles[-1].close for all positions
- `src/finalayze/orchestration/trading_loop.py:1497-1500` — hardcoded 0.5 * 1.2 = 0.6 CAUTION threshold
- `src/finalayze/execution/simulated_broker.py` — _execute_sell uses min(order.quantity, held)

### Established Patterns
- BrokerRouter.route(market_id).get_portfolio() returns portfolio with positions
- Position has symbol, quantity, avg_price fields
- Segment presets loaded via _load_preset() with min_combined_confidence field

</code_context>

<specifics>
## Specific Ideas

No specific requirements — bug-fix phase.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>

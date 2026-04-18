# Execution

## Purpose
Order routing, broker abstraction, and trade execution for US (Alpaca) and MOEX (Tinkoff Invest) markets, plus simulated brokers for backtesting.

## Layer
Layer 5 -- Execution. Can import from layers 0-4. Never import from layer 6.

## Key Files
- `broker_base.py` -- BrokerBase ABC, OrderRequest, OrderResult dataclasses
- `broker_router.py` -- BrokerRouter: dispatches orders to correct broker by market_id
- `alpaca_broker.py` -- Alpaca API broker (US market, paper + live)
- `tinkoff_broker.py` -- Tinkoff Invest gRPC broker (MOEX, sandbox + live)
- `simulated_broker.py` -- SimulatedBroker for backtesting (fills at candle open)
- `bond_simulated_broker.py` -- Bond-specific simulated broker (accrued interest, coupon income)
- `retry.py` -- RetryPolicy: exponential backoff with jitter for transient errors (ConnectionError, TimeoutError, gRPC errors)
- `impact.py` -- Market impact / slippage estimation
- `sandbox_tracker.py` -- Sandbox order tracking and P&L attribution

## Public API
- `BrokerBase` -- abstract interface: `submit_order()`, `get_portfolio()`, `has_position()`, `get_positions()`, `cancel_order()`
- `OrderRequest` -- frozen dataclass (symbol, side, quantity)
- `OrderResult` -- frozen dataclass (filled, fill_price, symbol, side, quantity, reason)
- `BrokerRouter` -- `route(market_id) -> BrokerBase`, `submit(order, market_id) -> OrderResult`
- `RetryPolicy` -- `execute(fn)` and `async_execute(fn)` with configurable retries

## Contracts
- Input: OrderRequest with symbol, side (BUY/SELL), quantity (Decimal)
- Output: OrderResult with filled flag, fill_price, reason for rejection
- Invariants: SimulatedBroker fills at candle open price (next bar). Live brokers raise BrokerError on rejection. InsufficientFundsError and InstrumentNotFoundError are NOT retried. RetryPolicy uses exponential backoff (base 1s, max 30s) with random jitter.

## Testing
- Test location: `tests/unit/test_broker.py`, `tests/unit/test_broker_router.py`, `tests/unit/test_alpaca_broker.py`
- Run: `uv run pytest tests/unit/test_broker.py tests/unit/test_broker_router.py -v`

## Common Patterns
- All brokers implement BrokerBase ABC (strategy pattern)
- BrokerRouter maps market_id -> BrokerBase instance
- Tinkoff broker must use `target="invest-public-api.tbank.ru:443"` (old domain deprecated)
- `fill_candle` parameter used only by simulated brokers; live brokers ignore it
- Retry separates fatal errors (InsufficientFundsError) from transient (ConnectionError)

---

## Graph

- **Parent:** [`src/finalayze/AGENTS.md`](../AGENTS.md)
- **Agent owner:** `execution-agent`
- **Layer:** 5
- **Imports from:** `core/`, `config/`, `data/`, `markets/`, `strategies/`, `risk/`
- **Imported by:** `orchestration/`, `backtest/` (simulated broker), `api/`, `monitoring/`
- **Keywords:** `BrokerBase`, `BrokerRouter`, `Alpaca`, `Tinkoff`, `SimulatedBroker`, `OrderRequest`, `OrderResult`, `RetryPolicy`, `market_impact`, `slippage`, `sandbox_tracker`, `bond_simulated_broker`

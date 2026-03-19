# Broker Integration Contracts

How the system interfaces with brokers. All brokers implement `AbstractBroker` ABC.

## Broker Hierarchy

```
AbstractBroker (execution/base.py)
├── AlpacaBroker    → US equities (paper + live)
├── TinkoffBroker   → MOEX equities + bonds (sandbox + live)
└── SimulatedBroker → Backtesting (no real orders)
```

## AbstractBroker Interface

```python
class AbstractBroker(ABC):
    async def submit_order(order: Order) -> OrderResult
    async def cancel_order(order_id: str) -> None
    async def get_positions() -> list[Position]
    async def get_portfolio_value() -> Decimal
    async def get_order_status(order_id: str) -> OrderStatus
```

## TinkoffBroker (MOEX)

| Property | Value |
|----------|-------|
| API | T-Invest gRPC via `t-tech-investments` SDK |
| Endpoint (sandbox) | `sandbox-invest-public-api.tbank.ru:443` |
| Endpoint (live) | `invest-public-api.tbank.ru:443` |
| Auth | Bearer token (`FINALAYZE_TINKOFF_TOKEN`) |
| Order types | Limit only for bonds, Market/Limit for equities |
| Instrument ID | FIGI (e.g., `BBG000B9XRY4` for SBER) |
| Currency | RUB |
| Lot size | Per-instrument (SBER=10, GAZP=10, YNDX=1) |

**Key implementation details:**
- Must pass `target="invest-public-api.tbank.ru:443"` to `AsyncClient` (SDK default is broken)
- Set `GRPC_DNS_RESOLVER=native` env var (C-ares resolver fails)
- `AsyncClient` used as context manager: `async with client as services:`
- Separate instances for equities (`moex`) and bonds (`moex_bonds`)
- `cancel_order_safe()` returns bool (vs `cancel_order()` which raises)
- All open orders on startup treated as stale and cancelled

**SDK import:**
```python
from t_tech.invest import AsyncClient, CandleInterval, OrderDirection, OrderType
from t_tech.invest.sandbox.async_client import AsyncSandboxClient
```

## AlpacaBroker (US)

| Property | Value |
|----------|-------|
| API | REST via `alpaca-trade-api` SDK |
| Endpoint | Paper: `paper-api.alpaca.markets`, Live: `api.alpaca.markets` |
| Auth | API key + secret (`FINALAYZE_ALPACA_API_KEY`, `FINALAYZE_ALPACA_SECRET_KEY`) |
| Order types | Market, Limit |
| Instrument ID | Symbol (e.g., `AAPL`) |
| Currency | USD |

## BrokerRouter

Routes orders to correct broker by market:
```python
router.get_broker("moex")       → TinkoffBroker (equities)
router.get_broker("moex_bonds") → TinkoffBroker (bonds, separate instance)
router.get_broker("alpaca")     → AlpacaBroker
```

## SimulatedBroker (Backtesting)

- Fills all market orders at next-bar open price
- Applies transaction costs from `MOEX_COSTS` or `US_COSTS`
- Tracks positions, equity, drawdown in memory
- No external API calls

## Error Handling

All brokers use `RetryPolicy` with exponential backoff:
- Max retries: 3
- Base delay: 1s
- Retryable: network errors, rate limits
- Non-retryable: insufficient funds, invalid instrument

Broker errors raise `BrokerError` (from `core/exceptions.py`).

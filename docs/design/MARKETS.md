# Multi-Market Design

## Overview

Finalayze supports trading across multiple markets via distinct broker integrations.
Each market has its own data sources, broker, currency, and trading hours.

## Supported Markets

| Market | Exchange | Broker | Currency | Protocol | Data Source |
|--------|----------|--------|----------|----------|-------------|
| US | NYSE, NASDAQ | Alpaca | USD | REST + WebSocket | Finnhub, Alpha Vantage, yfinance |
| MOEX | Moscow Exchange | Tinkoff Invest | RUB | gRPC + Streaming | Tinkoff Invest API |

## Trading Hours (UTC)

| Market | Open | Close | Notes |
|--------|------|-------|-------|
| US | 14:30 | 21:00 | Pre-market 09:00, after-hours 01:00 |
| MOEX | 07:00 | 15:40 | Main session; evening session 16:05-20:50 |

**Overlap:** Both markets are open 14:30-15:40 UTC. Cross-market signals are possible
during this window.

## Broker Integration Details

### Alpaca (US)

- **SDK:** `alpaca-py`
- **Paper trading:** Free, separate endpoint
- **Live trading:** Commission-free
- **Market data:** REST polling + WebSocket streaming
- **Order types:** Market, limit, stop, stop-limit, trailing stop

### Tinkoff Invest (MOEX)

- **SDK:** `tinkoff-investments`
- **Sandbox:** Free, separate gRPC endpoint (`sandbox-invest-public-api.tinkoff.ru:443`)
- **Live trading:** Standard commissions apply
- **Market data:** gRPC streaming (free for all clients)
- **Instruments:** Identified by FIGI codes
- **Lot sizes:** MOEX instruments have lot sizes (e.g., SBER = 10 shares per lot)

## Currency Handling

- All P&L normalized to base currency (default: USD)
- USD/RUB rate fetched from Tinkoff API
- Currency rate stored in `currency_rates` hypertable
- Cross-market portfolio aggregation uses latest rate

## Status

**Phase 1:** US market (Alpaca) implemented.
**Phase 2:** MOEX market (Tinkoff) to be implemented.

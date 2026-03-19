# Markets

## Purpose
Market definitions, instrument registry, currency conversion, FX service, and market schedule utilities for US and MOEX exchanges.

## Layer
Layer 2 -- Data / Repository. Can import from layers 0-1. Never import from layers 3-6.

## Key Files
- `registry.py` -- MarketDefinition dataclass, MarketRegistry (US + MOEX), market hours check
- `instruments.py` -- Instrument dataclass (symbol, market_id, FIGI, lot_size, bond fields) and InstrumentRegistry
- `currency.py` -- CurrencyConverter with USDRUB/RUBUSD rates, `convert()` and `set_rate()`
- `fx_service.py` -- Live FX rate updates from external sources
- `schedule.py` -- Trading calendar and schedule utilities

## Public API
- `MarketRegistry`, `MarketDefinition` -- exchange definitions and open/closed checks
- `default_registry()` -- pre-loaded US + MOEX registry
- `InstrumentRegistry`, `Instrument` -- symbol lookup keyed by (symbol, market_id)
- `CurrencyConverter` -- cross-currency amount conversion

## Contracts
- Input: market_id strings ("us", "moex"), UTC-aware datetimes for hours check
- Output: MarketDefinition, Instrument, converted Decimal amounts
- Invariants: Market IDs are lowercase. Instrument lookup raises `InstrumentNotFoundError` on miss. MOEX instruments carry `figi` for Tinkoff API. `lot_size` defaults to 1 (MOEX often > 1).

## Testing
- Test location: `tests/unit/markets/`, `tests/unit/test_instruments.py`
- Run: `uv run pytest tests/unit/markets/ tests/unit/test_instruments.py -v`

## Common Patterns
- All dataclasses are frozen (immutable)
- MarketRegistry.is_market_open() converts UTC to local timezone before checking hours
- CurrencyConverter auto-derives inverse pair when `set_rate()` is called
- Weekend check: `weekday() >= 5` means Saturday or Sunday

# Phase 1: MOEX Equity Foundation - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix RUB position sizing, wire MOEX transaction costs, and integrate MOEX holiday calendar into both backtest engine and live TradingLoop. Phase 1 delivers correct MOEX infrastructure — Phase 2 tunes strategy parameters for positive PnL.

</domain>

<decisions>
## Implementation Decisions

### RUB Position Sizing
- Make the entire sizing pipeline currency-aware (not just a convert-at-entry hack)
- Use existing CurrencyConverter with CBR daily rates for RUB/USD conversion
- Portfolio equity tracked in RUB for MOEX segments
- Starting capital for MOEX backtest: 1,000,000 RUB
- US backward compatibility NOT required — can break US segments if needed (MVP is MOEX-only)
- Position size as % of equity: Claude's discretion to calibrate in Phase 2

### MOEX Commission Model
- Tariff: Трейдер (Trader) — 0.04% commission rate from trade amount
- No per-share commission (unlike US); purely percentage-based
- Spread and slippage bps: Claude's discretion based on MOEX blue chip liquidity analysis

### Holiday Calendar
- Add transferred holidays (переносные выходные) as static per-year lists (2020-2026)
- Existing 14 fixed holidays remain
- Wire holiday check into BOTH backtest engine AND live TradingLoop (unified approach)
- Skip MOEX non-trading days (holidays + weekends) in bar iteration

### Backtest Validation
- Phase 1 pass criteria: positions sized at 10-20% of equity AND positive PnL
- Test on ALL three MOEX segments: ru_blue_chips, ru_energy, ru_finance
- Backtest period: 2020-2025 (5 years, includes COVID + sanctions crisis)
- Data source: T-Invest API primary, MOEX ISS to fill gaps if T-Invest history is insufficient

### Claude's Discretion
- Exact spread/slippage bps values for MOEX
- Position size percentage (10-15% range)
- How to handle the USD-to-RUB pipeline migration internally
- Whether to refactor PositionSizingPipeline or add currency conversion layer

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CurrencyConverter` in `markets/`: already handles RUB/USD conversion
- `CBRFetcher` in `data/fetchers/cbr.py`: fetches daily USDRUB from CBR XML API
- `TransactionCosts` in `backtest/costs.py`: has `commission_rate` field (currently unused for MOEX)
- `moex_calendar.py` in `data/`: has `is_moex_holiday()` and `trading_days_gap()` — needs transferred holidays
- `MarketSchedule` in `markets/`: has MOEX weekday guard but no holiday integration

### Established Patterns
- `SegmentConfig.market` field distinguishes "us" from "moex" segments
- `TransactionCosts` is a frozen dataclass — create MOEX instance, don't modify US defaults
- Pipeline uses `HalfKelly` → `PositionSizingPipeline` → `PreTradeCheck` chain

### Integration Points
- `backtest/engine.py`: bar iteration loop — needs holiday skip
- `risk/position_sizing_pipeline.py`: pipeline floor (15%) — needs currency-aware equity
- `scripts/run_iteration.py`: creates TransactionCosts — needs MOEX cost config
- `core/trading_loop.py`: scheduler — needs holiday-aware scheduling gate

</code_context>

<specifics>
## Specific Ideas

- CBR daily rates already fetched and cached — reuse existing MarketDataLoader/CBRFetcher
- The `commission_rate` field on TransactionCosts already exists but defaults to 0 — just set it to 0.0004 for MOEX
- moex_calendar.py structure (frozenset of month/day tuples) can be extended with per-year transferred holidays dict

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-moex-equity-foundation*
*Context gathered: 2026-03-14*

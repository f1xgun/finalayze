# Phase 3: Bond Data Pipeline - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Fetch bond data from T-Invest API, compute bond math (YTM, duration, NKD/dirty price), cache macro data with persistence, and register bond instruments with FIGI mapping. Covers all bond types: OFZ-PD (fixed), OFZ-PK (floating), OFZ-IN (inflation-linked), and corporate bonds filtered by liquidity and risk. Phase 3 builds the data pipeline — Phase 4 handles execution and trading strategies.

</domain>

<decisions>
## Implementation Decisions

### Bond Universe Scope
- Broad MOEX bond market — all listed bonds, not just OFZ
- Liquidity filter: minimum 10M RUB/day turnover
- Risk filter: T-Invest risk_level <= 2 (low to moderate risk)
- Maturity filter: exclude bonds with < 3 months to maturity
- Include amortizing bonds (track remaining nominal from T-Invest amortization schedule)
- Include OFZ-IN (inflation-linked) — needed for breakeven inflation analysis
- Subordinated bonds: no special treatment (risk_level filter handles them)
- OAS for callable/puttable bonds (оферты): skip for MVP, use simple YTM
- Auto-discover bonds from T-Invest GetBonds() API, apply filters automatically
- Handles new issuances and maturities on startup/scheduled refresh

### Bond Segments
- By type: ru_ofz (government) and ru_corporate (filtered corporate bonds)
- Different risk profiles → different strategies per segment
- Existing ru_ofz_pd and ru_ofz_pk presets as starting point

### Bond Math Engine (Hybrid approach)
- Keep existing bond_math.py for standard fixed-coupon bonds (YTM, duration, convexity, DV01)
- Add QuantLib for edge cases: amortizing bonds and floating-coupon bonds (OFZ-PK)
- QuantLib FloatingRateBond for OFZ-PK pricing with RUONIA curve
- Effective duration via ±25bps rate shock for floaters (not analytical duration)
- Per-bond day-count convention (read from T-Invest metadata, not hardcoded actual/365)
- One-time validation test suite: compare bond_math.py vs QuantLib for 5-10 known bonds
- For amortizing bonds: Claude's discretion on which QuantLib functions are needed

### Floating Coupon Handling (OFZ-PK)
- Project future coupons using current RUONIA spot (7-day avg from MacroSnapshot) + bond spread
- QuantLib FloatingRateBond for YTM and duration calculations
- Effective duration via rate shock (±25bps RUONIA, reprice, numerical duration)

### OFZ-IN (Inflation-Linked) Handling
- Use CBR published daily indexation coefficients for nominal adjustment
- get_latest_published_cpi_month() already exists in CBRFetcher — extend it

### Macro Data
- Daily refresh + force-refresh on CBR meeting days (is_cbr_meeting_day() already exists)
- Extend MacroSnapshot with: breakeven inflation (OFZ-IN vs OFZ-PD spread)
- Yield curve points and additional macro fields: Claude's discretion on what's most impactful
- Data sources: CBR XML API primary, MOEX ISS API fallback (both with redundancy)
- Persist macro history to TimescaleDB via existing async SQLAlchemy (not in-memory only)
- Add MacroSnapshot ORM model, reuse existing connection pool and session factory

### Bond Data Caching
- Cache bond candles and coupon schedules in TimescaleDB
- Coupon schedules rarely change — fetch once, update periodically
- Daily candle append for active bonds
- Reduces API calls and supports backtesting

### Coupon Events
- Emit CouponEvent on ex-coupon date via event bus
- Strategies can react (sell before ex-date, hold for income, etc.)
- Aligns with existing event_driven architecture

### Amortizing Bond Position Sizing
- Track remaining nominal per bond from T-Invest amortization schedule
- Position size based on current (not original) face value

### Claude's Discretion
- Which QuantLib functions to use for amortizing bond calculations
- Whether to add yield curve points to MacroSnapshot (alongside inflation expectations)
- Bond candle timeframe (daily vs intraday)
- Auto-discovery refresh frequency (startup only vs daily)
- Exact ORM model schema for MacroSnapshot persistence
- Bond-specific logging and error handling patterns

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bond_math.py`: Pure Python YTM, duration, convexity, DV01, NKD — actual/365. Layer 0, no I/O.
- `TinkoffFetcher`: Already has `fetch_bond_info()`, `fetch_bond_coupons()`, `fetch_accrued_interest()` — wired and working
- `MacroCacheService` + `MacroContextProvider`: Daily refresh with CBR meeting detection
- `CBRFetcher`: Key rate, RUONIA, FX, CPI, CBR meeting calendar
- `BondInfo`, `CouponPayment`, `AccruedInterest` schemas in `core/schemas.py`
- `BondCycleProcessor` in `core/bond_cycle.py`: Bond trading loop processor (Phase 4 consumer)
- `BondSimulatedBroker` in `execution/bond_simulated_broker.py`: For bond backtesting
- `bond_engine.py`, `bond_walk_forward.py`, `bond_metrics.py`: Bond backtest infrastructure
- Event bus: Existing publish/subscribe pattern for CouponEvent emission

### Established Patterns
- `SegmentConfig.market` field: "us" vs "moex" — extend for bond segments
- Instrument registry with FIGI mapping: `markets/instruments.py`
- ORM models via SQLAlchemy 2.0 async — reuse for MacroSnapshot persistence
- APScheduler for background refresh tasks (sync → asyncio.to_thread for DB writes)

### Integration Points
- `instruments.py`: Add bond instrument registration with auto-discovery
- `macro_cache.py`: Extend MacroSnapshot fields, add DB persistence
- `cbr.py`: Add yield curve and inflation expectations endpoints
- `tinkoff_data.py`: Bond data methods already exist, may need GetBonds() list endpoint
- `core/schemas.py`: May need bond registry schema extensions
- Event bus: Wire CouponEvent emission

</code_context>

<specifics>
## Specific Ideas

- T-Invest GetBonds() API lists all available bonds — use for auto-discovery with filters
- CBR publishes zero-coupon yield curve at cbr.ru/hd_base/zcyc — extend CBRFetcher
- MOEX ISS API as fallback for yield curve data
- Amortization schedule available via T-Invest API — fetch alongside coupon schedule
- OFZ-IN daily indexation coefficients published by CBR — deterministic, no modeling needed
- ru_ofz_pd.yaml and ru_ofz_pk.yaml presets already exist as templates

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-bond-data-pipeline*
*Context gathered: 2026-03-14*

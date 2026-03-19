# Phase 3: Bond Data Pipeline - Research

**Researched:** 2026-03-14
**Domain:** MOEX bond data fetching, bond math (YTM/duration/NKD), macro data persistence, coupon events
**Confidence:** HIGH

## Summary

Phase 3 builds the bond data pipeline: auto-discovering bonds from T-Invest API, computing bond math (YTM, duration, NKD, dirty price) with a hybrid engine (existing bond_math.py + QuantLib for floaters/amortizers), persisting macro snapshots to TimescaleDB, and emitting coupon events via the Redis Streams event bus.

The codebase already has substantial bond infrastructure: `bond_math.py` (Layer 0, pure computation), `TinkoffFetcher` with `fetch_bond_info()`, `fetch_bond_coupons()`, `fetch_accrued_interest()`, `MacroCacheService` + `MacroContextProvider`, `CBRFetcher` (key rate, FX, CPI), `InstrumentRegistry` with 12 OFZ instruments pre-registered, `BondCycleProcessor` (Layer 6 consumer), and 14 bond-related test files. The primary work is: (1) adding auto-discovery via `services.instruments.bonds()`, (2) integrating QuantLib for floater/amortizer pricing, (3) adding MacroSnapshot ORM model for DB persistence, (4) extending CBRFetcher with yield curve data, and (5) wiring CouponEvent emission through the existing EventBus.

**Primary recommendation:** Build incrementally on existing infrastructure. The T-Invest SDK `bonds()` method returns all listed bonds with full metadata (including `risk_level`, `amortization_flag`, `floating_coupon_flag`, `liquidity_flag`). QuantLib 1.41 is available via `uv pip install QuantLib` and provides `FloatingRateBond` and `AmortizingFixedRateBond` classes. CBR publishes yield curve points at 12 standard maturities (0.25-30 years) via HTML table at `cbr.ru/hd_base/zcyc_params/`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Broad MOEX bond market -- all listed bonds, not just OFZ
- Liquidity filter: minimum 10M RUB/day turnover
- Risk filter: T-Invest risk_level <= 2 (low to moderate risk)
- Maturity filter: exclude bonds with < 3 months to maturity
- Include amortizing bonds (track remaining nominal from T-Invest amortization schedule)
- Include OFZ-IN (inflation-linked) -- needed for breakeven inflation analysis
- Subordinated bonds: no special treatment (risk_level filter handles them)
- OAS for callable/puttable bonds: skip for MVP, use simple YTM
- Auto-discover bonds from T-Invest GetBonds() API, apply filters automatically
- Handles new issuances and maturities on startup/scheduled refresh
- Bond segments: ru_ofz (government) and ru_corporate (filtered corporate)
- Keep existing bond_math.py for standard fixed-coupon bonds
- Add QuantLib for edge cases: amortizing bonds and floating-coupon bonds (OFZ-PK)
- QuantLib FloatingRateBond for OFZ-PK pricing with RUONIA curve
- Effective duration via +/-25bps rate shock for floaters (not analytical)
- Per-bond day-count convention (read from T-Invest metadata, not hardcoded actual/365)
- One-time validation test suite: compare bond_math.py vs QuantLib for 5-10 known bonds
- Floating coupon handling: project future coupons using current RUONIA spot (7d avg from MacroSnapshot) + bond spread
- OFZ-IN: use CBR published daily indexation coefficients; extend get_latest_published_cpi_month()
- Daily macro refresh + force-refresh on CBR meeting days
- Extend MacroSnapshot with breakeven inflation (OFZ-IN vs OFZ-PD spread)
- Data sources: CBR XML API primary, MOEX ISS API fallback
- Persist macro history to TimescaleDB via existing async SQLAlchemy
- Add MacroSnapshot ORM model, reuse existing connection pool and session factory
- Cache bond candles and coupon schedules in TimescaleDB
- Emit CouponEvent on ex-coupon date via event bus
- Track remaining nominal per bond from T-Invest amortization schedule

### Claude's Discretion
- Which QuantLib functions to use for amortizing bond calculations
- Whether to add yield curve points to MacroSnapshot (alongside inflation expectations)
- Bond candle timeframe (daily vs intraday)
- Auto-discovery refresh frequency (startup only vs daily)
- Exact ORM model schema for MacroSnapshot persistence
- Bond-specific logging and error handling patterns

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BDP-01 | Bond candle data fetched via T-Invest API (GetBonds, GetBondCoupons) | T-Invest SDK `bonds()` returns all listed bonds; `get_bond_coupons()` and `get_bond_events()` already wired. `BondsResponse.instruments` contains full Bond proto with 55 fields including risk_level, amortization_flag, floating_coupon_flag |
| BDP-02 | NKD (accrued coupon interest) and dirty price computed correctly | `bond_math.py` already has `nkd()` and `dirty_price()`. `fetch_accrued_interest()` in TinkoffFetcher fetches from API. Need per-bond day-count convention support |
| BDP-03 | MacroCacheService provides real-time CBR key rate and FX data | `MacroCacheService` + `MacroContextProvider` exist. CBR XML API for key rate + FX wired. Need: DB persistence, yield curve extension, breakeven inflation |
| BDP-04 | QuantLib integration for YTM, modified duration, convexity calculations | QuantLib 1.41 available. Use `FloatingRateBond` for OFZ-PK, `AmortizingFixedRateBond` for amortizers. Effective duration via rate shock |
| BDP-05 | Bond instrument registry with FIGI mapping for OFZ and corporate bonds | `InstrumentRegistry` with `list_by_type("moex", "bond")` exists. 12 OFZ pre-registered. Need auto-discovery via `bonds()` to populate dynamically |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| QuantLib | 1.41 | Floater/amortizer pricing, effective duration | Industry standard for bond math; FloatingRateBond and AmortizingFixedRateBond handle edge cases bond_math.py cannot |
| t-tech-investments | (installed) | T-Invest gRPC API client | Only SDK for MOEX bond data; `bonds()`, `bond_by()`, `get_bond_coupons()`, `get_bond_events()`, `get_accrued_interests()` |
| SQLAlchemy | 2.0 async | MacroSnapshot ORM persistence | Already used project-wide; reuse connection pool and session factory |
| httpx | (installed) | CBR XML API calls | Already used in CBRFetcher for key rate and FX |
| lxml | (installed) | XML parsing for CBR responses | Already used in CBRFetcher |
| redis.asyncio | (installed) | Event bus for CouponEvent | Existing EventBus uses Redis Streams |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic | v2 | Bond schemas (BondInfo, CouponPayment) | Already used for all schemas in core/schemas.py |
| structlog | (installed) | Bond-specific structured logging | Project standard |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| QuantLib for all bond math | bond_math.py for everything | QuantLib only needed for floaters + amortizers; bond_math.py is simpler and faster for fixed-coupon |
| CBR HTML scraping for yield curve | MOEX ISS API | CBR is authoritative source; MOEX ISS is fallback |
| Redis Streams for CouponEvent | In-process callback | Redis Streams already wired, enables decoupled consumers |

**Installation:**
```bash
uv add QuantLib
```

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/
  core/
    bond_math.py           # (existing) Pure Python fixed-coupon bond math
    bond_math_quantlib.py  # (NEW) QuantLib wrapper for floaters/amortizers
    schemas.py             # (extend) CouponEvent schema
    events.py              # (extend) CouponEvent stream constant
    models.py              # (extend) MacroSnapshotModel, BondCandleModel, CouponScheduleModel
  data/
    fetchers/
      tinkoff_data.py      # (extend) add fetch_all_bonds() for auto-discovery
      cbr.py               # (extend) add yield curve fetching, extend MacroSnapshot
    macro_cache.py          # (extend) add DB persistence, yield curve, breakeven inflation
    bond_discovery.py       # (NEW) auto-discovery service with filters
  markets/
    instruments.py          # (extend) dynamic bond registration from discovery
```

### Pattern 1: Bond Auto-Discovery Service
**What:** Fetches all bonds from T-Invest `bonds()` API, applies filters (liquidity, risk_level, maturity), registers qualifying bonds in InstrumentRegistry.
**When to use:** On startup and optionally daily refresh.
**Example:**
```python
# Source: T-Invest SDK verified via inspection
async def discover_bonds(services) -> list[Bond]:
    """Fetch all MOEX bonds and apply filters."""
    from t_tech.invest.schemas import InstrumentStatus
    resp = await services.instruments.bonds(
        instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
    )
    return resp.instruments  # list of Bond proto objects

# Each Bond has: figi, ticker, isin, name, nominal, initial_nominal,
# coupon_quantity_per_year, maturity_date, floating_coupon_flag,
# amortization_flag, risk_level, aci_value, lot, currency, class_code,
# subordinated_flag, liquidity_flag, sector, bond_type, call_date
```

### Pattern 2: Hybrid Bond Math Engine
**What:** Route bond math calls to either bond_math.py (fixed-coupon) or QuantLib wrapper (floaters/amortizers) based on bond type.
**When to use:** For all YTM/duration/convexity calculations.
**Example:**
```python
# bond_math_quantlib.py -- QuantLib wrapper
import QuantLib as ql

def effective_duration_rate_shock(
    bond: ql.FloatingRateBond,
    yield_curve: ql.YieldTermStructureHandle,
    shock_bps: int = 25,
) -> float:
    """Effective duration via +/-25bps parallel rate shock."""
    base_price = bond.cleanPrice()
    # Shock up
    up_curve = parallel_shift(yield_curve, +shock_bps)
    bond.setPricingEngine(ql.DiscountingBondEngine(up_curve))
    price_up = bond.cleanPrice()
    # Shock down
    dn_curve = parallel_shift(yield_curve, -shock_bps)
    bond.setPricingEngine(ql.DiscountingBondEngine(dn_curve))
    price_dn = bond.cleanPrice()
    # Numerical duration
    dy = 2 * shock_bps / 10000
    return (price_dn - price_up) / (2 * base_price * dy)
```

### Pattern 3: MacroSnapshot DB Persistence
**What:** Store MacroSnapshot to TimescaleDB on each refresh, using existing async session factory.
**When to use:** Daily refresh and CBR meeting day force-refresh.
**Example:**
```python
# Extend MacroCacheService.refresh() with DB write
async def _persist_snapshot(self, snapshot: MacroSnapshot) -> None:
    factory = get_async_session_factory()
    async with factory() as session:
        model = MacroSnapshotModel(
            timestamp=datetime.now(tz=UTC),
            key_rate=snapshot.key_rate,
            ruonia_7d_avg=snapshot.ruonia_7d_avg,
            cpi_yoy=snapshot.cpi_yoy,
            last_cbr_decision=snapshot.last_cbr_decision,
            breakeven_inflation=snapshot.breakeven_inflation,
            yield_curve=snapshot.yield_curve,
        )
        session.add(model)
        await session.commit()
```

### Pattern 4: CouponEvent Emission
**What:** Emit CouponEvent on ex-coupon date (record_date = T-2 before payment) via existing Redis Streams EventBus.
**When to use:** During daily bond cycle, check coupon schedule against current date.
**Example:**
```python
# New Pydantic schema in core/schemas.py or core/events.py
class CouponEvent(BaseModel):
    bond_figi: str
    bond_ticker: str
    coupon_date: date
    record_date: date
    amount_per_bond: Decimal
    coupon_number: int
    is_floating: bool = False

# Emit via EventBus
STREAM_COUPONS = "coupons"
await event_bus.publish(STREAM_COUPONS, coupon_event)
```

### Anti-Patterns to Avoid
- **Hardcoding bond universe:** Use auto-discovery, not static lists. Static OFZ instruments in instruments.py serve as fallback only.
- **Calling bond_math.py for floaters:** Newton-Raphson YTM with fixed coupon rate gives wrong YTM for OFZ-PK. Must use QuantLib FloatingRateBond.
- **Hardcoding actual/365 day count:** T-Invest Bond proto does not have a day_count field directly, but different bond types use different conventions. Read from issuer metadata or infer from class_code.
- **Synchronous DB writes in refresh():** MacroCacheService.refresh() is sync (called by APScheduler). Use `asyncio.to_thread()` or `asyncio.run()` for DB persistence.
- **Blocking gRPC in event loop:** TinkoffFetcher uses `asyncio.run()` for sync interface. For bond discovery (potentially 1000+ bonds), consider batch processing.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Floating-rate bond YTM | Custom Newton-Raphson with projected coupons | QuantLib `FloatingRateBond` + RUONIA curve | RUONIA curve construction, coupon projection, day count handling are deceptively complex |
| Amortizing bond pricing | Modified bond_math.py with decreasing nominal | QuantLib `AmortizingFixedRateBond` | Amortization schedule interaction with coupon payments creates many edge cases |
| Effective duration for floaters | Manual reprice loop | QuantLib rate shock via `DiscountingBondEngine` | Curve construction and parallel shift logic is standard QuantLib functionality |
| Yield curve interpolation | Linear interpolation between CBR points | QuantLib `ZeroCurve` or `PiecewiseYieldCurve` | Proper interpolation/extrapolation with consistent day counting |
| Event bus | Custom pub/sub | Existing `EventBus` (Redis Streams) | Already wired, supports consumer groups, at-least-once delivery |

**Key insight:** Bond math for standard fixed-coupon OFZ is well-served by the existing `bond_math.py`. QuantLib is needed only for the edge cases (floaters, amortizers, inflation-linked) where the cash flow structure is non-trivial.

## Common Pitfalls

### Pitfall 1: T-Invest API Rate Limiting
**What goes wrong:** Fetching 1000+ bonds with individual `bond_by()` calls hits rate limits.
**Why it happens:** T-Invest API has per-minute request limits (~300 unary requests/minute).
**How to avoid:** Use `bonds()` (single call returns all bonds) for discovery. Use `get_bond_coupons()` and `get_bond_events()` in batch with rate limiter.
**Warning signs:** gRPC RESOURCE_EXHAUSTED errors.

### Pitfall 2: QuantLib Date Convention Mismatch
**What goes wrong:** QuantLib uses its own `ql.Date` type, not Python `datetime.date`. Conversion errors cause off-by-one in coupon dates.
**Why it happens:** QuantLib `Date(day, month, year)` uses integer month (1-12), different from `date(year, month, day)`.
**How to avoid:** Create explicit conversion helpers: `to_ql_date(d: date) -> ql.Date` and `from_ql_date(d: ql.Date) -> date`.
**Warning signs:** Duration/YTM values differ from expected by small amounts.

### Pitfall 3: RUONIA Curve Construction
**What goes wrong:** FloatingRateBond pricing requires a proper RUONIA forward curve, not just a single spot rate.
**Why it happens:** QuantLib `FloatingRateBond` expects a `YieldTermStructureHandle` with a full curve.
**How to avoid:** Build a flat curve from current RUONIA spot (7-day avg) as a starting point. For MVP, a `FlatForward` curve is acceptable. Enhance to bootstrapped curve later.
**Warning signs:** Floater YTM differs significantly from market-quoted YTM.

### Pitfall 4: Bond Price in Percentage vs Absolute
**What goes wrong:** MOEX quotes bond prices as percentage of face value (e.g., 85.50%). Mixing percentage and absolute prices corrupts all calculations.
**Why it happens:** Different systems use different conventions. T-Invest API uses percentage for `aci_value` but absolute `MoneyValue` for some fields.
**How to avoid:** Consistently use `clean_price_pct` (Decimal, % of face) throughout. Document convention on every function.
**Warning signs:** YTM computed as 7000% instead of 7%.

### Pitfall 5: Sync/Async Boundary in MacroCacheService
**What goes wrong:** MacroCacheService.refresh() is sync (APScheduler), but DB writes need async SQLAlchemy.
**Why it happens:** APScheduler BackgroundScheduler runs sync callbacks. SQLAlchemy 2.0 async requires an event loop.
**How to avoid:** Use `asyncio.to_thread()` pattern or create a new event loop for DB writes. The existing pattern comment in macro_cache.py says: "Future LiveMacroContextProvider with httpx must use asyncio.to_thread()."
**Warning signs:** "no running event loop" errors, deadlocks.

### Pitfall 6: Bond Discovery Filter Ordering
**What goes wrong:** Applying liquidity filter before maturity filter wastes API calls checking turnover for bonds about to mature.
**Why it happens:** Not thinking about filter cost hierarchy.
**How to avoid:** Filter order: (1) maturity > 3 months (free, from bond metadata), (2) risk_level <= 2 (free, from bond metadata), (3) currency = RUB (free), (4) api_trade_available_flag (free), then (5) liquidity check (may require separate API call or cached turnover data).
**Warning signs:** Discovery taking 10+ minutes instead of seconds.

## Code Examples

### T-Invest bonds() API -- All Bonds Listing
```python
# Source: Verified via SDK inspection -- InstrumentsService.bonds()
from t_tech.invest import AsyncClient
from t_tech.invest.schemas import InstrumentStatus

async def fetch_all_bonds(token: str, target: str) -> list:
    """Fetch all listed bonds from T-Invest API."""
    client = AsyncClient(token, target=target)
    async with client as services:
        resp = await services.instruments.bonds(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )
        return resp.instruments

# Each Bond object has 55 fields including:
# figi, ticker, isin, name, lot, currency, nominal, initial_nominal,
# coupon_quantity_per_year, maturity_date, floating_coupon_flag,
# amortization_flag, risk_level (RiskLevel enum), aci_value,
# class_code, subordinated_flag, liquidity_flag, sector, bond_type,
# call_date, perpetual_flag, api_trade_available_flag
```

### T-Invest get_bond_events() -- Amortization Schedule
```python
# Source: Verified via SDK inspection
from t_tech.invest.schemas import GetBondEventsRequest, EventType

async def fetch_amortization_schedule(services, instrument_id: str):
    """Fetch amortization events for a bond."""
    from datetime import datetime, UTC
    request = GetBondEventsRequest(
        instrument_id=instrument_id,
        type=EventType.EVENT_TYPE_MTY,  # maturity/amortization events
        from_=datetime(2020, 1, 1, tzinfo=UTC),
        to=datetime(2040, 1, 1, tzinfo=UTC),
    )
    resp = await services.instruments.get_bond_events(request)
    return resp.events
    # BondEvent fields: instrument_id, event_number, event_date,
    # event_type, event_total_vol, pay_one_bond, money_flow_val,
    # coupon_start_date, coupon_end_date, coupon_period, coupon_interest_rate
```

### QuantLib FloatingRateBond for OFZ-PK
```python
# Source: QuantLib-Python docs (quantlib-python-docs.readthedocs.io)
import QuantLib as ql

def price_floating_rate_bond(
    settlement_date: ql.Date,
    maturity_date: ql.Date,
    face_value: float,
    spread: float,  # e.g. 0.013 for 130bps spread over RUONIA
    ruonia_rate: float,  # current RUONIA 7-day avg
    coupon_frequency: int = 2,
) -> tuple[float, float]:
    """Price an OFZ-PK floater. Returns (clean_price, ytm)."""
    ql.Settings.instance().evaluationDate = settlement_date
    calendar = ql.Russia()
    day_count = ql.Actual365Fixed()  # Russian market standard

    # Build flat RUONIA forward curve (MVP simplification)
    ruonia_curve = ql.FlatForward(settlement_date, ruonia_rate, day_count)
    ruonia_handle = ql.YieldTermStructureHandle(ruonia_curve)

    # Create RUONIA index
    ruonia_index = ql.OvernightIndex(
        "RUONIA", 1, ql.RUBCurrency(), calendar, day_count, ruonia_handle
    )

    # Bond schedule
    schedule = ql.Schedule(
        settlement_date, maturity_date,
        ql.Period(int(12 / coupon_frequency), ql.Months),
        calendar,
        ql.ModifiedFollowing, ql.ModifiedFollowing,
        ql.DateGeneration.Backward, False,
    )

    bond = ql.FloatingRateBond(
        settlementDays=1,
        faceAmount=face_value,
        schedule=schedule,
        index=ruonia_index,
        paymentDayCounter=day_count,
        spreads=[spread],
    )

    # Pricing engine
    bond.setPricingEngine(ql.DiscountingBondEngine(ruonia_handle))

    return bond.cleanPrice(), bond.bondYield(day_count, ql.Compounded, ql.Semiannual)
```

### MacroSnapshot ORM Model
```python
# Source: Existing models.py patterns in project
class MacroSnapshotModel(Base):
    """Macro data snapshot persisted to TimescaleDB."""

    __tablename__ = "macro_snapshots"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    key_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    ruonia_7d_avg: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    cpi_yoy: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    last_cbr_decision: Mapped[str | None] = mapped_column(String(10))
    breakeven_inflation: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    yield_curve: Mapped[dict | None] = mapped_column(JSONB)  # {maturity_years: yield_pct}
    usdrub: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
```

### CBR Yield Curve Scraping
```python
# Source: Verified via WebFetch -- CBR publishes at cbr.ru/hd_base/zcyc_params/
# Data format: HTML table with maturities (0.25-30y) and yields (% per annum)
# Maturities: [0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 10, 15, 20, 30]

_ZCYC_URL = "https://www.cbr.ru/hd_base/zcyc_params/"

def fetch_yield_curve(self, as_of: date) -> dict[str, Decimal]:
    """Fetch zero-coupon yield curve from CBR.

    Returns dict mapping maturity (years as string) to yield (% per annum).
    """
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": as_of.strftime("%d.%m.%Y"),
        "UniDbQuery.To": as_of.strftime("%d.%m.%Y"),
    }
    content = self._request("GET", _ZCYC_URL, params=params)
    return self._parse_zcyc_html(content)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded OFZ list (12 bonds) | Auto-discovery via `bonds()` API | Phase 3 | Captures all MOEX bonds, handles new issuances |
| In-memory MacroSnapshot only | TimescaleDB persistence | Phase 3 | Historical macro data for backtesting/ML features |
| bond_math.py for all bonds | Hybrid: bond_math.py + QuantLib | Phase 3 | Correct pricing for floaters and amortizers |
| Static CPI data dict | CBR-fetched + breakeven inflation | Phase 3 | Real-time inflation expectations |

**Deprecated/outdated:**
- `DEFAULT_MOEX_OFZ_INSTRUMENTS` static list: Remains as fallback but auto-discovery is primary
- RUONIA proxy (`key_rate - 50bps`): Acceptable for backtest but live should use actual RUONIA from CBR

## Open Questions

1. **CBR yield curve endpoint format**
   - What we know: CBR publishes at `/hd_base/zcyc_params/` with 12 maturity points as HTML table
   - What's unclear: No documented XML/JSON API; will need HTML scraping with lxml
   - Recommendation: Scrape HTML table; add MOEX ISS API (`moex.com/s478`) as fallback. Both are reliable.

2. **Bond liquidity data source**
   - What we know: Filter requires 10M RUB/day turnover
   - What's unclear: T-Invest Bond proto doesn't have a `daily_turnover` field; `liquidity_flag` is boolean only
   - Recommendation: Use `liquidity_flag=True` as primary filter (from Bond proto). For precise turnover, fetch last 5 days of candle volume * close price. Can also use MOEX ISS API for aggregate turnover data.

3. **QuantLib Russia calendar completeness**
   - What we know: QuantLib has `ql.Russia()` calendar
   - What's unclear: Whether it includes all MOEX transferred holidays (government decrees)
   - Recommendation: Our existing `moex_calendar.py` with static frozensets is more complete. Use for business day calculations, QuantLib calendar only for QuantLib internal scheduling.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/ -x -q` |
| Full suite command | `uv run pytest --cov=src/finalayze` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BDP-01 | Bond discovery fetches and filters bonds from T-Invest | unit (mock gRPC) | `uv run pytest tests/unit/test_bond_discovery.py -x` | -- Wave 0 |
| BDP-01 | Bond candles cached to TimescaleDB | unit | `uv run pytest tests/unit/test_bond_candle_cache.py -x` | -- Wave 0 |
| BDP-02 | NKD computed correctly with per-bond day count | unit | `uv run pytest tests/unit/test_bond_math.py -x` | Exists (extend) |
| BDP-02 | Dirty price = clean + NKD | unit | `uv run pytest tests/unit/test_bond_math.py -x` | Exists |
| BDP-03 | MacroSnapshot persisted to DB | unit | `uv run pytest tests/unit/test_macro_persistence.py -x` | -- Wave 0 |
| BDP-03 | Yield curve fetched from CBR | unit | `uv run pytest tests/unit/test_cbr_yield_curve.py -x` | -- Wave 0 |
| BDP-03 | Breakeven inflation computed | unit | `uv run pytest tests/unit/test_macro_cache.py -x` | Exists (extend) |
| BDP-04 | QuantLib YTM matches bond_math.py for fixed bonds | unit | `uv run pytest tests/unit/test_bond_math_quantlib.py -x` | -- Wave 0 |
| BDP-04 | QuantLib FloatingRateBond prices OFZ-PK | unit | `uv run pytest tests/unit/test_bond_math_quantlib.py -x` | -- Wave 0 |
| BDP-04 | Effective duration via rate shock | unit | `uv run pytest tests/unit/test_bond_math_quantlib.py -x` | -- Wave 0 |
| BDP-05 | Bond registry populated from auto-discovery | unit | `uv run pytest tests/unit/test_bond_discovery.py -x` | -- Wave 0 |
| BDP-05 | FIGI mapping works for discovered bonds | unit | `uv run pytest tests/unit/test_instruments.py -x` | Exists (extend) |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/ -x -q --timeout=30`
- **Per wave merge:** `uv run pytest --cov=src/finalayze`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_bond_discovery.py` -- covers BDP-01, BDP-05
- [ ] `tests/unit/test_bond_math_quantlib.py` -- covers BDP-04
- [ ] `tests/unit/test_macro_persistence.py` -- covers BDP-03
- [ ] `tests/unit/test_cbr_yield_curve.py` -- covers BDP-03
- [ ] `tests/unit/test_bond_candle_cache.py` -- covers BDP-01
- [ ] QuantLib install: `uv add QuantLib` -- not yet in dependencies

## Discretion Recommendations

Based on research, here are recommendations for Claude's discretion areas:

1. **QuantLib for amortizing bonds:** Use `ql.AmortizingFixedRateBond` with amortization schedule from `get_bond_events(EventType.EVENT_TYPE_MTY)`. The `pay_one_bond` field gives the amortization amount per event.

2. **Yield curve in MacroSnapshot:** YES, add yield curve points. Store as JSONB `{"0.25": 12.85, "0.5": 12.90, ..., "30.0": 13.60}`. Critical for: (a) breakeven inflation calculation (OFZ-IN vs OFZ-PD spread at matching maturities), (b) duration-rotation strategy signals, (c) future curve-based valuation.

3. **Bond candle timeframe:** Daily only (`"1d"`). Bond trading is low-frequency; intraday adds complexity without value for carry/duration strategies.

4. **Auto-discovery refresh frequency:** On startup + once daily at 07:00 UTC (before MOEX open at 07:00 UTC). New bond issuances happen infrequently; daily is sufficient. Maturities are handled by maturity filter automatically.

5. **MacroSnapshot ORM schema:** See code example above. Timestamp as PK (TimescaleDB hypertable candidate), yield_curve as JSONB, all rate fields as Numeric(5,2).

6. **Logging patterns:** Use existing structlog pattern. Bond-specific events: `bond_discovered`, `bond_filtered`, `bond_matured`, `coupon_event_emitted`, `macro_snapshot_persisted`, `quantlib_pricing_failed` (with fallback to bond_math.py).

## Sources

### Primary (HIGH confidence)
- T-Invest SDK inspection: `InstrumentsService.bonds()`, `bond_by()`, `get_bond_coupons()`, `get_bond_events()` -- verified via Python introspection
- Bond proto fields (55 fields) -- verified via protobuf DESCRIPTOR inspection
- EventType enum: `EVENT_TYPE_CPN` (coupon), `EVENT_TYPE_CALL` (call), `EVENT_TYPE_MTY` (maturity), `EVENT_TYPE_CONV` (conversion) -- verified
- BondEvent fields (21 fields including coupon_interest_rate, coupon_period) -- verified
- Existing codebase: bond_math.py, tinkoff_data.py, cbr.py, macro_cache.py, events.py, models.py, instruments.py -- all read and analyzed
- QuantLib 1.41 availability confirmed via `uv pip install --dry-run QuantLib`

### Secondary (MEDIUM confidence)
- [QuantLib-Python docs](https://quantlib-python-docs.readthedocs.io/en/latest/instruments/bonds.html) -- FloatingRateBond, AmortizingFixedRateBond API
- [CBR yield curve page](https://www.cbr.ru/hd_base/zcyc_params/) -- HTML table format with 12 maturity points, verified via WebFetch
- [CBR ZCYC endpoint](https://www.cbr.ru/eng/hd_base/zcyc_params/zcyc/) -- chart visualization, data available 2003-2026

### Tertiary (LOW confidence)
- QuantLib `ql.Russia()` calendar completeness -- needs validation against our moex_calendar.py
- MOEX ISS API for yield curve fallback (`moex.com/s478`) -- URL referenced by CBR page but not yet tested

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries verified installed or available, SDK methods inspected
- Architecture: HIGH - building on well-established existing patterns (instruments.py, events.py, models.py)
- Pitfalls: HIGH - based on direct SDK inspection and existing codebase analysis
- QuantLib integration: MEDIUM - API verified via docs but OFZ-PK pricing with RUONIA not yet tested end-to-end

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable domain, QuantLib releases quarterly)

# Stack Research: v2.0 MOEX Profitability

**Domain:** MOEX-native profitable strategies, ML with Russian macro features
**Researched:** 2026-03-20
**Confidence:** HIGH (all recommendations verified against existing codebase)

## Context: What Already Exists (v1.0 stack)

The existing codebase (Python 3.12, async-first, uv) already provides:
- `t-tech-investments` gRPC SDK -- candles, instruments, dividends (`tinkoff_data.py`)
- `CBRFetcher` -- key rate (SOAP XML), FX rates (REST XML), yield curve, CPI data
- `MacroSnapshot` + `MacroCacheService` -- key_rate, USDRUB, CPI, RUONIA cached daily
- `MoexISSFetcher` -- IMOEX index candles, market turnover
- `yfinance` -- Brent crude (`BZ=F`) already in `MoexMarketData.commodity_candles`
- `PairsStrategy` -- Kalman filter + Engle-Granger cointegration spread z-score
- `DividendGapStrategy` -- basic gap closure with `DividendEntry`/`_GapTracker`
- `CBRCalendar` + `CBRStrategyWrapper` -- CBR rate event contrarian signals
- `rub_oil_regime.py` -- RUB/oil decorrelation regime detection
- `commodity_currency.py` -- Brent-RUB premium for energy exporters
- ML features: `brent_zscore_60d`, `real_rate_zscore` (45 features total)
- `statsmodels` (cointegration), `pandas-ta`, `numpy`, `scipy`, `hmmlearn`, `arch`
- `XGBoost`, `LightGBM`, `CatBoost`, `PyTorch` -- ML ensemble pipeline
- `QuantLib` -- bond math (YTM, duration, convexity)
- `Telethon` -- Telegram channel reader
- `feedparser` -- RSS parsing (RBC, Interfax, TASS)
- `aiogram` / Telegram alerting
- Full strategy framework: `BaseStrategy`, `StrategyCombiner`, ADX routing, presets

---

## Key Finding: Zero New Dependencies Required

All v2.0 MOEX profitability features can be built using the existing dependency set. The work is purely at the application layer: new strategy modules, ML feature extensions, combiner configuration, and risk pipeline wiring.

---

## Recommended Stack (Application-Layer Additions Only)

### New Modules to Create

| New Module | Purpose | Existing Deps Used | Complexity |
|-----------|---------|-------------------|------------|
| `strategies/moex_sector_rotation.py` | Sector rotation using MOEX ISS sector indices | `MoexISSFetcher`, `pandas`, `numpy` | Medium |
| `strategies/pref_arb.py` OR extend `pairs.py` | Preferred share arbitrage (SBER/SBERP) | `statsmodels.coint`, `numpy` (existing `PairsStrategy`) | Low |
| `risk/cbr_regime_gate.py` | CBR rate regime classification + equity gating | `MacroSnapshot`, `MacroCacheService` | Low |
| `risk/portfolio_allocator.py` | 40% OFZ + 60% equity with RUB crisis brake | `CircuitBreaker`, `MacroSnapshot`, `rub_oil_regime` | Medium |
| Extend `ml/features/technical.py` | 10 new Russian macro ML features | `MoexMarketData` (existing schema) | Medium |
| Extend `strategies/dividend_gap.py` | Expand dividend events, regime filtering | `TinkoffFetcher.get_dividends()`, existing strategy | Medium |
| Extend `data/loader.py` | Fetch MOEX sector index candles | `MoexISSFetcher` (existing) | Low |

### Existing Modules to Extend

| Existing Module | Extension | What Changes |
|----------------|-----------|-------------|
| `strategies/combiner.py` | Add Brent gate for ru_energy, CBR regime gate for all ru_* | New hook logic in `_on_generate_start()` |
| `strategies/pairs.py` | Configure for MOEX preferred/common pairs | Add pair definitions, adjust Kalman parameters |
| `strategies/dividend_gap.py` | Expand from 43 to 150+ events, add regime filter | Batch dividend fetch, sector-conditional entry |
| `ml/features/technical.py` | Add 10 macro features (see table below) | New `_compute_moex_macro_features()` function |
| `data/loader.py` | Fetch MOEX sector index candles | Add `_fetch_sector_indices()` method |
| `config/segments.py` | Add new strategies to `active_strategies` lists | YAML preset weights |
| `strategies/presets/*.yaml` | Add weights for dividend_gap, sector_rotation, pairs | Configuration only |

---

## Detailed Stack Decisions by Feature

### 1. Dividend Gap Closure Expansion

**Decision:** Extend existing `DividendGapStrategy`, no new deps.

**What exists:**
- `strategies/dividend_gap.py`: `DividendEntry`, `_GapTracker`, gap closure logic
- `TinkoffFetcher`: already wired for `instruments.get_dividends()` per symbol
- Strategy framework: `BaseStrategy` interface, combiner integration

**What to build:**
- `DividendCalendar` class: batch-load historical dividends for all MOEX blue chips
- Regime filter: skip entries during CRISIS regime (use existing `rub_oil_regime`)
- Sector-conditional entry: skip energy dividends when Brent < $55
- Historical dividend persistence in TimescaleDB (use existing `alembic` migrations)

**Why no new deps:** T-Invest gRPC already provides `get_dividends()`. All filtering uses existing regime/commodity modules.

### 2. CBR Rate Regime Gating

**Decision:** New `risk/cbr_regime_gate.py`, no new deps.

**What exists:**
- `CBRFetcher.fetch_key_rate()` -- fetches full key rate history
- `MacroSnapshot.key_rate`, `MacroSnapshot.last_cbr_decision` -- point-in-time macro state
- `MacroCacheService` -- daily refresh, CBR-day force-refresh
- `CBR_MEETINGS` -- static calendar of CBR rate decisions with surprise detection

**What to build:**
- `CBRRegimeGate` class with three regimes:
  - EASING: last 2+ decisions were cuts, or rate delta 3m < -100bps
  - HOLD: rate unchanged for 2+ meetings
  - TIGHTENING: last 2+ decisions were hikes, or rate delta 3m > +100bps
- Wire into `StrategyCombiner._on_generate_start()` hook:
  - TIGHTENING: suppress financials (SBER, VTBR), boost defensive sectors
  - EASING: boost financials, suppress OFZ PD duration (rates falling = bond rally handled by carry)
- Uses only existing `MacroSnapshot` fields. No new data sources.

### 3. Brent-Conditional Energy Trading

**Decision:** Logic in combiner hook, no new module needed.

**What exists:**
- `MarketContext.moex_data.commodity_candles["BZ=F"]` -- Brent candles
- `commodity_currency_premium()` in `risk/commodity_currency.py` -- spread signal
- `brent_zscore_60d` ML feature in `technical.py`

**What to build:**
- Gate in `StrategyCombiner._on_generate_start()` for `ru_energy` segment:
  - Brent 20d return < -15%: suppress all energy longs
  - Brent 20d return > +10%: boost energy signal confidence by 20%
  - Brent < $55 absolute: block energy entries entirely
- Computation uses existing `commodity_candles` data path. No new fetcher needed.

### 4. Preferred Share Arbitrage

**Decision:** Extend existing `PairsStrategy`, no new deps.

**What exists:**
- `PairsStrategy` with Kalman filter hedge ratio, cointegration test, z-score spread
- `set_peer_candles(symbol, candles)` API for providing counterpart data
- `statsmodels.tsa.stattools.coint` already imported

**What to build:**
- MOEX pair definitions (configuration, not code):
  - SBER/SBERP (ru_finance): structural discount ~5-8%, widens pre-dividend
  - TATN/TATNP (ru_energy): structural discount ~10-15%
  - SNGS/SNGSP (ru_energy): driven by hidden FX reserves
- Kalman parameter tuning for MOEX pairs:
  - Wider z-score entry bands (2.5 vs 2.0) due to lower liquidity
  - Larger observation noise R (0.01 vs 0.001) for MOEX spreads
  - Seasonal dividend adjustment: preferred/common spread widens before record date
- Wire peer candle loading in backtest engine to ensure both legs available

**Key consideration:** `PairsStrategy.generate_signal()` currently requires peer candles via `set_peer_candles()`. The backtest engine needs to call this for each pair leg. This is a wiring issue in `backtest/engine.py`, not a library issue.

### 5. MOEX Sector Rotation

**Decision:** New `strategies/moex_sector_rotation.py`, no new deps.

**What exists:**
- `MoexISSFetcher` fetches any MOEX index candles (tested with IMOEX)
- `MarketDataLoader` orchestrates ambient data loading
- ADX regime routing in combiner

**What to build:**
- Fetch 8 MOEX sector index candles via `MoexISSFetcher`:
  ```
  MOEXOG (Oil&Gas), MOEXFN (Finance), MOEXMM (Metals),
  MOEXCN (Consumer), MOEXTL (Telecom), MOEXCH (Chemicals),
  MOEXEU (Utilities), MOEXIT (IT)
  ```
- `MoexSectorRotationStrategy(BaseStrategy)`:
  - Compute 21-day relative strength vs IMOEX for each sector
  - Rank sectors, overweight top 3, underweight bottom 3
  - CBR-conditioned: financials boost during EASING, energy boost when Brent rising
- Extend `MarketDataLoader._load_moex()` to fetch sector index candles
- Store in `MoexMarketData` (add `sector_index_candles: dict[str, tuple[Candle, ...]]`)

**Why no new deps:** `MoexISSFetcher` already handles MOEX ISS API pagination, timezone conversion, and retry logic. Just needs different ticker symbols.

### 6. ML with Russian Macro Features

**Decision:** Extend `ml/features/technical.py`, no new deps.

**What exists:**
- `_compute_commodity_features()` returns `brent_zscore_60d`
- `_compute_macro_features()` returns `real_rate_zscore`
- `MoexMarketData` carries `fx_rates`, `key_rates`, `commodity_candles`, `turnover`
- `MarketContext` carries `benchmark_candles` (IMOEX)
- Feature selection pipeline, sequential bootstrapping, quality gates

**New features to add (10 features):**

| Feature Name | Formula | Data Source | Rationale |
|-------------|---------|-------------|-----------|
| `cbr_key_rate_level` | Current rate normalized to [0,1] range (rate/30) | `key_rates` | Rate level affects all MOEX sectors differently |
| `cbr_rate_delta_3m` | rate_now - rate_3months_ago | `key_rates` | Direction of monetary policy |
| `cbr_rate_direction` | +1 hiking, 0 hold, -1 cutting (from last 2 decisions) | `key_rates` | Categorical regime indicator |
| `usdrub_return_20d` | 20-day log return of USDRUB | `fx_rates` | RUB weakness/strength signal |
| `usdrub_zscore_60d` | 60-day z-score of USDRUB level | `fx_rates` | Mean-reversion signal for FX |
| `usdrub_vol_20d` | 20-day realized vol of USDRUB (annualized) | `fx_rates` | FX stress indicator |
| `brent_return_20d` | 20-day log return of Brent | `commodity_candles` | Energy sector leading indicator |
| `brent_rub_spread` | brent_return_20d - usdrub_return_20d | Both | Commodity-currency decorrelation signal |
| `imoex_relative_21d` | stock_return_21d / imoex_return_21d | `benchmark_candles` | Relative strength vs market |
| `moex_turnover_zscore` | 60-day z-score of MOEX market turnover | `turnover` | Liquidity/sentiment indicator |

**Implementation:** Add `_compute_moex_macro_features_extended()` function, called from `compute_features()` when `moex_data is not None`. Follows existing pattern of `_compute_commodity_features()`.

### 7. Portfolio Allocation (40% OFZ + 60% Equity)

**Decision:** New `risk/portfolio_allocator.py`, no new deps.

**What exists:**
- `PortfolioLayer` enum (CORE/STRATEGIC/TACTICAL/SHORT)
- `CircuitBreaker` with 3-level escalation
- `rub_oil_regime.py` -- CRISIS detection
- `MacroSnapshot` -- key_rate, USDRUB
- Per-segment `max_allocation_pct` in `SegmentConfig`

**What to build:**
- `PortfolioAllocator` class:
  - Normal: 40% OFZ (ru_ofz_pk + ru_ofz_pd), 60% equity (all ru_* stock segments)
  - RUB crisis: 80% OFZ, 20% equity (triggered by `rub_oil_regime` CRISIS)
  - Rate cutting cycle: shift OFZ from PK to PD (duration exposure for capital gains)
- Wire into `TradingLoop` strategy cycle and backtest engine

---

## What NOT to Add

| Avoid | Why | Already Have |
|-------|-----|-------------|
| `moexalgo` | Third-party MOEX wrapper, adds complexity, not maintained | `MoexISSFetcher` (direct ISS API) |
| `investpy` | Deprecated, Investing.com blocked scraping | `CBRFetcher` + `MoexISSFetcher` |
| `ta-lib` (C library) | Requires system-level install, pandas-ta covers all needed indicators | `pandas-ta` (already installed) |
| `zipline` / `backtrader` | Entire backtesting frameworks, conflict with existing engine | Custom `backtest/engine.py` |
| New async HTTP client (aiohttp) | Would duplicate existing HTTP capability | `httpx` (already installed) |
| `sklearn` feature engineering | Custom features needed for MOEX-specific signals | Manual feature computation in `technical.py` |
| Any new ML framework | Existing ensemble (XGB+LGBM+CatBoost) is sufficient | `ml/models/` (already operational) |
| `fredapi` / US macro data | US market is out of scope for v2.0 | N/A |

---

## Data Pipeline: New vs Existing

| Data Need | Already Fetched? | Source | Action |
|-----------|-----------------|--------|--------|
| Historical dividends (150+ events) | Partially (per-symbol on demand) | T-Invest gRPC | Batch fetch for all blue chips, cache results |
| CBR key rate history | YES | CBR SOAP API | Already in `MacroSnapshot` |
| USDRUB daily rates | YES | CBR REST API | Already in `MoexMarketData.fx_rates` |
| Brent crude daily | YES | yfinance `BZ=F` | Already in `commodity_candles` |
| MOEX sector indices (8 tickers) | **NO** | MOEX ISS | **New: fetch MOEXOG, MOEXFN, MOEXMM, etc.** |
| IMOEX index | YES | MOEX ISS | Already used as benchmark |
| MOEX turnover | YES | MOEX ISS | Already in `MoexMarketData.turnover` |
| CPI YoY data | YES (static) | Hardcoded in `cbr.py` | Update static table through 2026 |

**Only truly new data to fetch:** MOEX sector index candles (8 indices via existing `MoexISSFetcher`).

---

## Schema Additions

### MoexMarketData (extend existing dataclass)

```python
@dataclass(frozen=True)
class MoexMarketData:
    fx_rates: tuple[FXRate, ...] | None = None
    key_rates: tuple[KeyRateRecord, ...] | None = None
    commodity_candles: dict[str, tuple[Candle, ...]] | None = None
    turnover: tuple[TurnoverRecord, ...] | None = None
    # NEW for v2.0:
    sector_index_candles: dict[str, tuple[Candle, ...]] | None = None  # "MOEXOG" -> candles
```

### Strategy Preset Updates (YAML)

```yaml
# strategies/presets/ru_blue_chips.yaml -- add:
dividend_gap: 0.25         # primary alpha engine
sector_rotation: 0.15      # MOEX sector momentum

# strategies/presets/ru_energy.yaml -- add:
dividend_gap: 0.25         # high-yield energy dividends
pairs: 0.10                # TATN/TATNP, SNGS/SNGSP arb

# strategies/presets/ru_finance.yaml -- add:
dividend_gap: 0.20         # SBER dividend gap
cbr_calendar: 0.15         # increase weight (CBR-sensitive)
pairs: 0.10                # SBER/SBERP arb
```

### MOEX Sector Index Constants

```python
MOEX_SECTOR_INDICES: dict[str, str] = {
    "energy": "MOEXOG",
    "finance": "MOEXFN",
    "metals": "MOEXMM",
    "consumer": "MOEXCN",
    "telecom": "MOEXTL",
    "chemicals": "MOEXCH",
    "utilities": "MOEXEU",
    "tech": "MOEXIT",
}
```

---

## Installation

```bash
# No new packages to install. Existing stack covers all v2.0 needs.
uv sync  # ensure current deps are up to date
```

---

## Confidence Assessment

| Area | Confidence | Rationale |
|------|------------|-----------|
| Zero new pip deps | HIGH | Verified every feature against pyproject.toml and existing modules |
| CBR data availability | HIGH | CBRFetcher already operational with SOAP/REST XML |
| PairsStrategy reuse for pref arb | HIGH | Code reviewed, Kalman + cointegration confirmed working |
| MOEX ISS sector indices | MEDIUM | ISS API confirmed for index candles, but specific sector tickers (MOEXOG, MOEXFN) need live validation |
| ML macro features feasibility | HIGH | All data sources already in MoexMarketData schema |
| Dividend event expansion | MEDIUM | T-Invest supports get_dividends(), but batch perf for 50+ symbols untested |
| Portfolio allocator | HIGH | All building blocks exist (CircuitBreaker, regime, MacroSnapshot) |

---

## Sources

- Codebase review: `pyproject.toml`, all modules in `src/finalayze/` listed above
- MOEX ISS API reference: `https://iss.moex.com/iss/reference/` (sector index availability)
- CBR API: `https://www.cbr.ru/scripts/XML_dynamic.asp` (already used by CBRFetcher)
- T-Invest API: `t-tech-investments` package proto definitions
- MOEX sector indices: `https://www.moex.com/en/indices` (index list)

---
*Stack research for: v2.0 MOEX Profitability -- dividend gap, CBR regime, sector rotation, pref arb, ML macro*
*Researched: 2026-03-20*

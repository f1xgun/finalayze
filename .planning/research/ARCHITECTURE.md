# Architecture: MOEX Profitability Integration

**Domain:** MOEX-native strategies, macro regime gating, portfolio-level allocation
**Researched:** 2026-03-20
**Confidence:** HIGH -- derived from direct inspection of existing codebase (dividend_gap.py, cbr_calendar.py, rub_oil_regime.py, combiner.py, position_sizing_pipeline.py, backtest engine, schemas, and all relevant modules)

## Recommended Architecture

The v2.0 MOEX profitability features integrate into the existing 7-layer architecture without violating layer boundaries. No new layers needed. The design principle is: **new strategies extend BaseStrategy (L4), new regime providers implement RegimeProvider protocol (L4), new data fetchers go to L2, and portfolio allocation is a new L4/L5 orchestrator**.

### Integration Overview

```
Layer 2: Data
  CBRFetcher (EXISTING) ── key rates, FX
  MoexISSFetcher (EXISTING) ── IMOEX, turnover
  YFinanceFetcher (EXISTING) ── Brent BZ=F
  TinkoffFetcher (EXISTING) ── dividends, candles
  + BrentDataProvider (NEW) ── wraps yfinance Brent with caching
  + DividendCalendarLoader (NEW) ── bulk load ex-div dates from Tinkoff
  + PreferredShareMapper (NEW) ── SBER↔SBERP FIGI mapping

Layer 4: Strategy / Risk
  STRATEGIES:
    DividendGapStrategy (EXISTING) ── needs calendar wiring
    CBRStrategyWrapper (EXISTING) ── needs combiner registration
    + SectorRotationStrategy (NEW) ── Brent-gated energy, CBR-gated finance
    + PreferredShareArbStrategy (NEW) ── common/pref spread convergence
  RISK:
    RubOilRegimeSignal (EXISTING) ── needs wiring into backtest
    + CBRRegimeGate (NEW) ── hiking=restrict equity, cutting=allow
    + BrentConditionGate (NEW) ── Brent < $60 blocks energy longs
    + PortfolioAllocator (NEW) ── 40% OFZ / 60% equity capital split
    + RUBCrisisBrake (NEW) ── USDRUB spike blocks all equity

Layer 5: Execution / Backtest
  BacktestEngine (MODIFY) ── pass macro context per bar
  + PortfolioBacktestOrchestrator (NEW) ── run bond + equity backtests, merge
```

### Component Boundaries

| Component | Layer | Responsibility | Communicates With |
|-----------|-------|---------------|-------------------|
| `DividendCalendarLoader` | L2 | Bulk-load ex-div dates from Tinkoff API for all ru_* symbols | TinkoffFetcher, DividendGapStrategy |
| `BrentDataProvider` | L2 | Provide Brent candles (yfinance BZ=F) with caching | YFinanceFetcher, SectorRotationStrategy |
| `PreferredShareMapper` | L2 | Map common↔preferred share FIGIs (SBER/SBERP etc.) | InstrumentRegistry, PreferredShareArbStrategy |
| `CBRRegimeGate` | L4 | CBR hiking/cutting cycle → regime state for equity gating | MacroContextProvider, PositionSizingPipeline |
| `BrentConditionGate` | L4 | Brent price threshold → energy sector position gate | BrentDataProvider, SectorRotationStrategy |
| `SectorRotationStrategy` | L4 | MOEX sector rotation: energy=Brent-gated, finance=CBR-gated | BrentConditionGate, CBRRegimeGate, StrategyCombiner |
| `PreferredShareArbStrategy` | L4 | Trade common/pref spread when it exceeds historical norms | PreferredShareMapper, StrategyCombiner |
| `PortfolioAllocator` | L4 | Split capital 40% OFZ / 60% equity, enforce caps | PositionSizingPipeline, BacktestEngine |
| `RUBCrisisBrake` | L4 | USDRUB > threshold → block all equity, shift to OFZ | RubOilRegimeSignal, PortfolioAllocator |
| `PortfolioBacktestOrchestrator` | L5 | Run bond + equity backtests separately, merge PnL | BacktestEngine, BondBacktestEngine, PerformanceAnalyzer |

### Data Flow

#### 1. Macro Context Per Bar (new flow)

The backtest engine currently receives a `RegimeProvider` and `MarketContext` at init time. For v2.0, strategies need **per-bar macro data** (CBR rate, Brent price, USDRUB) that changes during the backtest window.

```
MacroContextProvider (EXISTING, L2)
  + get_snapshot(as_of: date) -> MacroSnapshot
    ├── key_rate: from CBR meeting calendar
    ├── ruonia_7d_avg: proxy from key_rate
    ├── cpi_yoy: from Rosstat publication dates
    ├── brent_close: NEW - from Brent candle series
    └── usdrub: from CBR FX rates

BacktestEngine._process_bar():
  macro = self._macro_provider.get_snapshot(candle.timestamp.date())
  signal = strategy.generate_signal(
      symbol, candles, segment_id,
      regime_state=regime_state,
      macro_snapshot=macro,         # NEW kwarg
  )
```

**Key design decision:** Pass `MacroSnapshot` via `**kwargs` to `generate_signal()` rather than changing the BaseStrategy interface. The existing `**kwargs` pattern (used by `regime_state` already) avoids breaking all 8+ strategy implementations.

#### 2. Dividend Gap Calendar Wiring

```
BEFORE (current):
  DividendGapStrategy._calendar is empty dict
  → 0 trades in backtest

AFTER:
  DividendCalendarLoader.load_all(symbols, start, end)
    → TinkoffFetcher.get_dividends(figi) for each symbol
    → Returns dict[str, list[DividendEntry]]

  BacktestEngine/run_iteration.py:
    calendar_data = loader.load_all(symbols, start, end)
    dividend_strategy.populate_calendar(calendar_data)
    → DividendGapStrategy._calendar populated before run()
```

**Integration point:** `run_iteration.py` (L5 script) populates the calendar before `engine.run()`. This keeps the strategy stateless w.r.t. data fetching (L4 never imports L2).

#### 3. Sector Rotation Signal Flow

```
SectorRotationStrategy.generate_signal(symbol, candles, segment_id, **kwargs):
  1. Determine symbol's sector (energy, finance, other)
  2. If energy sector:
     brent = kwargs.get("macro_snapshot").brent_close
     if brent < BRENT_FLOOR → return None (block)
     if brent > BRENT_MOMENTUM_THRESHOLD → boost confidence
  3. If finance sector:
     cbr_decision = kwargs.get("macro_snapshot").last_cbr_decision
     if decision == "hike" → return None (block new longs)
     if decision == "cut" → boost confidence
  4. Apply relative IMOEX momentum (sector vs index)
  5. Return Signal with direction + adjusted confidence
```

**Combiner integration:** SectorRotationStrategy extends BaseStrategy, registered in combiner alongside existing strategies. Preset YAML weights control its influence per segment.

#### 4. Preferred Share Arbitrage Flow

```
PreferredShareArbStrategy.generate_signal(symbol, candles, segment_id, **kwargs):
  1. Check if symbol is in a common/pref pair (SBER/SBERP, TATN/TATNP, SNGS/SNGSP)
  2. Get counterpart candles from kwargs["pair_candles"]
  3. Compute spread = common_price / pref_price
  4. Compare to rolling mean spread (60-day window)
  5. If spread > mean + 1.5*std → BUY pref, SELL common
  6. If spread < mean - 1.5*std → BUY common, SELL pref
  7. Exit when spread reverts to mean
```

**Data requirement:** The engine must provide both common and preferred candles simultaneously. This requires a multi-symbol context, which the current per-symbol backtest loop does not support natively.

**Solution:** Run PreferredShareArbStrategy as a **pairs-style strategy** (like the existing `pairs.py`). The strategy receives the primary symbol's candles and fetches the counterpart from a pre-loaded dict passed via `set_market_context()`.

#### 5. Portfolio-Level Allocation Flow

```
PortfolioAllocator:
  total_capital = 2_000_000 RUB

  OFZ allocation = 40% = 800K
    → BondBacktestEngine runs OFZ carry + CBR event strategies
    → PnL from bond layer

  Equity allocation = 60% = 1_200K
    → BacktestEngine runs equity strategies per segment
    → Position sizing capped at equity_allocation, not total_capital

  RUB Crisis Brake:
    if usdrub_daily_change > 5% OR rub_oil_corr < 0.1:
      equity_allocation = max(20%, equity_allocation * 0.5)
      ofz_allocation = total_capital - equity_allocation
      → Shift capital to OFZ carry

  Rebalance:
    Monthly: check actual vs target allocation
    If drift > 5%: rebalance

PortfolioBacktestOrchestrator:
  1. Split initial_cash: 40% → bond engine, 60% → equity engine
  2. Run both engines independently
  3. Monthly rebalance: transfer PnL drift back to target weights
  4. Merge PortfolioState timeseries
  5. Compute aggregate Sharpe, DD, PF across combined portfolio
```

## Patterns to Follow

### Pattern 1: Macro-Gated Strategy (Strategy receives macro via kwargs)

**What:** Strategies that depend on macro conditions (Brent, CBR, USDRUB) receive a `MacroSnapshot` through the existing `**kwargs` mechanism in `generate_signal()`.

**When:** Any strategy that needs CBR rate, Brent price, or FX data for signal generation.

**Why:** Avoids changing BaseStrategy ABC signature. Already precedented by `regime_state` kwarg in DividendGapStrategy.

```python
class SectorRotationStrategy(BaseStrategy):
    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
        **kwargs: object,
    ) -> Signal | None:
        macro: MacroSnapshot | None = kwargs.get("macro_snapshot")
        if macro is None:
            return None  # Cannot operate without macro context

        sector = self._get_sector(symbol)
        if sector == "energy" and macro.brent_close is not None:
            if float(macro.brent_close) < self._brent_floor:
                return None  # Block energy longs when Brent is low
        # ... signal logic
```

### Pattern 2: Regime Gate as Sizing Step (PositionSizingStep protocol)

**What:** New macro conditions (CBR regime, Brent gate) plug into the position sizing pipeline as additional `PositionSizingStep` implementations.

**When:** Conditions that should scale position size rather than block signals entirely.

**Why:** Follows existing pipeline architecture. CBR hiking cycle should reduce equity sizing, not eliminate all signals.

```python
class CBRRegimeStep:
    """Scale equity positions based on CBR rate cycle."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        if context.cbr_regime == "hiking":
            return (size * Decimal("0.6")).quantize(_FOUR_DP)
        if context.cbr_regime == "cutting":
            return (size * Decimal("1.2")).quantize(_FOUR_DP)  # more aggressive
        return size  # "hold" = neutral
```

**Pipeline order:** Kelly -> VolTarget -> Regime -> **CBRRegime** -> **BrentGate** -> MetaLabel -> HardCaps

### Pattern 3: Calendar-Driven Strategy Initialization

**What:** Strategies that depend on calendar data (dividend dates, CBR meetings) are populated before the backtest run, not during.

**When:** DividendGapStrategy, CBRStrategyWrapper.

**Why:** Keeps strategy logic (L4) separate from data fetching (L2). The orchestration script (L5+) wires them together.

```python
# In run_iteration.py (orchestration layer):
dividend_loader = DividendCalendarLoader(tinkoff_fetcher)
calendar_data = dividend_loader.load_all(symbols, start, end)

dividend_strategy = DividendGapStrategy(min_gap_pct=3.0, max_hold_bars=60)
for symbol, entries in calendar_data.items():
    for entry in entries:
        dividend_strategy.add_dividend(symbol, entry)
```

### Pattern 4: Multi-Asset Backtest via Orchestrator

**What:** Run bond and equity backtests separately with independent engines, then merge results at the portfolio level.

**When:** Portfolio-level allocation (40% OFZ + 60% equity).

**Why:** Bond and equity backtests have fundamentally different engines (BondBacktestEngine vs BacktestEngine). Merging at the PnL level avoids coupling them.

```python
class PortfolioBacktestOrchestrator:
    def run(self, equity_candles, bond_candles, total_capital):
        bond_capital = total_capital * Decimal("0.40")
        equity_capital = total_capital * Decimal("0.60")

        bond_result = self._bond_engine.run(bond_candles, bond_capital)
        equity_result = self._equity_engine.run(equity_candles, equity_capital)

        return self._merge_results(bond_result, equity_result)
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Strategy Fetching Its Own Data
**What:** A strategy class importing from L2 to fetch Brent prices or CBR rates directly.
**Why bad:** Violates layer boundaries (L4 cannot import L2 for data fetching). Creates hidden I/O in signal generation. Makes backtesting non-deterministic.
**Instead:** Pass all external data via `MacroSnapshot` kwarg or `set_market_context()`.

### Anti-Pattern 2: Single Monolithic Backtest Engine for Bond+Equity
**What:** Modifying BacktestEngine to handle both bond and equity logic internally.
**Why bad:** Bond and equity have different signal interfaces (BondCarryStrategy does not extend BaseStrategy), different risk parameters, different cost models. Coupling them creates a god object.
**Instead:** Use PortfolioBacktestOrchestrator to run separate engines and merge results.

### Anti-Pattern 3: Hardcoded Sector Classification
**What:** `if symbol in ["LKOH", "ROSN", "TATN"]: sector = "energy"` scattered through strategy code.
**Why bad:** Fragile, duplicated, breaks when universe changes.
**Instead:** Create a `SectorClassifier` in L2 that maps symbols to sectors using InstrumentRegistry or a static YAML. Strategies receive sector as metadata.

### Anti-Pattern 4: Look-Ahead in Macro Data
**What:** Using today's CBR rate or Brent close to make decisions about yesterday's bar.
**Why bad:** Overfits backtests, creates phantom alpha that vanishes in live trading.
**Instead:** MacroContextProvider already enforces point-in-time access. All new macro features (Brent, USDRUB) must respect publication dates. Use `as_of` date pattern.

### Anti-Pattern 5: Portfolio Rebalancing Inside Position Sizing
**What:** Putting OFZ/equity capital split logic inside `PositionSizingPipeline`.
**Why bad:** Sizing pipeline operates per-position. Portfolio allocation is a cross-asset concern that must see total equity and all positions simultaneously.
**Instead:** PortfolioAllocator sits above the sizing pipeline, setting `equity` in `SizingContext` to the allocated capital (not total portfolio).

## New vs Modified Components

### New Components (must be created)

| Component | File | Layer | Complexity |
|-----------|------|-------|------------|
| `DividendCalendarLoader` | `data/dividend_calendar.py` | L2 | Low |
| `BrentDataProvider` | Extension of MarketDataLoader | L2 | Low |
| `PreferredShareMapper` | `data/preferred_shares.py` | L2 | Low |
| `SectorClassifier` | `data/sector_classifier.py` | L2 | Low |
| `SectorRotationStrategy` | `strategies/sector_rotation.py` | L4 | Medium |
| `PreferredShareArbStrategy` | `strategies/preferred_arb.py` | L4 | Medium |
| `CBRRegimeStep` | Addition to `risk/position_sizing_pipeline.py` | L4 | Low |
| `BrentGateStep` | Addition to `risk/position_sizing_pipeline.py` | L4 | Low |
| `PortfolioAllocator` | `risk/portfolio_allocator.py` | L4 | Medium |
| `RUBCrisisBrake` | `risk/rub_crisis_brake.py` | L4 | Low |
| `PortfolioBacktestOrchestrator` | `backtest/portfolio_orchestrator.py` | L5 | High |

### Modified Components (extend existing)

| Component | File | Change | Complexity |
|-----------|------|--------|------------|
| `MacroSnapshot` | `data/fetchers/cbr.py` | Add `brent_close`, `usdrub_daily_change` fields | Low |
| `MacroContextProvider` | `data/fetchers/cbr.py` | Populate Brent and USDRUB from candle series | Low |
| `SizingContext` | `risk/position_sizing_pipeline.py` | Add `cbr_regime`, `brent_price` fields | Low |
| `BacktestEngine._process_bar()` | `backtest/engine.py` | Pass `macro_snapshot` to strategy kwargs | Low |
| `StrategyCombiner` | `strategies/combiner.py` | Register new strategies, pass macro kwargs | Low |
| YAML presets (`ru_*.yaml`) | `strategies/presets/` | Add weights for `sector_rotation`, `preferred_arb`, `dividend_gap` | Low |
| `run_iteration.py` | `scripts/` | Wire DividendCalendarLoader, macro provider | Medium |
| `MarketDataLoader._load_moex()` | `data/loader.py` | Include Brent candles in MarketContext | Low |

## Suggested Build Order (Dependency Chain)

The build order must respect both layer dependencies and feature dependencies:

```
Phase 1: Data Foundation (L2)
  1.1 DividendCalendarLoader — needed by DividendGapStrategy wiring
  1.2 SectorClassifier — needed by SectorRotationStrategy
  1.3 PreferredShareMapper — needed by PreferredShareArbStrategy
  1.4 MacroSnapshot extensions (Brent, USDRUB) — needed by all macro-gated strategies

Phase 2: Strategy Implementation (L4)
  2.1 Wire DividendGapStrategy calendar (EXISTING strategy, just needs data)
  2.2 CBRRegimeStep + BrentGateStep in sizing pipeline
  2.3 SectorRotationStrategy (depends on 1.2, 1.4)
  2.4 PreferredShareArbStrategy (depends on 1.3)
  2.5 Wire CBRStrategyWrapper into combiner (EXISTING strategy)
  2.6 Wire RubOilRegimeSignal into backtest regime provider (EXISTING)

Phase 3: Portfolio Allocation (L4-L5)
  3.1 PortfolioAllocator (40/60 split)
  3.2 RUBCrisisBrake
  3.3 PortfolioBacktestOrchestrator

Phase 4: Integration & Backtest (L5)
  4.1 Update run_iteration.py to wire all new components
  4.2 Update YAML presets with new strategy weights
  4.3 Run walk-forward backtests per segment
  4.4 Run combined portfolio backtest
```

**Phase ordering rationale:**
- Phase 1 first because all strategies depend on data providers.
- Phase 2 next because strategies can be individually backtested.
- Phase 3 after strategies work individually, because portfolio allocation composes them.
- Phase 4 last because integration requires all components to exist.

**Within Phase 2:** DividendGapStrategy wiring (2.1) first because it is the highest expected-alpha feature (documented 70%+ gap closure) and the strategy already exists -- just needs data. Sizing pipeline steps (2.2) next because they affect all equity strategies. SectorRotation (2.3) before PreferredArb (2.4) because sector rotation applies broadly while arb is niche.

## Layer Violation Analysis

All proposed components respect the 7-layer dependency rules:

| New Component | Layer | Imports From | Violation? |
|---------------|-------|-------------|------------|
| DividendCalendarLoader | L2 | L0 (schemas), L1 (config) | No |
| SectorClassifier | L2 | L0 (schemas) | No |
| SectorRotationStrategy | L4 | L0 (schemas), L4 (base) | No |
| PreferredShareArbStrategy | L4 | L0 (schemas), L4 (base) | No |
| CBRRegimeStep | L4 | L0 (Decimal), L4 (SizingContext) | No |
| PortfolioAllocator | L4 | L0 (Decimal), L4 (SizingContext) | No |
| PortfolioBacktestOrchestrator | L5 | L0-L5 | No |

**No upward imports proposed.** All macro data flows downward from L2 to L4 via `MacroSnapshot` kwarg.

## Scalability Considerations

| Concern | Current (5 symbols) | At 30 symbols | At 150+ dividend events |
|---------|---------------------|---------------|------------------------|
| Dividend calendar loading | N/A (empty) | 30 Tinkoff API calls | ~150 events, <1MB RAM |
| Brent data | Already fetched via yfinance | Same (single series) | Same |
| Preferred pair backtesting | N/A | 3 pairs = 6 symbols | 3 pairs, negligible |
| Portfolio orchestrator | N/A | 2 engines (bond+equity) | 2 engines, linear time |
| MacroSnapshot per bar | ~1K bars | ~1K bars | O(1) lookup per bar |

The dominant cost is the Tinkoff API calls for dividend data. With 30 symbols and 3 years of history, this is ~90 API calls total -- well within rate limits.

## Sources

- Direct codebase inspection: `dividend_gap.py`, `cbr_calendar.py`, `cbr_event.py`, `cbr_strategy_wrapper.py`, `rub_oil_regime.py`, `regime.py`, `position_sizing_pipeline.py`, `combiner.py`, `engine.py`, `loader.py`, `schemas.py`, `bond_carry.py`
- Existing CBR meeting calendar: `data/fetchers/cbr.py` (2022-2026, 40+ meetings)
- Existing CPI publication dates: `data/fetchers/cbr.py` (2024-2025, Rosstat lag-aware)
- MOEX segment presets: `strategies/presets/ru_*.yaml` (6 MOEX equity + 2 OFZ segments)
- PortfolioLayer enum: `core/schemas.py` (CORE/STRATEGIC/TACTICAL/SHORT)
- LayerConfig defaults: 45% CORE + 27.5% STRATEGIC + 17.5% TACTICAL + 10% SHORT

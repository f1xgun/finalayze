# Stock Segment System Design

This document describes the segment system as implemented in Phase 1.
Segment definitions live in `config/segments.py`; strategy presets are in
`src/finalayze/strategies/presets/`.

## 1. What Are Segments?

A **segment** is a logical grouping of stocks that share:
- The same market (US or MOEX) and broker (Alpaca or Tinkoff)
- The same currency
- A common set of active trading strategies with shared parameters
- Shared news language preferences and portfolio allocation limits

Segments allow the system to apply differentiated strategy configs to different
stock groups (e.g., tech stocks need different RSI thresholds than ETFs) without
creating per-symbol configuration explosion.

## 2. SegmentConfig Fields

Defined in `config/segments.py` as a frozen dataclass:

```python
@dataclass(frozen=True)
class SegmentConfig:
    segment_id: str                              # unique key, e.g. "us_tech"
    market: str                                  # "us" | "moex"
    broker: str                                  # "alpaca" | "tinkoff"
    currency: str                                # "USD" | "RUB"
    symbols: list[str]                           # ticker symbols in this segment
    active_strategies: list[str]                 # strategy names to run
    strategy_params: dict[str, dict[str, object]]  # inline param overrides
    ml_model_id: str | None                      # segment-specific ML model (Phase 2+)
    news_sources: list[str]                      # news provider IDs
    news_languages: list[str]                    # e.g. ["en"] or ["ru", "en"]
    max_allocation_pct: float                    # max % of portfolio for segment
    trading_hours: str                           # human-readable schedule string
```

| Field | Purpose |
|---|---|
| `segment_id` | Primary key used everywhere: YAML preset filenames, DB lookups, signal routing |
| `market` / `broker` | Determines which exchange and API to use |
| `currency` | All Decimal arithmetic for this segment uses this currency |
| `symbols` | Tickers seeded into the instrument registry at startup |
| `active_strategies` | Logical filter; actual enabling is controlled by YAML preset |
| `strategy_params` | Optional inline overrides (currently unused; presets take precedence) |
| `ml_model_id` | Reserved for per-segment XGBoost/LightGBM model (Phase 2) |
| `news_languages` | Filters news ingestion to relevant languages |
| `max_allocation_pct` | Risk constraint enforced by `pre_trade_check.py` |
| `trading_hours` | Informational; enforcement is via `MarketRegistry.is_market_open()` |

## 3. Defined Segments

Eight segments are defined in `DEFAULT_SEGMENTS`:

| Segment ID | Market | Broker | Currency | Symbols | Max Alloc | Trading Hours (UTC) |
|---|---|---|---|---|---|---|
| `us_tech` | us | alpaca | USD | AAPL, MSFT, GOOGL, NVDA, META, AMZN | 30% | 14:30-21:00 |
| `us_healthcare` | us | alpaca | USD | JNJ, PFE, UNH, ABBV, MRK | 25% | 14:30-21:00 |
| `us_finance` | us | alpaca | USD | JPM, BAC, GS, MS, WFC | 25% | 14:30-21:00 |
| `us_broad` | us | alpaca | USD | SPY, QQQ, DIA | 30% | 14:30-21:00 |
| `ru_blue_chips` | moex | tinkoff | RUB | SBER, GAZP, LKOH, GMKN | 30% | 07:00-15:40 |
| `ru_energy` | moex | tinkoff | RUB | ROSN, TATN, NVTK | 25% | 07:00-15:40 |
| `ru_tech` | moex | tinkoff | RUB | YNDX, OZON, VKCO | 20% | 07:00-15:40 |
| `ru_finance` | moex | tinkoff | RUB | SBER, VTBR, TCSG | 25% | 07:00-15:40 |

### Segment Characteristics

**us_tech** — Large-cap US technology. High growth, high volatility. Momentum
strategy weighted 0.4, event-driven 0.4 (earnings, product launches, FDA for
portfolio holdings). Tighter min_confidence thresholds (0.6+).

**us_healthcare** — Pharma and health insurance. Event-driven dominant (weight
0.5) due to FDA approval events and clinical trial results. Tighter RSI bands
(oversold: 35, overbought: 65) to reduce noise.

**us_finance** — Banks and investment firms. Mean reversion dominant (weight
0.4); financial stocks tend to range-bound behavior between earnings. Sensitive
to interest rate and macro events.

**us_broad** — US broad market ETFs (SPY, QQQ, DIA). Equal weight between
momentum and mean reversion. Lower volatility → lower min_confidence thresholds
(0.55/0.6). No event-driven strategy.

**ru_blue_chips** — Russian blue chips (SBER, GAZP, LKOH, GMKN). Event-driven
dominant (weight 0.5) for geopolitical, sanctions, commodity, and CBR events.
Wider Bollinger Bands (bb_std_dev: 2.5), tighter RSI (oversold: 25, overbought: 75).
News ingested in both Russian and English.

**ru_energy** — Russian oil and gas (ROSN, TATN, NVTK). Similar to ru_blue_chips
but event types are oil-focused: opec, oil_price, commodity_price.

**ru_tech** — Russian internet/tech (YNDX, OZON, VKCO). Balanced: momentum 0.4,
mean_reversion 0.35, event_driven 0.25. Russian-language news only.

**ru_finance** — Russian banks (SBER, VTBR, TCSG). Mean reversion dominant (0.4);
event types include cbr_rate, regulatory, sanctions.

## 4. How Segments Map to Strategies via YAML Presets

Each segment has a corresponding YAML file at
`src/finalayze/strategies/presets/<segment_id>.yaml`. The preset defines:
- Which strategies are enabled
- The weight each strategy gets in the combiner
- The per-strategy parameters

Example for `us_broad`:
```yaml
segment_id: us_broad
strategies:
  momentum:
    enabled: true
    weight: 0.5
    params:
      rsi_period: 14
      rsi_oversold: 30
      rsi_overbought: 70
      macd_fast: 12
      macd_slow: 26
      min_confidence: 0.55
  mean_reversion:
    enabled: true
    weight: 0.5
    params:
      bb_period: 20
      bb_std_dev: 2.0
      min_confidence: 0.6
```

The `StrategyCombiner` loads this YAML at runtime via `_load_config(segment_id)`.
Individual strategies (`MomentumStrategy`, `MeanReversionStrategy`) call
`get_parameters(segment_id)` which reads the same YAML.

## 5. Loading at Startup

`DEFAULT_SEGMENTS` is a module-level list in `config/segments.py`:

```python
DEFAULT_SEGMENTS: list[SegmentConfig] = [
    SegmentConfig(segment_id="us_tech", ...),
    SegmentConfig(segment_id="us_healthcare", ...),
    # ... 8 segments total
]
```

These are available for import anywhere in the codebase. The backtest engine
and strategy combiner both receive segment configs as constructor arguments,
so they can be substituted or extended in tests without modifying the defaults.

In a future phase, segments will be stored in the database (the `segments` table
created by Alembic) and `DEFAULT_SEGMENTS` will serve as the seed data, with
DB rows overriding or extending the defaults at startup.

## 6. Relationship: Markets, Segments, Instruments

```
MarketDefinition (us / moex)
  └── SegmentConfig (us_tech, us_broad, ...)
        └── Instrument (AAPL, MSFT, ...)
```

- `MarketDefinition` is managed by `MarketRegistry` (`src/finalayze/markets/registry.py`).
  It carries timezone, open/close times, and currency.
- `SegmentConfig` is a logical grouping within a market. Multiple segments can
  belong to the same market (all `us_*` segments belong to market `"us"`).
- `Instrument` is an individual tradeable security. It belongs to exactly one
  market via `market_id`. The `InstrumentRegistry` maps `(symbol, market_id)`
  to `Instrument` metadata (name, type, FIGI for MOEX, lot size, currency).

The `segment_id` field on `Signal` and `Candle` schemas links runtime data
back to the segment that generated or owns it.

## 7. How to Add a New Segment

1. Add a `SegmentConfig` entry to `DEFAULT_SEGMENTS` in `config/segments.py`:

```python
SegmentConfig(
    segment_id="us_energy",
    market="us",
    broker="alpaca",
    currency="USD",
    symbols=["XOM", "CVX", "COP"],
    active_strategies=["momentum", "event_driven"],
    news_languages=["en"],
    max_allocation_pct=0.20,
    trading_hours="14:30-21:00 UTC",
),
```

2. Create `src/finalayze/strategies/presets/us_energy.yaml` with strategy
   weights and parameters for the new segment.

3. Register any new instruments in `src/finalayze/markets/instruments.py` by
   adding them to `DEFAULT_US_INSTRUMENTS` (or the MOEX equivalent in Phase 2).

4. Add unit tests verifying the new segment's strategy signals and combiner
   weights in `tests/unit/`.

## Status

| Segment | Phase 1 | Notes |
|---|---|---|
| us_tech | Operational | 6 symbols, momentum + mean_reversion presets |
| us_healthcare | Operational | 5 symbols, event_driven placeholder in preset |
| us_finance | Operational | 5 symbols |
| us_broad | Operational | 3 ETFs, balanced preset |
| ru_blue_chips | Defined | MOEX; Tinkoff integration pending (Phase 2) |
| ru_energy | Defined | MOEX; Tinkoff integration pending (Phase 2) |
| ru_tech | Defined | MOEX; Tinkoff integration pending (Phase 2) |
| ru_finance | Defined | MOEX; Tinkoff integration pending (Phase 2) |

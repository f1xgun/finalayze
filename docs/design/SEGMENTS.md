# Stock Segment System

## Overview

Stocks are grouped into **segments** -- logical categories that determine which
strategies, parameters, and ML models apply. Each segment belongs to exactly one market.

## Segment Registry

| Segment ID | Market | Description | Primary Strategy | Key Characteristics |
|-----------|--------|-------------|-----------------|---------------------|
| `us_tech` | US | Technology | Momentum + Event-driven | High volatility, news-sensitive |
| `us_healthcare` | US | Healthcare/Pharma | Event-driven + Mean reversion | FDA events dominant |
| `us_finance` | US | Financial sector | Mean reversion + Macro | Interest rate sensitive |
| `us_broad` | US | ETFs & broad market | Momentum + Mean reversion | Lower volatility |
| `ru_blue_chips` | MOEX | Russian blue chips | Momentum + Event-driven | Commodity-linked, geopolitical |
| `ru_energy` | MOEX | Russian energy | Commodity-driven + Event | Oil/gas, OPEC, sanctions |
| `ru_tech` | MOEX | Russian tech | Momentum + Mean reversion | Local market dynamics |
| `ru_finance` | MOEX | Russian banks/finance | Mean reversion + Macro | CBR rate driven |

## Configuration

Each segment is defined as a `SegmentConfig` dataclass (see `config/segments.py`)
with the following fields:

- `segment_id`, `market`, `broker`, `currency`
- `symbols` -- tickers in this segment
- `active_strategies` -- strategy names to run
- `strategy_params` -- per-strategy parameter overrides
- `ml_model_id` -- segment-specific ML model
- `news_sources`, `news_languages`
- `max_allocation_pct` -- max % of portfolio for this segment
- `trading_hours`

## Strategy Parameter Presets

Each segment has a YAML preset file in `src/finalayze/strategies/presets/`:
- `us_tech.yaml`, `us_healthcare.yaml`, `us_finance.yaml`, `us_broad.yaml`
- `ru_blue_chips.yaml`, `ru_energy.yaml`, `ru_tech.yaml`, `ru_finance.yaml`

## News Scope Routing

News is classified by geographic scope and routed to affected segments:
- **Global** (e.g., Fed rate decision) -> all segments
- **US** (e.g., FDA approval) -> `us_*` segments
- **Russia** (e.g., CBR rate decision) -> `ru_*` segments
- **Sector** (e.g., chip shortage) -> matching sector segments across markets

## Status

**Phase 0:** Segment definitions created in `config/segments.py`.
**Phase 1:** Segment framework + US segments operational.
**Phase 2:** MOEX segments added.

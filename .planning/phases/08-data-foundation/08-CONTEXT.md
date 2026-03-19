# Phase 8: Data Foundation - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase fixes the three data/parameter problems that invalidate all MOEX backtests: US-calibrated vol targets crushing positions, toxic symbols contaminating results, incomplete dividend calendar with look-ahead bias, and 2022 structural break distorting vol/ATR calculations. No new strategies — only data quality and parameter corrections.

</domain>

<decisions>
## Implementation Decisions

### Vol Target Calibration
- Set vol_target to 0.40 for all ru_* segments (ru_blue_chips, ru_energy, ru_finance, ru_tech)
- Keep vol_target_enabled: true — prevents over-concentration but now correctly scaled for MOEX vol (35-45% annualized)
- Current values (0.19-0.22) are US-calibrated and crush MOEX positions to 25-42% of intended size

### Universe Cleanup
- Remove GAZP from ru_blue_chips (PF 0.38, structural sanctions decline)
- Remove SNGS, SNGSP from ru_energy (PF 0.19, opaque structure)
- Remove VTBR from ru_finance (sanctioned bank, no recovery path)
- Remove IRAO from ru_utilities segment (PF 0.18, no signal)
- Remove ALRS from ru_metals segment (PF 0.22, EU diamond sanctions)
- Set min_combined_confidence to 0.38 on all ru_* segments (up from 0.15)

### 2022 Structural Break
- Exclude Feb 21 – Apr 1 2022 from vol/ATR calculations (MOEX closed Feb 28 – Mar 24, extreme dislocation)
- Implement as a date range exclusion in the vol/ATR computation, not a regime flag
- Prices during this period remain in OHLCV data for position tracking, but vol/ATR windows skip these bars

### Dividend Calendar Expansion
- Use T-Invest API batch fetch (get_dividends) for all symbols in all ru_* segments, 2020-2025 range
- Add status field to YAML: paid / cancelled / reduced
- Include GAZP 2022 cancelled dividend as status: cancelled (prevents look-ahead bias)
- Target: 150+ events across 20+ symbols
- Save to expanded moex_dividends.yaml with same format as current

### Claude's Discretion
- Exact implementation of vol/ATR date exclusion (windowed skip vs boolean mask)
- YAML structure for expanded dividend calendar (flat list vs nested by year)
- Whether to create a separate script for T-Invest dividend batch fetch or inline in existing fetcher

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/finalayze/strategies/presets/ru_blue_chips.yaml` — vol_target: 0.19 (needs updating to 0.40)
- `src/finalayze/strategies/presets/ru_energy.yaml` — vol_target: 0.22 (needs updating to 0.40)
- `src/finalayze/strategies/presets/ru_finance.yaml` — vol_target: 0.21 (needs updating to 0.40)
- `src/finalayze/strategies/presets/ru_tech.yaml` — vol_target: 0.20 (needs updating to 0.40)
- `src/finalayze/strategies/presets/moex_dividends.yaml` — 42 lines, 43 events across 6 symbols (SBER, GAZP, LKOH, GMKN, ROSN, TATN)
- `config/segments.py` — SegmentConfig with symbols lists for all segments
- `src/finalayze/data/tinkoff_data.py` — TinkoffFetcher with get_dividends capability

### Established Patterns
- YAML presets define per-segment strategy parameters
- SegmentConfig dataclass in config/segments.py defines symbol lists
- Backtest engine computes vol/ATR from candle data in walk-forward windows
- Position sizing pipeline: HalfKelly → VolTargetStep → RegimeStep → HardCapsStep

### Integration Points
- Vol target change: YAML presets only (no code changes needed)
- Universe cleanup: config/segments.py symbol lists
- 2022 break: backtest engine vol/ATR calculation (src/finalayze/backtest/engine.py)
- Dividend calendar: moex_dividends.yaml + potentially a fetch script

</code_context>

<specifics>
## Specific Ideas

- Vol target 0.40 based on MOEX average annualized vol of 35-45%
- Confidence threshold 0.38 matches existing _MIN_EXIT_CONFIDENCE
- Feb 21 – Apr 1 2022 exclusion covers MOEX closure + immediate aftermath
- T-Invest API get_dividends is already used for current 43 events — need to expand coverage

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

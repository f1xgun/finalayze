# Phase 40: MOEX Data Adapter & Macro Features - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire TinkoffFetcher and MOEX macro features into `scripts/auto_ml_research.py` so it can run end-to-end on all four ru_* equity segments (ru_blue_chips, ru_energy, ru_tech, ru_finance) with real MOEX data and macro context.

</domain>

<decisions>
## Implementation Decisions

### TinkoffFetcher Sync Bridge
- Use `_run_async()` self-managed daemon-thread loop — confirmed working for standalone scripts, no nest_asyncio needed
- One shared TinkoffFetcher instance across all MOEX segments — reuse gRPC channel, rate limiter handles throttling
- `sandbox=False` mandatory for historical candle fetches — sandbox endpoint has no historical data
- If `FINALAYZE_TINKOFF_TOKEN` not set, print error and skip MOEX segments gracefully — US segments continue working

### MOEX Segment Symbols
- Read symbols from `config/segments.py` `DEFAULT_SEGMENTS` at runtime — single source of truth, no hardcoded duplication
- Include all 4 ru_* equity segments: ru_blue_chips (SBER, LKOH, GMKN), ru_energy (ROSN, TATN, NVTK, SIBN, TATNP, TRNFP), ru_tech (YDEX, OZON, VKCO, HEAD, POSI), ru_finance (SBER, SBERP, T, CBOM, BSPB, MOEX)
- Exclude bond segments (ru_ofz_pd, ru_ofz_pk) — bonds use QuantLib pipeline, not ML ensemble
- Use IMOEX as MOEX benchmark (equivalent of SPY for US) via MoexISSFetcher

### Macro Feature Wiring
- Fetch CBR key rate and USDRUB via existing CBRFetcher, IMOEX via MoexISSFetcher — asyncio.run() wrapper, single fetch at startup
- Brent crude via yfinance `BZ=F` — only commodity source, not a MOEX ticker, already used in production
- Apply `shift(1)` on all macro series before feature join — look-ahead bias prevention with unit test
- Cache macro data once at script start and reuse across segments — macro context is market-wide

### Claude's Discretion
- Internal helper function naming and organization
- Error message formatting for missing tokens or API failures
- Test fixture structure for look-ahead bias unit test

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TinkoffFetcher` in `src/finalayze/data/fetchers/tinkoff_data.py` — async gRPC, `_run_async()` for sync
- `CBRFetcher` in `src/finalayze/data/fetchers/cbr.py` — key rate, FX rates
- `MoexISSFetcher` in `src/finalayze/data/fetchers/moex_iss.py` — IMOEX index candles
- `MoexMarketData` in `src/finalayze/core/schemas.py` — dataclass for macro context
- `MarketContext` in `src/finalayze/core/schemas.py` — wraps benchmark + VIX + MOEX data
- 10 MOEX macro features in `src/finalayze/ml/features/technical.py` — already compute from MoexMarketData
- `config/segments.py` `DEFAULT_SEGMENTS` — production MOEX symbol lists
- `scripts/train_models.py` — already handles MOEX with `_is_moex_segment()` pattern

### Established Patterns
- `build_triple_barrier_dataset()` accepts `market_context: MarketContext` including `moex_data: MoexMarketData`
- `_MOEX_LOOKBACK_DAYS = 730`, `_MOEX_MAX_FEATURES = 10`, `_MOEX_ATR_UPLIFT = 1.2` already stubbed in auto_ml_research.py
- Feature selection uses `select_features_efficient()` with configurable `max_features`

### Integration Points
- `auto_ml_research.py:_prepare_data()` — needs MOEX branch (currently only calls `_fetch_us_candles`)
- `auto_ml_research.py:build_full_dataset()` — needs `MoexMarketData` passed via `MarketContext`
- `auto_ml_research.py:_SEGMENT_SYMBOLS` — needs ru_* entries
- `auto_ml_research.py:main()` — needs `--segment` choices extended

</code_context>

<specifics>
## Specific Ideas

- Pattern from `train_models.py`: `_is_moex_segment()` checks `segment_id.startswith("ru_")` to branch data loading
- MOEX constants already stubbed in auto_ml_research.py — use them (730 days lookback, 10 max features, 1.2x ATR uplift)
- Research confirmed: all 10 macro features compute from MoexMarketData when it's non-None

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

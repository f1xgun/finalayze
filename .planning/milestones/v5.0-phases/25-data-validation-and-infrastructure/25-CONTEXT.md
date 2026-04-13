# Phase 25: Data Validation and Infrastructure - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire DataNormalizer for candle validation, add staleness detection, fix IMOEX volume semantic, make TinkoffFetcher reuse persistent gRPC channel, cache Brent crude in MarketDataLoader.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — infrastructure phase.
Key constraints from audit:
- DATA-01: Call DataNormalizer.validate() on candles in _process_instrument() before generate_signal()
- DATA-02: Add _is_candle_stale() call with configurable threshold, log warning, skip instrument
- DATA-03: moex_iss.py line 267 should use row[5] (volume) not row[4] (value/turnover)
- INFRA-01: TinkoffFetcher should use persistent AsyncClient + background event loop (like TinkoffBroker pattern)
- INFRA-02: MarketDataLoader._load_moex() should use _cached_fetch() for Brent, not _safe_fetch()

</decisions>

<code_context>
## Existing Code Insights

### Key Files
- `src/finalayze/data/normalizer.py` — DataNormalizer with validate() method (exists but unused)
- `src/finalayze/orchestration/trading_loop.py:242` — _is_candle_stale() defined but never called
- `src/finalayze/data/fetchers/moex_iss.py:267` — volume = row[4] (turnover, should be row[5])
- `src/finalayze/data/fetchers/tinkoff_data.py` — asyncio.run() per call, no persistent channel
- `src/finalayze/data/loader.py` — _safe_fetch() for Brent vs _cached_fetch() for FX/rates
- `src/finalayze/execution/tinkoff_broker.py` — reference pattern for persistent gRPC channel

### Established Patterns
- TinkoffBroker uses _run_async() with persistent background event loop + _loop_init_lock
- CachingFetcher wraps any fetcher with file-based cache
- MarketDataLoader._cached_fetch() uses GenericFileCache with file path pattern

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>

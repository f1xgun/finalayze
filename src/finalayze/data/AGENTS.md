# Data

## Purpose
Market data ingestion, caching, normalization, and orchestration. Provides fetcher adapters for yfinance (US), MOEX ISS, CBR, Tinkoff Invest gRPC, and news sources (Finnhub, NewsAPI, RSS, Telegram).

## Layer
Layer 2 -- Data / Repository. Can import from layers 0-1. Never import from layers 3-6.

## Key Files
- `loader.py` -- MarketDataLoader: orchestrates ambient data (benchmark, VIX, FX, key rates, Brent, turnover) per segment
- `normalizer.py` -- DataNormalizer: validates OHLCV candles, tags with market_id/source
- `cache.py` -- File-based caching for candle data
- `rate_limiter.py` -- Rate limiting for external API calls
- `macro_cache.py` -- Cache for macro-economic data (FX, key rates)
- `moex_calendar.py` -- MOEX trading calendar
- `bond_discovery.py` -- OFZ bond discovery via Tinkoff API
- `fetchers/base.py` -- BaseFetcher ABC (fetch_candles interface)
- `fetchers/yfinance.py` -- US market data via yfinance
- `fetchers/tinkoff_data.py` -- MOEX data via T-Bank gRPC API (must pass `target="invest-public-api.tbank.ru:443"`)
- `fetchers/moex_iss.py` -- MOEX ISS HTTP API (index candles, turnover)
- `fetchers/cbr.py` -- CBR FX rates and key rate
- `fetchers/caching.py` -- CachingFetcher decorator with file cache
- `fetchers/finnhub.py`, `fetchers/newsapi.py`, `fetchers/rss_fetcher.py`, `fetchers/telegram_reader.py` -- News data sources

## Public API
- `MarketDataLoader` -- load MarketContext for a segment (US or MOEX)
- `BaseFetcher` -- abstract interface for all data fetchers
- `DataNormalizer` -- validate and tag candles
- `CachingFetcher` -- transparent file-cache wrapper for any BaseFetcher

## Contracts
- Input: date ranges (start, end), symbol strings, segment config with `.market` attribute
- Output: `list[Candle]`, `MarketContext`, `list[FXRate]`, `list[KeyRateRecord]`
- Invariants: MOEX data MUST use Tinkoff Invest API (yfinance cannot fetch MOEX tickers). `GRPC_DNS_RESOLVER=native` must be set for gRPC. Fetch failures are logged and return empty lists (never crash the caller).

## Testing
- Test location: `tests/unit/test_caching_fetcher.py`, `tests/unit/test_tinkoff_data.py`, `tests/unit/test_tinkoff_persistent_client.py`
- Run: `uv run pytest tests/unit/test_caching_fetcher.py tests/unit/test_tinkoff_data.py -v`

## Common Patterns
- All fetchers inherit from `BaseFetcher` and implement `fetch_candles()`
- Tinkoff client uses `async with AsyncClient(target=...) as services:` context manager pattern
- `MarketDataLoader._safe_fetch()` wraps all external calls with broad exception handling
- File caches use `GenericFileCache` from `_cache_utils.py`

---

## Graph

- **Parent:** [`src/finalayze/AGENTS.md`](../AGENTS.md)
- **Agent owner:** `data-agent` (news pipeline: `news-pipeline-agent`)
- **Layer:** 2
- **Imports from:** `core/`, `config/`, `markets/`
- **Imported by:** `analysis/`, `ml/`, `strategies/`, `risk/`, `execution/`, `backtest/`, `orchestration/`, `api/`
- **Keywords:** `fetcher`, `yfinance`, `tinkoff_data`, `moex_iss`, `cbr`, `finnhub`, `newsapi`, `rss`, `telegram`, `cache`, `normalizer`, `rate_limiter`, `bond_discovery`, `moex_calendar`

# Architecture Research

**Domain:** MOEX autonomous trading with bond support, news analysis, and Telegram alerting
**Researched:** 2026-03-14
**Confidence:** HIGH — derived from direct codebase inspection (367 Python files, 2325+ tests)

## Standard Architecture

### System Overview

The system is a layered trading engine. New components (bonds, news pipeline, Telegram) slot
into existing layers without restructuring. The dependency rule is strictly enforced: imports
flow downward only (Layer 0 → Layer 6). No upward imports permitted.

```
+─────────────────────────────────────────────────────────────────────────────+
│  LAYER 6 — Orchestration / API / Dashboard                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │  TradingLoop     │  │  BondCycleProc.  │  │  FastAPI + Streamlit     │  │
│  │  (APScheduler)   │  │  (APScheduler)   │  │  REST + Prometheus       │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────────────────┘  │
│           │ 3 cycles:           │ 1 cycle:                                   │
│           │ news, strategy,     │ bond (daily)                              │
│           │ daily reset         │                                            │
+───────────┼─────────────────────┼──────────────────────────────────────────+
│  LAYER 5 — Execution                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  BrokerRouter                                                        │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │   │
│  │  │ TinkoffBroker  │  │ AlpacaBroker   │  │ SimulatedBroker        │ │   │
│  │  │ (gRPC live +   │  │ (US markets)   │  │ (backtest / sandbox)   │ │   │
│  │  │  sandbox)      │  │                │  │                        │ │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
+─────────────────────────────────────────────────────────────────────────────+
│  LAYER 4 — Strategy / Risk                                                  │
│  ┌─────────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │  Strategy Engine (stocks)       │  │  Bond Strategy Engine            │ │
│  │  StrategyCombiner               │  │  BondCarryStrategy               │ │
│  │  ADX regime router              │  │  BondDurationRotation            │ │
│  │  5 active strategies            │  │  CBREventStrategy                │ │
│  └─────────────────────────────────┘  │  CBRStrategyWrapper              │ │
│                                        └──────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Risk Pipeline                                                       │   │
│  │  CircuitBreaker  │  LayerCircuitBreaker  │  AggregateBondBreaker    │   │
│  │  PreTradeCheck   │  DV01BudgetStep       │  EqualWeightBondSizer    │   │
│  │  YieldStop       │  HalfKelly sizing     │  PositionSizingPipeline  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
+─────────────────────────────────────────────────────────────────────────────+
│  LAYER 3 — Analysis / ML                                                    │
│  ┌─────────────────────────────┐  ┌────────────────────────────────────┐   │
│  │  News Pipeline              │  │  ML Ensemble                       │   │
│  │  LLMClient (Claude Sonnet)  │  │  XGBoost + LightGBM + CatBoost     │   │
│  │  NewsAnalyzer (EN/RU)       │  │  + meta-learner                    │   │
│  │  EventClassifier            │  │  45 technical features             │   │
│  │  ImpactEstimator            │  │  Feature selection pipeline        │   │
│  └─────────────────────────────┘  └────────────────────────────────────┘   │
+─────────────────────────────────────────────────────────────────────────────+
│  LAYER 2 — Data / Markets                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ TinkoffFetch │  │ CBRFetcher   │  │ MOEXISSFetch │  │ MacroCacheSvc  │ │
│  │ (candles,    │  │ (FX rates,   │  │ (IMOEX index,│  │ (key rate,     │ │
│  │  dividends,  │  │  key rate,   │  │  turnover)   │  │  RUONIA,       │ │
│  │  instruments)│  │  RUONIA)     │  │              │  │  CPI history)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  InstrumentRegistry  (symbol → FIGI, lot_size, bond metadata)       │  │
│  │  MarketSchedule  (MOEX trading hours, MOEX holidays)                │  │
│  │  CurrencyConverter  (RUB/USD, live + cached)                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
+─────────────────────────────────────────────────────────────────────────────+
│  LAYER 1 — Configuration                                                    │
│  Settings (Pydantic BaseSettings)  │  Modes (debug/sandbox/test/real)       │
│  Segments (ru_ofz_pk, ru_ofz_pd, ru_blue_chips, …)                         │
+─────────────────────────────────────────────────────────────────────────────+
│  LAYER 0 — Types & Schemas                                                  │
│  Candle  Signal  TradeResult  PortfolioState  NewsArticle  SentimentResult  │
│  PortfolioLayer  LayerConfig  InstrumentType  SignalDirection                │
│  LayerLedger  (tracks per-layer cash, positions, peak equity)               │
+─────────────────────────────────────────────────────────────────────────────+
```

### Component Responsibilities

| Component | Layer | Responsibility | Communicates With |
|-----------|-------|----------------|-------------------|
| `TradingLoop` | L6 | APScheduler orchestrator. 3 cycles: news (5min), strategy (15min), daily reset | NewsAnalyzer, StrategyCombiner, BrokerRouter, TelegramAlerter |
| `BondCycleProcessor` | L6 | Bond trading cycle: yield stops → strategy signals → DV01 sizing → execute | Bond strategies, DV01BudgetStep, BrokerRouter, TelegramAlerter |
| `TelegramAlerter` | L6 (fire-and-forget) | Sends trade fills, rejections, circuit breaker trips, daily PnL, coupon alerts, CBR events | Telegram Bot API (httpx) |
| `BrokerRouter` | L5 | Routes orders to correct broker by `market_id` | TinkoffBroker, AlpacaBroker |
| `TinkoffBroker` | L5 | gRPC order submission to T-Invest. Supports sandbox + live. Retry policy | T-Invest gRPC API |
| `StrategyCombiner` | L4 | Weighted signal aggregation across 5 strategies. ADX regime gating. Hook pattern | BaseStrategy subclasses, ADX router |
| `BondCarryStrategy` | L4 | OFZ-PK floater carry: maturity ladder + quarterly rebalancing. Macro-aware | MacroCacheService |
| `CBREventStrategy` | L4 | Tactical OFZ-PD trades around CBR rate meetings (entry 2-7d before, exit T+2) | CBR calendar (static data in cbr.py) |
| `BondDurationRotationStrategy` | L4 | Rotates OFZ-PD duration based on yield curve shape | MacroSnapshot |
| `CircuitBreaker` / `LayerCircuitBreaker` | L4 | 3-level per-layer drawdown limits (L1=caution, L2=halt, L3=liquidate) | LayerLedger |
| `AggregateBondBreaker` | L4 | Portfolio-wide bond drawdown halt at 3% combined DD | All LayerLedgers |
| `DV01BudgetStep` | L4 | Sizes fixed-rate bond positions against DV01 risk budget | — |
| `YieldStop` | L4 | Force-exits bonds when YTM spread deteriorates past threshold | — |
| `NewsAnalyzer` | L3 | Calls Claude Sonnet to produce SentimentResult for each NewsArticle. EN/RU prompts | LLMClient |
| `EventClassifier` | L3 | Classifies news into event types (earnings, macro, geopolitical) | LLMClient |
| `ImpactEstimator` | L3 | Estimates price impact magnitude from event type + sentiment | — |
| `ML Ensemble` | L3 | XGBoost + LightGBM + CatBoost + meta-learner. Per-segment models. Conformal calibration | Feature pipeline, MLModelRegistry |
| `TinkoffFetcher` | L2 | Fetches MOEX candles, instruments, dividends via T-Invest gRPC. FIGI-based | T-Invest gRPC API |
| `CBRFetcher` | L2 | Fetches FX rates (XML), key rate (SOAP), RUONIA from cbr.ru | CBR REST/SOAP APIs |
| `MacroCacheService` | L2 | Caches macro snapshot (key rate, RUONIA, CPI). Daily refresh + CBR-day force-refresh | CBRFetcher, CBR calendar |
| `InstrumentRegistry` | L2 | Maps (symbol, market_id) → Instrument (includes bond metadata: face value, maturity, coupon) | — |
| `LayerLedger` | L0 | Per-layer virtual sub-account: cash, positions, drawdown tracking | — |

## Recommended Project Structure

The structure is already established. New components fit into existing modules:

```
src/finalayze/
├── core/                    # L0/L6 — schemas, events, orchestrators
│   ├── schemas.py           # Candle, Signal, PortfolioLayer, LayerConfig, LayerLedger
│   ├── alerts.py            # TelegramAlerter (DONE — all bond/trade/CBR alert methods)
│   ├── trading_loop.py      # TradingLoop (DONE — 3-cycle APScheduler orchestrator)
│   ├── bond_cycle.py        # BondCycleProcessor (DONE — skeleton, sizing/exec stubs)
│   └── layer_ledger.py      # LayerLedger (DONE)
├── config/
│   └── segments.py          # Add: ru_ofz_pk, ru_ofz_pd, ru_blue_chips segment defs
├── data/
│   ├── fetchers/
│   │   ├── tinkoff_data.py  # DONE — candles, dividends, instruments via gRPC
│   │   ├── cbr.py           # DONE — FX rates, key rate, RUONIA, CBR calendar
│   │   └── newsapi.py       # Extend: add T-Invest news endpoint + Telegram channel polling
│   ├── macro_cache.py       # DONE — MacroCacheService with CBR-day refresh
│   └── moex_calendar.py     # DONE — MOEX trading day schedule
├── analysis/
│   ├── llm_client.py        # DONE — Claude Sonnet async wrapper
│   ├── news_analyzer.py     # DONE — EN/RU sentiment analysis
│   ├── event_classifier.py  # DONE — event type classification
│   └── impact_estimator.py  # DONE — price impact estimation
├── strategies/
│   ├── bond_carry.py        # DONE — OFZ-PK floater carry (Core layer)
│   ├── bond_duration_rotation.py  # DONE — OFZ-PD duration rotation (Strategic)
│   ├── cbr_event.py         # DONE — CBR meeting event strategy (Tactical)
│   ├── cbr_strategy_wrapper.py    # DONE — macro-aware wrapper
│   └── presets/
│       └── moex_bonds.yaml  # ADD: bond strategy params, layer allocations
├── risk/
│   ├── dv01_sizing.py       # DONE — DV01BudgetStep + EqualWeightBondSizer
│   ├── yield_stop.py        # DONE — YieldStop for bond exits
│   ├── layer_circuit_breaker.py   # DONE — BondLayerBreaker + AggregateBondBreaker
│   └── bond_equity_correlation.py # DONE — correlation-based regime detection
├── execution/
│   ├── tinkoff_broker.py    # DONE — gRPC order submission, sandbox + live
│   ├── broker_router.py     # DONE — market_id dispatch
│   └── bond_simulated_broker.py   # DONE — backtest simulator for bonds
└── markets/
    └── instruments.py       # DONE — InstrumentRegistry with bond metadata fields
```

### Structure Rationale

- **Bond strategies are NOT subclasses of BaseStrategy.** They use a different interface (accept `key_rate`, `ruonia_7d_avg`, `cpi_yoy`, `last_cbr_decision` kwargs). This is intentional — bonds are fundamentally different assets and sharing the equity strategy interface would force awkward abstractions.
- **BondCycleProcessor lives in `core/`** for import convenience even though it is architecturally L6. All higher-layer imports are deferred inside methods via `TYPE_CHECKING` to prevent import-time circular dependencies.
- **MacroCacheService lives in `data/`** (L2) because it is a data provider — it fetches and caches, but does not analyze or decide.
- **TelegramAlerter lives in `core/`** as a fire-and-forget sink. It imports nothing upward (only uses httpx and standard library at runtime). Errors are always suppressed — alerts must never crash the trading loop.

## Architectural Patterns

### Pattern 1: Scheduled Cycle Decomposition

**What:** The trading loop runs multiple independent cycles on different schedules via APScheduler BackgroundScheduler. Each cycle is fault-isolated — an exception in the news cycle does not stop the strategy cycle.

**When to use:** Any new periodic process (news polling, dividend collection, macro refresh) should be wired as a separate scheduled job, not embedded in the strategy cycle.

**Trade-offs:** Simple to reason about, easy to add new cycles. Concurrent cycle execution possible if cycles overlap — guard with threading.Lock for shared state (e.g., `_sentiment_cache` uses `_sentiment_lock`).

```python
# TradingLoop wires cycles at startup
scheduler.add_job(self._news_cycle, "interval", minutes=5)
scheduler.add_job(self._strategy_cycle, "interval", minutes=15)
scheduler.add_job(self._bond_cycle, "interval", minutes=30)
scheduler.add_job(self._daily_reset, "cron", hour=0, minute=5)
```

### Pattern 2: 4-Layer Bond Portfolio (LayerLedger per layer)

**What:** Bond capital is partitioned into 4 virtual sub-accounts (Core, Strategic, Tactical, Short), each with its own LayerLedger, circuit breaker, and strategy set. BondCycleProcessor iterates layers in order: aggregate breaker check → per-layer breaker → yield stops → signals → sizing → execute.

**When to use:** Whenever adding a new bond segment or changing allocation weights. Layer separation ensures Core (OFZ-PK floaters) is never liquidated by portfolio-level drawdown events.

**Trade-offs:** More bookkeeping than a flat portfolio. Payoff is independent risk limits per layer and clear separation of strategy intent (Core = income, Strategic = duration rotation, Tactical = CBR events, Short = inverse/hedge).

```
Bond Portfolio Split (target allocations):
  Core    (OFZ-PK floaters)    ~45% — buy-and-hold, max DD 9%
  Strategic (OFZ-PD 3-7Y)     ~27.5% — duration rotation, max DD 5%
  Tactical  (OFZ-PD 2-5Y)     ~17.5% — CBR event trades, max DD 3%
  Short     (inverse ETF/cash) ~10%   — hedge, max DD 3%
```

### Pattern 3: Macro Context Injection via MacroCacheService

**What:** All bond strategies receive macro context (key_rate, ruonia_7d_avg, cpi_yoy, last_cbr_decision) as kwargs from BondCycleProcessor. The processor reads from MacroCacheService — a cached snapshot refreshed daily and force-refreshed on CBR meeting days. Bond strategies never call CBRFetcher directly.

**When to use:** Any new bond strategy or ML feature that needs CBR/macro data.

**Trade-offs:** Strategies receive possibly-stale data (up to 24h). Acceptable for daily-bar bond strategies. Not suitable for intraday FX or real-time rate moves.

### Pattern 4: News Pipeline → Sentiment Cache → Event-Driven Signals

**What:** TradingLoop's `_news_cycle` fetches articles, runs NewsAnalyzer (Claude Sonnet call), updates `_sentiment_cache` (protected by threading.Lock). The `_strategy_cycle` reads from the cache. The news cycle is intentionally decoupled from the strategy cycle — a slow LLM call does not block order generation.

**When to use:** Adding new news sources (T-Invest API, Telegram channels, RBC RSS). Each source produces NewsArticle objects; the same NewsAnalyzer handles them regardless of source.

**Trade-offs:** Sentiment is always at most one cycle stale (5 min lag maximum). For slow-moving MOEX instruments on daily bars, this is acceptable. If an article breaks 30 seconds before market close, it won't be acted on until next open — acceptable risk.

### Pattern 5: Fire-and-Forget Alerting

**What:** TelegramAlerter uses `asyncio.get_event_loop().create_task()` if a loop is running, else `asyncio.run()`. All exceptions are suppressed. If `bot_token` is empty, all methods are no-ops (safe for debug/test modes).

**When to use:** Every new alert type (coupon received, macro event, ML model degradation) should add a dedicated method to TelegramAlerter rather than calling `send_alert()` directly with formatted strings from business logic.

**Trade-offs:** Alert delivery is not guaranteed (fire-and-forget). This is correct — a Telegram delivery failure must never abort a trade execution.

## Data Flow

### Bond Trading Cycle Flow

```
APScheduler (daily, 10:15 Moscow)
    ↓
MacroCacheService.get()  →  MacroSnapshot (key_rate, ruonia, cpi, last_decision)
    ↓
AggregateBondBreaker.check()
    ↓ (if NOT halted)
for each layer [Core, Strategic, Tactical, Short]:
    BondLayerBreaker.check()
        ↓ (if NOT halted)
    YieldStop.evaluate(positions)  →  SELL signals for deteriorated bonds
        ↓
    TinkoffFetcher.fetch_candles(symbol, 90d)  (per bond)
        ↓
    BondStrategy.generate_signal(candles, macro_kwargs)  →  Signal | None
        ↓
    MLFilter.filter(signals)  [no-op until bond ML models trained]
        ↓
    DV01BudgetStep / EqualWeightBondSizer → quantity (int)
        ↓
    BrokerRouter.submit(order, market_id="moex")
        ↓
    TinkoffBroker.submit_order()  →  OrderResult
        ↓
    TelegramAlerter.on_trade_filled() / on_trade_rejected()
    LayerLedger.add_position() / debit_cash()
```

### News Analysis Flow

```
APScheduler (every 5 min)
    ↓
NewsFetcher.fetch_articles()  ← T-Invest news API + RSS sources
    ↓  (list[NewsArticle])
NewsAnalyzer.analyze(article)  ← LLMClient → Claude Sonnet
    ↓  (SentimentResult: score [-1,1], confidence, reasoning)
EventClassifier.classify(article)  →  EventType
    ↓
ImpactEstimator.estimate(event_type, sentiment)  →  impact_score
    ↓
_sentiment_cache[symbol] = SentimentResult  (with threading.Lock)
    ↓
[15 min later] StrategyCombiner reads sentiment cache
    event_driven strategy weight applied to combined signal
```

### Stock Strategy Cycle Flow

```
APScheduler (every 15 min)
    ↓
for each MOEX instrument in registry:
    TinkoffFetcher.fetch_candles(symbol, recent)
        ↓
    ADX router: trend (ADX>30) → momentum pool | MR (ADX<20) → MR pool
        ↓
    StrategyCombiner.generate_signal(candles, sentiment_score)
        ↓  (weighted signal from 5 strategies + optional ML reinforcer)
    CircuitBreaker.check(market_id="moex")
        ↓ (if NOT halted)
    PreTradeCheck.run(signal, portfolio)  — 11 checks including RUB sizing
        ↓ (if APPROVED)
    BrokerRouter.submit(order, market_id="moex")
        ↓
    TelegramAlerter.on_trade_filled()
```

### Telegram Alert Flow

```
Trade event occurs in TradingLoop / BondCycleProcessor
    ↓
TelegramAlerter.<specific_method>(...)
    ↓
send_alert(formatted_text)
    ↓  (fire-and-forget)
asyncio task → POST https://api.telegram.org/bot{token}/sendMessage
    ↓
User receives notification on Telegram (best-effort)
```

## Scaling Considerations

This is a single-account autonomous trading system, not a multi-tenant SaaS. Scaling means handling more instruments and more news volume, not more users.

| Scale Concern | Current Approach | Limit / When to Revisit |
|---------------|-----------------|--------------------------|
| News LLM calls | Synchronous sequential per article | At >50 articles/cycle, parallelize with asyncio.gather() |
| Bond instrument count | Sequential fetch per bond in cycle | At >100 bonds, batch candle fetches via T-Invest streaming |
| MacroCache refresh | Daily batch from cbr.ru | CBR API rate limits are not a concern at 1 request/day |
| T-Invest gRPC rate limits | Retry with backoff in TinkoffBroker | Monitor 429s; add token bucket if needed |
| Sentiment cache size | In-memory dict (symbol → SentimentResult) | No concern at <1000 symbols |
| APScheduler jobs | BackgroundScheduler (thread-based) | Fine for <10 cycles. If cycles overlap frequently, add job coalesce=True |

## Anti-Patterns

### Anti-Pattern 1: Bond Strategies as BaseStrategy Subclasses

**What people do:** Try to fit BondCarryStrategy, CBREventStrategy into BaseStrategy by adding optional kwargs.

**Why it's wrong:** BaseStrategy.generate_signal() expects standard candle/market context. Bond strategies need macro kwargs (key_rate, ruonia, cpi_yoy, last_cbr_decision) that have no meaning for equity strategies. Forcing the same interface creates leaky abstractions and requires every equity strategy to accept and ignore bond-specific kwargs.

**Do this instead:** Keep bond strategies as independent classes with their own `generate_signal(symbol, candles, open_positions, bar_idx, **macro_kwargs)` interface. BondCycleProcessor knows about this interface explicitly.

### Anti-Pattern 2: Calling CBRFetcher Directly from Strategies

**What people do:** Bond strategies import CBRFetcher and call it directly to get the latest key rate.

**Why it's wrong:** Creates L2 → L4 upward dependency violation. Also causes redundant HTTP calls (one per strategy per bond per cycle). CBR rate limits are not generous.

**Do this instead:** MacroCacheService provides the macro snapshot. BondCycleProcessor injects it as kwargs. Strategies only consume what is passed in — they never fetch.

### Anti-Pattern 3: Blocking the Trading Loop with LLM Calls

**What people do:** Call `await news_analyzer.analyze(article)` inside the strategy cycle to get "fresh" sentiment.

**Why it's wrong:** Claude API calls take 1-5 seconds each. With 10+ articles/cycle this blocks order generation for up to 50 seconds. MOEX orders need to be submitted before market close.

**Do this instead:** Keep news cycle and strategy cycle separate (already done in TradingLoop). The strategy cycle reads from `_sentiment_cache` — always fast, never blocking.

### Anti-Pattern 4: Telegram Alerts That Can Raise Exceptions

**What people do:** Add new alert methods that do not wrap exceptions, expecting callers to handle them.

**Why it's wrong:** An alert failure (network timeout, invalid token, Telegram API error) must never propagate into the trading loop and abort an order.

**Do this instead:** All TelegramAlerter methods catch all exceptions internally. The calling code never needs a try/except around alert calls.

### Anti-Pattern 5: Mixing RUB and USD Position Sizing

**What people do:** Use the same HalfKelly position sizer for MOEX that was calibrated for USD-denominated US equities.

**Why it's wrong:** MOEX instruments are priced in RUB. Using USD-calibrated sizing produces positions that are ~80x too small (e.g. 0.02% instead of 15%). This was identified as a known bug in the MVP plan.

**Do this instead:** CurrencyConverter provides RUB/USD conversion. All MOEX position sizing must be done in RUB before converting to lot quantities. The DV01BudgetStep and EqualWeightBondSizer already operate in RUB natively (face_value = 1000 RUB).

## Integration Points

### External Services

| Service | Protocol | Integration Pattern | Notes |
|---------|----------|---------------------|-------|
| T-Invest (Tinkoff) gRPC | gRPC | `AsyncClient` as context manager, `target="invest-public-api.tbank.ru:443"` | Old `tinkoff.ru` domain no longer works. Set `GRPC_DNS_RESOLVER=native`. |
| T-Invest Sandbox | gRPC | `AsyncSandboxClient`, `target="sandbox-invest-public-api.tbank.ru:443"` | Separate endpoint. Use for all pre-production validation. |
| CBR XML/SOAP | HTTP | Sync httpx client in CBRFetcher. Never call from async without `asyncio.to_thread()` | Rate-limit-free. CBR allows repeated polling. |
| Claude Sonnet (Anthropic) | REST | `LLMClient.complete(prompt, system)` async | Costs money per call — cache aggressively. Do not call per-instrument per-cycle. |
| Telegram Bot API | HTTP | httpx POST to `api.telegram.org/bot{token}/sendMessage` | Fire-and-forget. Token configured via `FINALAYZE_TELEGRAM_BOT_TOKEN` and `FINALAYZE_TELEGRAM_CHAT_ID`. |
| MOEX ISS | REST | `MOEXISSFetcher` — index levels, instrument lists | Free, no auth. Use for IMOEX benchmark. |

### Internal Boundaries

| Boundary | Communication Pattern | Notes |
|----------|-----------------------|-------|
| TradingLoop ↔ BondCycleProcessor | Direct method call (`run_cycle()`) | BondCycleProcessor is instantiated inside TradingLoop |
| StrategyCombiner ↔ EventDriven | `_sentiment_cache` dict read | Protected by `threading.Lock`. Strategy reads latest cached sentiment. |
| BondCycleProcessor ↔ MacroCacheService | `macro_cache.get()` → MacroSnapshot | Sync read. MacroCacheService refreshes on separate APScheduler job. |
| Bond Strategies ↔ Risk | DV01BudgetStep / EqualWeightBondSizer called by BondCycleProcessor | Strategies produce Signals; BondCycleProcessor handles sizing — strategies never self-size. |
| TelegramAlerter ↔ Trading Loop | Method calls at event points | Alerter is injected into TradingLoop and BondCycleProcessor constructors. |
| All → InstrumentRegistry | `.get(symbol, market_id)` or `.list_by_type()` | Registry is populated at startup from TinkoffFetcher instrument discovery. |

## Build Order Implications

The dependency structure dictates a specific build order for new MOEX features. Each phase should produce independently testable artifacts.

```
Phase A: Data layer completeness (L2)
  → MOEX instrument discovery (TinkoffFetcher → InstrumentRegistry population)
  → MOEX bond candle fetch validation
  → MacroCacheService with live CBR data
  GATE: backtests on MOEX bonds produce non-empty candle series

Phase B: Bond strategy calibration (L4)
  Depends on: Phase A (need live MOEX candle data)
  → BondCarryStrategy parameter tuning (rebalance interval, confidence thresholds)
  → CBREventStrategy gap threshold validation against historical meetings
  → BondDurationRotationStrategy yield curve logic
  GATE: bond backtest shows positive PnL with walk-forward validation

Phase C: Risk wiring (L4)
  Depends on: Phase A (need live data for DV01 calculation)
  → DV01BudgetStep validation (requires real bond duration data from T-Invest)
  → YieldStop full implementation (YTM computation from price + remaining coupons)
  → BondLayerBreaker integration into BondCycleProcessor (complete _size_and_execute stub)
  GATE: BondCycleProcessor.run_cycle() executes orders in T-Invest sandbox

Phase D: News pipeline activation (L3 → L4)
  Depends on: Phase A (instrument list needed for entity matching)
  → T-Invest news API fetcher
  → Russian media RSS fetcher (RBC, Interfax, TASS)
  → Telegram channel poller (python-telegram-bot or pyrogram)
  → event_driven strategy enable for MOEX
  GATE: sentiment signals influence combined signal on MOEX instruments in sandbox

Phase E: Telegram alerting validation (L6)
  Depends on: Phases B, C (need real trades to alert on)
  → TelegramAlerter integration test against sandbox trades
  → Daily P&L summary showing non-zero MOEX P&L
  → Coupon alert wiring (detect coupon payments from T-Invest portfolio events)
  GATE: operator receives correct alerts for every sandbox trade and daily summary

Phase F: Sandbox autonomous run (L6)
  Depends on: Phases A-E complete
  → End-to-end autonomous run for 5+ trading days in T-Invest sandbox
  → No critical errors, circuit breakers fire correctly, alerts arrive
  GATE: system operates 24/7 without human intervention for 5 days

Phase G: Real money deployment
  Depends on: Phase F GATE passed, max drawdown <5% in sandbox
  → Switch mode from sandbox → real
  → Monitor position sizing correctness (RUB amounts)
  → Hard stop: if drawdown >5% in first week, revert to sandbox
```

## Sources

All findings are HIGH confidence — derived from direct inspection of the production codebase.

- `src/finalayze/core/trading_loop.py` — TradingLoop scheduler architecture
- `src/finalayze/core/bond_cycle.py` — BondCycleProcessor 9-step pipeline
- `src/finalayze/core/alerts.py` — TelegramAlerter full API surface
- `src/finalayze/core/layer_ledger.py` — LayerLedger virtual sub-account pattern
- `src/finalayze/strategies/bond_carry.py` — BondCarryStrategy macro-aware interface
- `src/finalayze/strategies/cbr_event.py` — CBREventStrategy entry/exit logic
- `src/finalayze/risk/dv01_sizing.py` — DV01BudgetStep + EqualWeightBondSizer
- `src/finalayze/risk/layer_circuit_breaker.py` — 4-layer circuit breaker hierarchy
- `src/finalayze/data/macro_cache.py` — MacroCacheService daily refresh pattern
- `src/finalayze/execution/broker_router.py` — market_id dispatch pattern
- `src/finalayze/markets/instruments.py` — InstrumentRegistry with bond metadata
- `docs/architecture/DEPENDENCY_LAYERS.md` — enforced L0-L6 import rules
- `docs/architecture/DATA_FLOW.md` — original event bus and data flow diagrams

---
*Architecture research for: MOEX autonomous trading with bonds, news analysis, and alerting*
*Researched: 2026-03-14*

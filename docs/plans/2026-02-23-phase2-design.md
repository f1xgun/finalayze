# Phase 2 Design: MOEX + Tinkoff, Intelligence & Test Mode

**Date:** 2026-02-23
**Status:** Approved
**Phase:** 2 of 4

---

## Goal

Extend Phase 1's US-only backtest stack into a two-market system with:
- MOEX market data and paper trading via Tinkoff (t-tech-investments SDK)
- LLM-powered news analysis pipeline using an abstract, provider-agnostic client
- ML inference scaffolding (XGBoost + LightGBM per segment)
- Event-driven trading strategy
- Alpaca paper trading broker
- Broker routing layer

---

## Parallel Implementation Strategy

Two independent git worktrees developed simultaneously:

| Track | Branch | Scope |
|-------|--------|-------|
| A — Intelligence | `feature/phase2-intelligence` | LLM client, news analysis, ML pipeline, event strategy, DB migration |
| B — MOEX | `feature/phase2-moex` | t-tech-investments, TinkoffFetcher, brokers (Alpaca + Tinkoff), broker router |

Both tracks build on Phase 1 foundations (`BaseFetcher`, `BrokerBase`, `core/schemas.py`, `core/exceptions.py`).
**Merge order:** Track A first, then Track B.

---

## Track A: Intelligence Pipeline

### A1 — DB Migration + NewsArticle Schema

**New Pydantic schema** (`core/schemas.py`):
```python
class NewsArticle(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    source: str
    title: str
    content: str
    url: str
    language: str          # "en" | "ru"
    published_at: datetime
    symbols: list[str] = []
    affected_segments: list[str] = []
    scope: str | None = None   # "global" | "us" | "russia" | "sector"
    raw_sentiment: float | None = None
    credibility_score: float | None = None
```

**New Alembic migration** (`002_news_sentiment.py`):
- `news_articles` table (matches `NewsArticleModel` ORM)
- `sentiment_scores` table (TimescaleDB hypertable on `timestamp`)

### A2 — NewsAPI Fetcher

```
src/finalayze/data/fetchers/newsapi.py
```

- Extends `BaseFetcher` interface pattern (but returns `list[NewsArticle]` not `list[Candle]`)
- `httpx.Client` (sync, consistent with `FinnhubFetcher`)
- Integrates with existing `RateLimiter` (100 req/day free tier → `capacity=100, refill_rate=100/86400`)
- Maps NewsAPI JSON to `NewsArticle` schema
- Language detected from `language` field in response

### A3 — Abstract LLM Client

```
src/finalayze/analysis/llm_client.py
```

```python
class LLMClient(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str) -> str: ...

class OpenRouterClient(LLMClient):   # default
class OpenAIClient(LLMClient):       # GPT-4o etc.
class AnthropicClient(LLMClient):    # Claude (requires console API key)

def create_llm_client(settings: Settings) -> LLMClient:
    """Factory — reads settings.llm_provider."""
```

**Config additions** (`config/settings.py`):
```python
llm_provider: Literal["openrouter", "openai", "anthropic"] = "openrouter"
llm_model: str = "meta-llama/llama-3.1-8b-instruct:free"
llm_api_key: str = ""
```

**Implementation:**
- `OpenRouterClient` + `OpenAIClient` both use the `openai` Python SDK
  - OpenRouter: `base_url="https://openrouter.ai/api/v1"`
  - OpenAI: default `base_url`
- `AnthropicClient` uses the `anthropic` Python SDK
- All three: exponential backoff retry (3 attempts, 2^n seconds)
- Simple in-memory cache: `dict[str, str]` keyed by `sha256(system + prompt)`

### A4 — News Analyzer

```
src/finalayze/analysis/news_analyzer.py
src/finalayze/analysis/prompts/sentiment_en.txt
src/finalayze/analysis/prompts/sentiment_ru.txt
```

```python
class SentimentResult(BaseModel):
    sentiment: float    # -1.0 to +1.0
    confidence: float   # 0.0 to 1.0
    reasoning: str

class NewsAnalyzer:
    async def analyze(self, article: NewsArticle) -> SentimentResult: ...
```

- Selects prompt file by `article.language`
- Prompts instruct LLM to return JSON: `{"sentiment": float, "confidence": float, "reasoning": str}`
- Parses JSON response; on parse failure returns `SentimentResult(sentiment=0.0, confidence=0.0, reasoning="parse_error")`

### A5 — Event Classifier

```
src/finalayze/analysis/event_classifier.py
src/finalayze/analysis/prompts/classify_event.txt
```

```python
class EventType(StrEnum):
    EARNINGS = "earnings"
    FDA = "fda"
    MACRO = "macro"
    GEOPOLITICAL = "geopolitical"
    CBR_RATE = "cbr_rate"
    OIL_PRICE = "oil_price"
    SANCTIONS = "sanctions"
    OTHER = "other"

class EventClassifier:
    async def classify(self, article: NewsArticle) -> EventType: ...
```

- LLM returns one of the `EventType` string values
- On unrecognized value → `EventType.OTHER`

### A6 — Impact Estimator

```
src/finalayze/analysis/impact_estimator.py
```

```python
class SegmentImpact(BaseModel):
    segment_id: str
    weight: float       # 1.0 = primary, 0.3 = secondary
    sentiment: float

class ImpactEstimator:
    def estimate(
        self,
        article: NewsArticle,
        event: EventType,
        sentiment: SentimentResult,
        active_segments: list[str],
    ) -> list[SegmentImpact]: ...
```

**Routing table (no LLM needed — pure logic):**
- `scope="global"` → all active segments (weight=1.0)
- `scope="us"` → `us_*` segments (weight=1.0), `ru_*` (weight=0.3 if commodity-related)
- `scope="russia"` → `ru_*` segments (weight=1.0), no secondary
- `scope="sector"` → matching segments by `EventType` → segment mapping

**EventType → segment mapping:**
```
OIL_PRICE     → ru_energy (1.0), ru_blue_chips (0.5)
CBR_RATE      → ru_finance (1.0), ru_blue_chips (0.3)
SANCTIONS     → ru_* (1.0)
EARNINGS      → symbol's segment only (1.0)
FDA           → us_healthcare (1.0)
MACRO         → all segments (0.5)
GEOPOLITICAL  → all segments (0.3)
```

### A7 — ML Pipeline Scaffolding

```
src/finalayze/ml/features/technical.py
src/finalayze/ml/models/xgboost_model.py
src/finalayze/ml/models/lightgbm_model.py
src/finalayze/ml/models/ensemble.py
src/finalayze/ml/registry.py
```

**Feature engineering** (`technical.py`):
```python
def compute_features(candles: list[Candle], sentiment_score: float = 0.0) -> dict[str, float]:
    # Returns: rsi_14, macd_hist, bb_pct_b, volume_ratio_20d, atr_14, sentiment
```

**Model interface:**
```python
class BaseMLModel(ABC):
    segment_id: str
    @abstractmethod
    def predict_proba(self, features: dict[str, float]) -> float: ...  # 0.0-1.0 BUY probability
    @abstractmethod
    def fit(self, X: list[dict[str, float]], y: list[int]) -> None: ...

class XGBoostModel(BaseMLModel): ...
class LightGBMModel(BaseMLModel): ...

class EnsembleModel:
    def predict_proba(self, features: dict[str, float]) -> float:
        # Average of XGBoost + LightGBM probabilities
```

**Model registry** (`registry.py`):
```python
class MLModelRegistry:
    def get(self, segment_id: str) -> EnsembleModel | None: ...
    def register(self, segment_id: str, model: EnsembleModel) -> None: ...
```

> **Scope note:** Phase 2 delivers inference scaffolding + unit tests with mock/synthetic data.
> Model training scripts (`scripts/train_models.py`) are delivered in Phase 3.
> Models start untrained — strategies degrade gracefully when ML model is absent.

### A8 — Event-Driven Strategy

```
src/finalayze/strategies/event_driven.py
src/finalayze/strategies/presets/us_tech.yaml  (already exists — update event_driven section)
```

```python
class EventDrivenStrategy(BaseStrategy):
    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        credibility: float = 1.0,
    ) -> Signal | None: ...
```

**Logic:**
- `sentiment_score > min_sentiment` → BUY; `< -min_sentiment` → SELL; else HOLD/None
- `confidence = min(1.0, abs(sentiment_score) * credibility)`
- `min_sentiment` read from YAML preset per segment
- Falls back gracefully to HOLD when `sentiment_score == 0.0` (no news)

---

## Track B: MOEX / Tinkoff

### B1 — Package Setup

`pyproject.toml`:
```toml
[[tool.uv.index]]
name = "tbank"
url = "https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple"
explicit = true  # only used for t-tech-investments, not all packages
```

Dependencies added: `t-tech-investments`, `alpaca-py` (already in pyproject for Phase 1).

New settings (`config/settings.py`):
```python
tinkoff_token: str = ""
tinkoff_sandbox: bool = True
alpaca_api_key: str = ""
alpaca_secret_key: str = ""
alpaca_paper: bool = True
```

### B2 — TinkoffFetcher

```
src/finalayze/data/fetchers/tinkoff_data.py
```

```python
class TinkoffFetcher(BaseFetcher):
    _TIMEFRAME_MAP: dict[str, CandleInterval] = {
        "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
        "1d": CandleInterval.CANDLE_INTERVAL_DAY,
    }

    def fetch_candles(self, symbol, start, end, timeframe="1d") -> list[Candle]: ...
    def _symbol_to_figi(self, symbol: str) -> str: ...
    def _quotation_to_decimal(self, q: Quotation) -> Decimal: ...
```

**Key implementation details:**
- `_symbol_to_figi`: looks up `InstrumentRegistry`, raises `InstrumentNotFoundError` if missing
- `_quotation_to_decimal`: converts Tinkoff's `Quotation(units=int, nano=int)` → `Decimal` via `units + nano/1_000_000_000`
- Wraps `AsyncClient.get_all_candles()` in `asyncio.run()` for sync interface
- Uses `market_id="moex"`, `source="tinkoff"`

### B3 — MOEX Instruments

Updates `src/finalayze/markets/instruments.py` `DEFAULT_MOEX_INSTRUMENTS`:

| Symbol | Name | FIGI |
|--------|------|------|
| SBER | Sberbank | BBG004730N88 |
| GAZP | Gazprom | BBG004730RP0 |
| LKOH | Lukoil | BBG004731032 |
| GMKN | Norilsk Nickel | BBG004731489 |
| YNDX | Yandex | BBG006L8G4H1 |
| NVTK | Novatek | BBG00475KKY8 |
| ROSN | Rosneft | BBG004731354 |
| VTBR | VTB Bank | BBG004730ZJ9 |

### B4 — Alpaca Broker

```
src/finalayze/execution/alpaca_broker.py
```

```python
class AlpacaBroker(BrokerBase):
    async def submit_order(self, order: OrderRequest) -> OrderResult: ...
    async def cancel_order(self, order_id: str) -> None: ...
    async def get_portfolio(self) -> PortfolioState: ...
    async def get_positions(self) -> dict[str, Decimal]: ...
```

- Uses `alpaca-py`: `TradingClient` for orders, `StockHistoricalDataClient` for prices
- Paper trading when `settings.alpaca_paper = True` (uses paper endpoint)
- Raises `BrokerError` on API failures, `InsufficientFundsError` on balance issues

### B5 — Tinkoff Broker

```
src/finalayze/execution/tinkoff_broker.py
```

```python
class TinkoffBroker(BrokerBase):
    async def submit_order(self, order: OrderRequest) -> OrderResult: ...
    async def cancel_order(self, order_id: str) -> None: ...
    async def get_portfolio(self) -> PortfolioState: ...
```

- Uses `t_tech.invest.AsyncClient` (production) or `SandboxClient` (sandbox)
- **Lot-size aware:** MOEX shares trade in lots — `quantity = floor(requested_qty / lot_size) * lot_size`
- Raises `InstrumentNotFoundError` if FIGI lookup fails

### B6 — Broker Router

```
src/finalayze/execution/broker_router.py
```

```python
class BrokerRouter:
    def __init__(self, brokers: dict[str, BrokerBase]) -> None: ...
    def route(self, order: OrderRequest) -> BrokerBase: ...
    # Routes: market_id="us" → AlpacaBroker, market_id="moex" → TinkoffBroker
    # Raises BrokerError if no broker registered for market_id
```

---

## Testing Approach

| Component | Strategy | Mock target |
|-----------|----------|-------------|
| NewsApiFetcher | Unit | `httpx.Client.get` |
| LLMClient (all 3) | Unit | underlying SDK methods |
| NewsAnalyzer | Unit | `LLMClient.complete` |
| EventClassifier | Unit | `LLMClient.complete` |
| ImpactEstimator | Unit | pure logic, no mock |
| ML models | Unit | synthetic `numpy` arrays |
| EventDrivenStrategy | Unit | sentiment score fixture |
| TinkoffFetcher | Unit | `AsyncClient.get_all_candles` |
| AlpacaBroker | Unit | `alpaca-py` TradingClient |
| TinkoffBroker | Unit | `AsyncClient` / `SandboxClient` |
| BrokerRouter | Unit | mock broker instances |

Coverage target: ≥80% per new module. No live API calls in tests.

---

## Dependency Notes

- `openai` Python SDK — covers OpenRouter + OpenAI (one package, different `base_url`)
- `anthropic` Python SDK — for `AnthropicClient` (optional use)
- `t-tech-investments` — from tbank custom PyPI index
- `alpaca-py` — already listed in pyproject.toml (Phase 1 stub)
- `xgboost`, `lightgbm` — add to pyproject.toml

---

## What's Deferred to Phase 3

- Streamlit dashboard
- LSTM model
- Model training scripts (`scripts/train_models.py`)
- Pairs trading strategy
- Per-market circuit breakers
- Currency risk monitoring alerts
- Reddit sentiment fetcher

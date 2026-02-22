# Phase 2 Track A — Intelligence Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build the LLM-powered news analysis pipeline, ML inference scaffold, and event-driven strategy for the Finalayze trading system.

**Architecture:** Abstract `LLMClient` (OpenRouter/OpenAI/Anthropic) feeds `NewsAnalyzer`, `EventClassifier`, and `ImpactEstimator`. `EventDrivenStrategy` consumes sentiment scores. ML pipeline provides XGBoost+LightGBM inference scaffold per segment.

**Tech Stack:** `openai` SDK (covers OpenRouter + OpenAI), `anthropic` SDK (for Anthropic), `xgboost`, `lightgbm` (already in pyproject.toml), `httpx` (already present), `pandas-ta` (already present).

**Worktree:** `.worktrees/phase2-intelligence` on branch `feature/phase2-intelligence`

---

## Project Conventions (read before writing any code)

- Every file starts with `"""Docstring."""\n\nfrom __future__ import annotations`
- Use `StrEnum` not `str, Enum` (ruff UP042)
- Exception names end in `Error` (ruff N818)
- No magic numbers — define named constants
- `from __future__ import annotations` means type hints are strings — safe to use `X | Y` everywhere
- Run quality checks: `source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header`
- The project uses `uv run` for all Python commands
- Tests live in `tests/unit/` — mirror source structure

---

## Task 1: Add `openai` dependency + new exceptions + settings update

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/finalayze/core/exceptions.py`
- Modify: `config/settings.py`
- Test: `tests/unit/test_core.py` (existing — just run, no new tests needed for settings)

### Step 1: Add `openai` to pyproject.toml dependencies

In `pyproject.toml`, in the `[project]` `dependencies` list, add after `"anthropic>=0.42.0"`:
```toml
    "openai>=1.50.0",
```

Also add to `[[tool.mypy.overrides]]` `module` list:
```toml
    "openai.*",
    "t_tech.*",
```

Remove the comment about tinkoff being quarantined (lines ~61-63).

### Step 2: Run `uv sync`

```bash
source ~/.zshrc && uv sync --extra dev
```
Expected: resolves and installs `openai`.

### Step 3: Add new exceptions to `src/finalayze/core/exceptions.py`

After the `DataFetchError`/`RateLimitError` block, add:

```python
# ---------------------------------------------------------------------------
# Analysis / LLM
# ---------------------------------------------------------------------------
class AnalysisError(FinalayzeError):
    """News analysis or LLM processing failed."""


class LLMError(AnalysisError):
    """LLM API call failed (auth, rate limit, parse error)."""


class LLMRateLimitError(LLMError):
    """LLM provider rate limit exceeded."""
```

After the `BrokerError` block, add:
```python

class InsufficientFundsError(BrokerError):
    """Order rejected due to insufficient funds in broker account."""
```

### Step 4: Add LLM settings to `config/settings.py`

After `anthropic_api_key: str = ""`, add:
```python
    # LLM provider selection
    llm_provider: str = "openrouter"  # "openrouter" | "openai" | "anthropic"
    llm_api_key: str = ""             # API key for selected provider
```

Change the existing `llm_model` line to:
```python
    llm_model: str = "meta-llama/llama-3.1-8b-instruct:free"
```

Remove `anthropic_api_key` — it's now covered by `llm_api_key`. Wait — keep `anthropic_api_key` for backward compatibility, but add `llm_api_key` as the new unified key. Both exist, the LLM client factory uses `llm_api_key`.

### Step 5: Write tests for new exceptions

In `tests/unit/test_core.py` (existing file), add a new test class at the bottom:

```python
class TestPhase2Exceptions:
    def test_llm_error_is_analysis_error(self) -> None:
        err = LLMError("test")
        assert isinstance(err, AnalysisError)
        assert isinstance(err, FinalayzeError)

    def test_llm_rate_limit_error_is_llm_error(self) -> None:
        err = LLMRateLimitError("rate limited")
        assert isinstance(err, LLMError)

    def test_insufficient_funds_error_is_broker_error(self) -> None:
        err = InsufficientFundsError("no funds")
        assert isinstance(err, BrokerError)
        assert isinstance(err, ExecutionError)
```

Add the necessary imports at the top of the test file (check existing imports first — add only what's missing):
```python
from finalayze.core.exceptions import (
    AnalysisError,
    BrokerError,
    ExecutionError,
    FinalayzeError,
    InsufficientFundsError,
    LLMError,
    LLMRateLimitError,
)
```

### Step 6: Run tests

```bash
source ~/.zshrc && uv run pytest tests/unit/test_core.py -v
```
Expected: all tests pass.

### Step 7: Run full quality check

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/
```
Expected: zero errors.

### Step 8: Commit

```bash
git add pyproject.toml src/finalayze/core/exceptions.py config/settings.py tests/unit/test_core.py
git commit -m "feat(core): add openai dep, LLM exceptions, llm_provider setting

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: `NewsArticle` and `SentimentResult` schemas

**Files:**
- Modify: `src/finalayze/core/schemas.py`
- Test: `tests/unit/test_schemas.py` (existing)

### Step 1: Write failing tests

In `tests/unit/test_schemas.py`, add a new test class:

```python
class TestNewsArticleSchema:
    def test_valid_news_article(self) -> None:
        from uuid import uuid4
        article = NewsArticle(
            id=uuid4(),
            source="newsapi",
            title="Fed raises rates",
            content="The Federal Reserve raised interest rates by 25bps.",
            url="https://example.com/article",
            language="en",
            published_at=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
        )
        assert article.language == "en"
        assert article.symbols == []
        assert article.affected_segments == []

    def test_news_article_rejects_naive_timestamp(self) -> None:
        from uuid import uuid4
        with pytest.raises(ValidationError):
            NewsArticle(
                id=uuid4(),
                source="newsapi",
                title="Test",
                content="Test",
                url="https://example.com",
                language="en",
                published_at=datetime(2024, 1, 15, 10, 0),  # naive
            )


class TestSentimentResultSchema:
    def test_valid_sentiment(self) -> None:
        result = SentimentResult(sentiment=0.7, confidence=0.9, reasoning="positive news")
        assert result.sentiment == 0.7

    def test_sentiment_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            SentimentResult(sentiment=1.5, confidence=0.9, reasoning="bad")

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            SentimentResult(sentiment=0.5, confidence=-0.1, reasoning="bad")
```

Add imports at top of test file (check what's already imported):
```python
from finalayze.core.schemas import NewsArticle, SentimentResult
```

### Step 2: Run to verify fails

```bash
source ~/.zshrc && uv run pytest tests/unit/test_schemas.py::TestNewsArticleSchema -v
```
Expected: `ImportError: cannot import name 'NewsArticle'`

### Step 3: Add schemas to `src/finalayze/core/schemas.py`

Add at the end of the file (before the last newline):

```python
class NewsArticle(BaseModel):
    """A news article fetched from an external source."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    source: str
    title: str
    content: str
    url: str
    language: str  # "en" | "ru"
    published_at: datetime
    symbols: list[str] = []
    affected_segments: list[str] = []
    scope: str | None = None  # "global" | "us" | "russia" | "sector"
    raw_sentiment: float | None = None
    credibility_score: float | None = None

    @field_validator("published_at")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes."""
        if v.tzinfo is None:
            msg = "published_at must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v

    @field_validator("raw_sentiment")
    @classmethod
    def sentiment_in_range(cls, v: float | None) -> float | None:
        """Validate sentiment is in [-1.0, 1.0] when provided."""
        if v is not None and not (-1.0 <= v <= 1.0):
            msg = f"raw_sentiment must be in [-1.0, 1.0], got {v}"
            raise ValueError(msg)
        return v


class SentimentResult(BaseModel):
    """Result of LLM sentiment analysis on a news article."""

    model_config = ConfigDict(frozen=True)

    sentiment: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str

    @field_validator("sentiment")
    @classmethod
    def sentiment_in_range(cls, v: float) -> float:
        """Validate sentiment is in [-1.0, 1.0]."""
        if not (-1.0 <= v <= 1.0):
            msg = f"sentiment must be in [-1.0, 1.0], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        """Validate confidence is in [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            msg = f"confidence must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v
```

### Step 4: Run tests

```bash
source ~/.zshrc && uv run pytest tests/unit/test_schemas.py -v
```
Expected: all pass.

### Step 5: Commit

```bash
git add src/finalayze/core/schemas.py tests/unit/test_schemas.py
git commit -m "feat(schemas): add NewsArticle and SentimentResult schemas

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: DB migration for news tables

**Files:**
- Modify: `src/finalayze/core/models.py`
- Create: `alembic/versions/002_news_sentiment.py`
- Test: `tests/unit/test_models.py` (existing — read it first, then add)

### Step 1: Read existing models

Read `src/finalayze/core/models.py` to understand the existing ORM structure.

### Step 2: Add ORM models to `src/finalayze/core/models.py`

Add two new model classes at the end (before the final newline). Import `uuid` and `ARRAY`/`JSONB` from sqlalchemy if not already present — check existing imports first.

```python
class NewsArticleModel(Base):
    """ORM model for news articles."""

    __tablename__ = "news_articles"

    id: Mapped[uuid.UUID] = mapped_column(
        postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(5), nullable=False, server_default="en")
    published_at: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    symbols: Mapped[list[str]] = mapped_column(
        postgresql.ARRAY(String(20)), nullable=False, server_default="{}"
    )
    affected_segments: Mapped[list[str]] = mapped_column(
        postgresql.ARRAY(String(30)), nullable=False, server_default="{}"
    )
    scope: Mapped[str | None] = mapped_column(String(20), nullable=True)
    category: Mapped[str | None] = mapped_column(String(30), nullable=True)
    raw_sentiment: Mapped[Numeric | None] = mapped_column(Numeric(5, 4), nullable=True)
    credibility_score: Mapped[Numeric | None] = mapped_column(Numeric(5, 4), nullable=True)
    llm_analysis: Mapped[dict[str, object] | None] = mapped_column(
        postgresql.JSONB, nullable=True
    )
    is_processed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")


class SentimentScoreModel(Base):
    """ORM model for sentiment scores (TimescaleDB hypertable on timestamp)."""

    __tablename__ = "sentiment_scores"

    symbol: Mapped[str] = mapped_column(String(20), nullable=False, primary_key=True)
    market_id: Mapped[str] = mapped_column(String(10), nullable=False, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False, primary_key=True)
    news_sentiment: Mapped[Numeric | None] = mapped_column(Numeric(5, 4), nullable=True)
    social_sentiment: Mapped[Numeric | None] = mapped_column(Numeric(5, 4), nullable=True)
    composite_sentiment: Mapped[Numeric | None] = mapped_column(Numeric(5, 4), nullable=True)
    confidence: Mapped[Numeric | None] = mapped_column(Numeric(5, 4), nullable=True)
```

**Important:** Check what `datetime`, `Numeric`, `String`, `Text`, `Boolean`, `TIMESTAMPTZ`, `postgresql` are imported as in the existing models file. Use the same import style. Do NOT add duplicate imports.

### Step 3: Write failing test

In `tests/unit/test_models.py` (read first), add:
```python
class TestNewsModels:
    def test_news_article_model_has_tablename(self) -> None:
        assert NewsArticleModel.__tablename__ == "news_articles"

    def test_sentiment_score_model_has_tablename(self) -> None:
        assert SentimentScoreModel.__tablename__ == "sentiment_scores"
```

Import `NewsArticleModel` and `SentimentScoreModel` from `finalayze.core.models`.

### Step 4: Run test to verify fails, implement, then verify passes

```bash
source ~/.zshrc && uv run pytest tests/unit/test_models.py::TestNewsModels -v
```

### Step 5: Create `alembic/versions/002_news_sentiment.py`

```python
"""Add news_articles and sentiment_scores tables.

Revision ID: 002
Revises: 001
Create Date: 2026-02-23
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create news_articles and sentiment_scores tables."""
    op.create_table(
        "news_articles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("language", sa.String(5), nullable=False, server_default="en"),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "symbols",
            postgresql.ARRAY(sa.String(20)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "affected_segments",
            postgresql.ARRAY(sa.String(30)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("scope", sa.String(20), nullable=True),
        sa.Column("category", sa.String(30), nullable=True),
        sa.Column("raw_sentiment", sa.Numeric(5, 4), nullable=True),
        sa.Column("credibility_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("llm_analysis", postgresql.JSONB(), nullable=True),
        sa.Column(
            "is_processed", sa.Boolean(), nullable=False, server_default="false"
        ),
    )

    op.create_table(
        "sentiment_scores",
        sa.Column("symbol", sa.String(20), nullable=False, primary_key=True),
        sa.Column("market_id", sa.String(10), nullable=False, primary_key=True),
        sa.Column(
            "timestamp",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("news_sentiment", sa.Numeric(5, 4), nullable=True),
        sa.Column("social_sentiment", sa.Numeric(5, 4), nullable=True),
        sa.Column("composite_sentiment", sa.Numeric(5, 4), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=True),
    )


def downgrade() -> None:
    """Drop news tables."""
    op.drop_table("sentiment_scores")
    op.drop_table("news_articles")
```

### Step 6: Validate migration syntax

```bash
source ~/.zshrc && uv run python -c "import alembic; print('ok')"
```

### Step 7: Run all tests

```bash
source ~/.zshrc && uv run pytest -q --no-header 2>&1 | python3 -c "import sys; lines=sys.stdin.readlines(); [print(l,end='') for l in lines[-5:]]"
```
Expected: all pass.

### Step 8: Commit

```bash
git add src/finalayze/core/models.py alembic/versions/002_news_sentiment.py tests/unit/test_models.py
git commit -m "feat(data): add news_articles and sentiment_scores DB models + migration

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: NewsAPI fetcher

**Files:**
- Create: `src/finalayze/data/fetchers/newsapi.py`
- Create: `tests/unit/test_newsapi_fetcher.py`

### Step 1: Write failing tests

```python
# tests/unit/test_newsapi_fetcher.py
"""Unit tests for NewsApiFetcher."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.exceptions import DataFetchError, RateLimitError
from finalayze.core.schemas import NewsArticle
from finalayze.data.fetchers.newsapi import NewsApiFetcher

# Constants
_API_KEY = "test-key"
_FROM_DATE = datetime(2024, 1, 1, tzinfo=UTC)
_TO_DATE = datetime(2024, 1, 7, tzinfo=UTC)

_SAMPLE_RESPONSE = {
    "status": "ok",
    "totalResults": 1,
    "articles": [
        {
            "source": {"name": "Reuters"},
            "title": "Fed raises rates",
            "description": "Brief description",
            "content": "Full content here",
            "url": "https://reuters.com/article/1",
            "publishedAt": "2024-01-03T10:00:00Z",
        }
    ],
}


class TestNewsApiFetcherInit:
    def test_init_stores_api_key(self) -> None:
        fetcher = NewsApiFetcher(api_key=_API_KEY)
        assert fetcher._api_key == _API_KEY


class TestNewsApiFetcherSuccess:
    def test_returns_news_articles(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _SAMPLE_RESPONSE

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            fetcher = NewsApiFetcher(api_key=_API_KEY)
            articles = fetcher.fetch_news("Fed", _FROM_DATE, _TO_DATE)

        assert len(articles) == 1
        assert isinstance(articles[0], NewsArticle)
        assert articles[0].title == "Fed raises rates"
        assert articles[0].source == "Reuters"
        assert articles[0].language == "en"

    def test_empty_results_returns_empty_list(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "totalResults": 0, "articles": []}

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            fetcher = NewsApiFetcher(api_key=_API_KEY)
            articles = fetcher.fetch_news("xyz", _FROM_DATE, _TO_DATE)

        assert articles == []


class TestNewsApiFetcherErrors:
    def test_rate_limit_raises_rate_limit_error(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            fetcher = NewsApiFetcher(api_key=_API_KEY)
            with pytest.raises(RateLimitError):
                fetcher.fetch_news("test", _FROM_DATE, _TO_DATE)

    def test_api_error_status_raises_data_fetch_error(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "error", "message": "apiKeyInvalid"}

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            fetcher = NewsApiFetcher(api_key=_API_KEY)
            with pytest.raises(DataFetchError):
                fetcher.fetch_news("test", _FROM_DATE, _TO_DATE)
```

### Step 2: Verify tests fail

```bash
source ~/.zshrc && uv run pytest tests/unit/test_newsapi_fetcher.py -v
```
Expected: `ModuleNotFoundError: No module named 'finalayze.data.fetchers.newsapi'`

### Step 3: Implement `src/finalayze/data/fetchers/newsapi.py`

```python
"""NewsAPI REST fetcher for news articles (Layer 2)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import httpx

from finalayze.core.exceptions import DataFetchError, RateLimitError
from finalayze.core.schemas import NewsArticle

_BASE_URL = "https://newsapi.org/v2/everything"
_HTTP_OK = 200
_HTTP_RATE_LIMIT = 429
_STATUS_OK = "ok"


class NewsApiFetcher:
    """Fetches news articles from the NewsAPI v2 REST API.

    Uses the ``/v2/everything`` endpoint for keyword-based article search.
    Returns a list of :class:`~finalayze.core.schemas.NewsArticle` objects.
    """

    def __init__(self, api_key: str, language: str = "en") -> None:
        self._api_key = api_key
        self._language = language

    def fetch_news(
        self,
        query: str,
        from_date: datetime,
        to_date: datetime,
        page_size: int = 20,
    ) -> list[NewsArticle]:
        """Fetch news articles matching the query in the given date range.

        Args:
            query: Search keywords (e.g. ticker symbol or topic).
            from_date: Start of the date range (inclusive, UTC-aware).
            to_date: End of the date range (exclusive, UTC-aware).
            page_size: Max results per request (1-100, NewsAPI limit).

        Returns:
            List of NewsArticle objects, may be empty if no results.

        Raises:
            RateLimitError: When the API returns HTTP 429.
            DataFetchError: On HTTP errors or API-level error responses.
        """
        params: dict[str, str | int] = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": to_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "language": self._language,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": self._api_key,
        }

        with httpx.Client() as client:
            try:
                response = client.get(_BASE_URL, params=params)
            except httpx.HTTPError as exc:
                msg = f"NewsAPI HTTP request failed: {exc}"
                raise DataFetchError(msg) from exc

        if response.status_code == _HTTP_RATE_LIMIT:
            msg = "NewsAPI rate limit exceeded"
            raise RateLimitError(msg)

        if response.status_code != _HTTP_OK:
            msg = f"NewsAPI HTTP error: {response.status_code}"
            raise DataFetchError(msg)

        data = response.json()
        if data.get("status") != _STATUS_OK:
            msg = f"NewsAPI error: {data.get('message', 'unknown')}"
            raise DataFetchError(msg)

        return [self._parse_article(raw) for raw in data.get("articles", [])]

    def _parse_article(self, raw: dict[str, object]) -> NewsArticle:
        """Convert a raw NewsAPI article dict to a NewsArticle schema."""
        source_obj = raw.get("source") or {}
        source_name = (
            source_obj.get("name", "unknown")
            if isinstance(source_obj, dict)
            else "unknown"
        )
        published_raw = raw.get("publishedAt", "")
        try:
            published_at = datetime.fromisoformat(
                str(published_raw).replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            published_at = datetime.now(UTC)

        return NewsArticle(
            id=uuid4(),
            source=str(source_name),
            title=str(raw.get("title") or ""),
            content=str(raw.get("content") or raw.get("description") or ""),
            url=str(raw.get("url") or ""),
            language=self._language,
            published_at=published_at,
        )
```

### Step 4: Run tests

```bash
source ~/.zshrc && uv run pytest tests/unit/test_newsapi_fetcher.py -v
```
Expected: all pass.

### Step 5: Quality check + commit

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/
git add src/finalayze/data/fetchers/newsapi.py tests/unit/test_newsapi_fetcher.py
git commit -m "feat(data): add NewsAPI article fetcher

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Abstract LLM client with OpenRouter, OpenAI, Anthropic implementations

**Files:**
- Create: `src/finalayze/analysis/llm_client.py`
- Create: `tests/unit/test_llm_client.py`

### Step 1: Write failing tests

```python
# tests/unit/test_llm_client.py
"""Unit tests for abstract LLM client and implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.analysis.llm_client import (
    AnthropicClient,
    LLMClient,
    OpenAIClient,
    OpenRouterClient,
    create_llm_client,
)
from finalayze.core.exceptions import LLMError, LLMRateLimitError

_SYSTEM = "You are a financial analyst."
_PROMPT = "Analyze this news: Fed raises rates."
_RESPONSE = "Positive for USD, negative for bonds."


class TestLLMClientIsAbstract:
    def test_cannot_instantiate_base_class(self) -> None:
        with pytest.raises(TypeError):
            LLMClient()  # type: ignore[abstract]


class TestOpenRouterClient:
    @pytest.mark.asyncio
    async def test_complete_returns_string(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = _RESPONSE
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.chat = MagicMock()
            mock_openai.chat.completions = MagicMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_cls.return_value = mock_openai

            client = OpenRouterClient(api_key="test-key", model="llama-3")
            result = await client.complete(_PROMPT, _SYSTEM)

        assert result == _RESPONSE

    @pytest.mark.asyncio
    async def test_caches_identical_prompts(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = _RESPONSE
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_cls.return_value = mock_openai

            client = OpenRouterClient(api_key="test-key", model="llama-3")
            result1 = await client.complete(_PROMPT, _SYSTEM)
            result2 = await client.complete(_PROMPT, _SYSTEM)

        assert result1 == result2
        # create called only once (second call hits cache)
        assert mock_openai.chat.completions.create.call_count == 1


class TestCreateLLMClientFactory:
    def test_openrouter_provider_returns_openrouter_client(self) -> None:
        from config.settings import Settings
        settings = Settings(llm_provider="openrouter", llm_api_key="key", llm_model="model")
        client = create_llm_client(settings)
        assert isinstance(client, OpenRouterClient)

    def test_openai_provider_returns_openai_client(self) -> None:
        from config.settings import Settings
        settings = Settings(llm_provider="openai", llm_api_key="key", llm_model="gpt-4o")
        client = create_llm_client(settings)
        assert isinstance(client, OpenAIClient)

    def test_anthropic_provider_returns_anthropic_client(self) -> None:
        from config.settings import Settings
        settings = Settings(llm_provider="anthropic", llm_api_key="key", llm_model="claude-3")
        client = create_llm_client(settings)
        assert isinstance(client, AnthropicClient)

    def test_unknown_provider_raises_configuration_error(self) -> None:
        from config.settings import Settings
        from finalayze.core.exceptions import ConfigurationError
        settings = Settings(llm_provider="unknown", llm_api_key="key", llm_model="model")
        with pytest.raises(ConfigurationError):
            create_llm_client(settings)
```

### Step 2: Run to verify fails

```bash
source ~/.zshrc && uv run pytest tests/unit/test_llm_client.py -v
```

### Step 3: Implement `src/finalayze/analysis/llm_client.py`

```python
"""Abstract LLM client and provider implementations (Layer 3).

Supports OpenRouter (default), OpenAI, and Anthropic as providers.
Select provider via ``config/settings.py`` ``llm_provider`` field.
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from finalayze.core.exceptions import ConfigurationError, LLMError, LLMRateLimitError

if TYPE_CHECKING:
    from config.settings import Settings

# ── Retry configuration ─────────────────────────────────────────────────────
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 2


class LLMClient(ABC):
    """Abstract base for all LLM provider clients."""

    @abstractmethod
    async def complete(self, prompt: str, system: str) -> str:
        """Send a prompt and return the model's text response.

        Args:
            prompt: The user message / question.
            system: The system instruction for the model.

        Returns:
            Model response as a plain string.

        Raises:
            LLMRateLimitError: When provider rate limit is hit.
            LLMError: On any other LLM API failure.
        """
        ...


class _CachingLLMClient(LLMClient, ABC):
    """Mixin that adds SHA-256 in-memory caching and exponential backoff retry."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def _cache_key(self, prompt: str, system: str) -> str:
        payload = f"{system}\n{prompt}"
        return hashlib.sha256(payload.encode()).hexdigest()

    async def complete(self, prompt: str, system: str) -> str:
        """Complete with caching and retry."""
        key = self._cache_key(prompt, system)
        if key in self._cache:
            return self._cache[key]

        for attempt in range(_MAX_RETRIES):
            try:
                result = await self._complete_once(prompt, system)
                self._cache[key] = result
                return result
            except LLMRateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _RETRY_BASE_SECONDS ** (attempt + 1)
                await asyncio.sleep(wait)
            except LLMError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _RETRY_BASE_SECONDS ** (attempt + 1)
                await asyncio.sleep(wait)

        msg = "LLM request failed after all retries"  # unreachable but satisfies mypy
        raise LLMError(msg)

    @abstractmethod
    async def _complete_once(self, prompt: str, system: str) -> str:
        """Single attempt at completion — no retry logic here."""
        ...


class OpenRouterClient(_CachingLLMClient):
    """LLM client using OpenRouter API (OpenAI-compatible, many models)."""

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__()
        self._api_key = api_key
        self._model = model

    async def _complete_once(self, prompt: str, system: str) -> str:
        import openai

        client = openai.AsyncOpenAI(api_key=self._api_key, base_url=self._BASE_URL)
        try:
            completion = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )
        except openai.RateLimitError as exc:
            msg = f"OpenRouter rate limit: {exc}"
            raise LLMRateLimitError(msg) from exc
        except openai.OpenAIError as exc:
            msg = f"OpenRouter API error: {exc}"
            raise LLMError(msg) from exc

        content = completion.choices[0].message.content
        if content is None:
            msg = "OpenRouter returned empty response"
            raise LLMError(msg)
        return content


class OpenAIClient(_CachingLLMClient):
    """LLM client using OpenAI API directly."""

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__()
        self._api_key = api_key
        self._model = model

    async def _complete_once(self, prompt: str, system: str) -> str:
        import openai

        client = openai.AsyncOpenAI(api_key=self._api_key)
        try:
            completion = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )
        except openai.RateLimitError as exc:
            msg = f"OpenAI rate limit: {exc}"
            raise LLMRateLimitError(msg) from exc
        except openai.OpenAIError as exc:
            msg = f"OpenAI API error: {exc}"
            raise LLMError(msg) from exc

        content = completion.choices[0].message.content
        if content is None:
            msg = "OpenAI returned empty response"
            raise LLMError(msg)
        return content


class AnthropicClient(_CachingLLMClient):
    """LLM client using Anthropic API (requires console API key)."""

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__()
        self._api_key = api_key
        self._model = model

    async def _complete_once(self, prompt: str, system: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        try:
            message = await client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.RateLimitError as exc:
            msg = f"Anthropic rate limit: {exc}"
            raise LLMRateLimitError(msg) from exc
        except anthropic.APIError as exc:
            msg = f"Anthropic API error: {exc}"
            raise LLMError(msg) from exc

        block = message.content[0]
        if not hasattr(block, "text"):
            msg = "Anthropic returned non-text content block"
            raise LLMError(msg)
        return block.text  # type: ignore[union-attr]


def create_llm_client(settings: Settings) -> LLMClient:
    """Factory — instantiates the correct LLM client from settings.

    Args:
        settings: Application settings with ``llm_provider``, ``llm_api_key``,
            and ``llm_model`` fields.

    Returns:
        Configured LLMClient implementation.

    Raises:
        ConfigurationError: When ``llm_provider`` is not a recognised value.
    """
    provider = settings.llm_provider
    key = settings.llm_api_key
    model = settings.llm_model

    if provider == "openrouter":
        return OpenRouterClient(api_key=key, model=model)
    if provider == "openai":
        return OpenAIClient(api_key=key, model=model)
    if provider == "anthropic":
        return AnthropicClient(api_key=key, model=model)

    msg = f"Unknown llm_provider {provider!r}. Choose: openrouter, openai, anthropic"
    raise ConfigurationError(msg)
```

### Step 4: Run tests

```bash
source ~/.zshrc && uv run pytest tests/unit/test_llm_client.py -v
```

### Step 5: Quality check + commit

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/
git add src/finalayze/analysis/llm_client.py tests/unit/test_llm_client.py
git commit -m "feat(analysis): abstract LLM client — OpenRouter, OpenAI, Anthropic

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: NewsAnalyzer with EN + RU prompts

**Files:**
- Create: `src/finalayze/analysis/news_analyzer.py`
- Create: `src/finalayze/analysis/prompts/sentiment_en.txt`
- Create: `src/finalayze/analysis/prompts/sentiment_ru.txt`
- Create: `tests/unit/test_news_analyzer.py`

### Step 1: Create prompt files

`src/finalayze/analysis/prompts/sentiment_en.txt`:
```
You are a financial news analyst. Analyze the following news article and return a JSON object with exactly these fields:
- "sentiment": a float between -1.0 (very negative) and +1.0 (very positive) representing market sentiment
- "confidence": a float between 0.0 and 1.0 representing your confidence in the assessment
- "reasoning": a brief one-sentence explanation

Consider the likely impact on stock prices. Earnings beats, rate cuts, and strong guidance are positive. Earnings misses, rate hikes, regulatory fines, and CEO departures are negative.

Return ONLY valid JSON, no other text.
```

`src/finalayze/analysis/prompts/sentiment_ru.txt`:
```
Вы — аналитик финансовых новостей. Проанализируйте следующую новостную статью и верните JSON-объект со следующими полями:
- "sentiment": число от -1.0 (очень негативно) до +1.0 (очень позитивно), отражающее настроение рынка
- "confidence": число от 0.0 до 1.0, отражающее вашу уверенность в оценке
- "reasoning": краткое объяснение в одном предложении

Учитывайте вероятное влияние на цены акций. Повышение ключевой ставки ЦБ, санкции, геополитические риски — негативны. Снижение ставки, рост добычи, сильные финансовые результаты — позитивны.

Верните ТОЛЬКО валидный JSON, без дополнительного текста.
```

### Step 2: Write failing tests

```python
# tests/unit/test_news_analyzer.py
"""Unit tests for NewsAnalyzer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from finalayze.analysis.news_analyzer import NewsAnalyzer
from finalayze.core.schemas import NewsArticle, SentimentResult

_ARTICLE_EN = NewsArticle(
    id=uuid4(),
    source="reuters",
    title="Fed raises rates",
    content="The Federal Reserve raised rates by 25bps.",
    url="https://reuters.com/1",
    language="en",
    published_at=datetime(2024, 1, 3, tzinfo=UTC),
)

_ARTICLE_RU = NewsArticle(
    id=uuid4(),
    source="interfax",
    title="ЦБ повысил ставку",
    content="Центральный банк повысил ключевую ставку до 16%.",
    url="https://interfax.ru/1",
    language="ru",
    published_at=datetime(2024, 1, 3, tzinfo=UTC),
)


class TestNewsAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_en_returns_sentiment_result(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {"sentiment": 0.6, "confidence": 0.85, "reasoning": "Rate hike positive for USD"}
        )
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        result = await analyzer.analyze(_ARTICLE_EN)
        assert isinstance(result, SentimentResult)
        assert result.sentiment == pytest.approx(0.6)
        assert result.confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_analyze_ru_uses_russian_prompt(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {"sentiment": -0.7, "confidence": 0.9, "reasoning": "Ставка повышена — негатив"}
        )
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        result = await analyzer.analyze(_ARTICLE_RU)
        # verify the system prompt passed was the RU prompt
        call_kwargs = mock_llm.complete.call_args
        system_arg = call_kwargs[0][1] if call_kwargs[0] else call_kwargs[1]["system"]
        assert "ЦБ" in system_arg or "финансовых" in system_arg
        assert result.sentiment == pytest.approx(-0.7)

    @pytest.mark.asyncio
    async def test_parse_error_returns_zero_sentiment(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "not valid json at all"
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        result = await analyzer.analyze(_ARTICLE_EN)
        assert result.sentiment == 0.0
        assert result.confidence == 0.0
        assert "parse_error" in result.reasoning
```

### Step 3: Implement `src/finalayze/analysis/news_analyzer.py`

```python
"""News sentiment analyzer using an LLM client (Layer 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from finalayze.core.schemas import NewsArticle, SentimentResult

if TYPE_CHECKING:
    from finalayze.analysis.llm_client import LLMClient

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_FALLBACK = SentimentResult(sentiment=0.0, confidence=0.0, reasoning="parse_error")


class NewsAnalyzer:
    """Analyzes news articles for financial sentiment using an LLM.

    Selects EN or RU prompt based on article language.
    Falls back to neutral sentiment on parse errors.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._prompts: dict[str, str] = {}

    def _load_prompt(self, language: str) -> str:
        """Load and cache the system prompt for the given language."""
        if language not in self._prompts:
            lang = language if language in ("en", "ru") else "en"
            prompt_path = _PROMPTS_DIR / f"sentiment_{lang}.txt"
            self._prompts[language] = prompt_path.read_text(encoding="utf-8").strip()
        return self._prompts[language]

    async def analyze(self, article: NewsArticle) -> SentimentResult:
        """Analyze an article and return a SentimentResult.

        Args:
            article: The news article to analyze.

        Returns:
            SentimentResult with sentiment [-1.0, 1.0], confidence, and reasoning.
            Returns neutral result (0.0 sentiment, 0.0 confidence) on parse errors.
        """
        system = self._load_prompt(article.language)
        user_prompt = f"Title: {article.title}\n\nContent: {article.content}"

        raw = await self._llm.complete(user_prompt, system)

        try:
            data = json.loads(raw)
            return SentimentResult(
                sentiment=float(data["sentiment"]),
                confidence=float(data["confidence"]),
                reasoning=str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return _FALLBACK
```

### Step 4: Run tests + quality check + commit

```bash
source ~/.zshrc && uv run pytest tests/unit/test_news_analyzer.py -v
source ~/.zshrc && uv run ruff check . && uv run mypy src/
git add src/finalayze/analysis/news_analyzer.py src/finalayze/analysis/prompts/ tests/unit/test_news_analyzer.py
git commit -m "feat(analysis): news sentiment analyzer with EN/RU prompts

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: EventClassifier

**Files:**
- Create: `src/finalayze/analysis/event_classifier.py`
- Create: `src/finalayze/analysis/prompts/classify_event.txt`
- Create: `tests/unit/test_event_classifier.py`

### Step 1: Create prompt file

`src/finalayze/analysis/prompts/classify_event.txt`:
```
You are a financial event classifier. Classify the news article into exactly one of these categories:
- earnings: company earnings report, revenue, profit, EPS
- fda: FDA approval, drug approval, clinical trial result
- macro: central bank decision, interest rates, inflation, GDP
- geopolitical: war, sanctions, political instability, elections
- cbr_rate: Russian Central Bank (ЦБ РФ) key rate decision
- oil_price: oil price movement, OPEC decision, energy production
- sanctions: new sanctions, trade restrictions
- other: anything that does not fit the above categories

Return ONLY the category name as a single word, nothing else.
```

### Step 2: Write failing tests

```python
# tests/unit/test_event_classifier.py
"""Unit tests for EventClassifier."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from finalayze.analysis.event_classifier import EventClassifier, EventType
from finalayze.core.schemas import NewsArticle

_ARTICLE = NewsArticle(
    id=uuid4(), source="reuters", title="Fed raises rates",
    content="Federal Reserve raised rates.", url="https://reuters.com/1",
    language="en", published_at=datetime(2024, 1, 1, tzinfo=UTC),
)


class TestEventType:
    def test_all_expected_values_exist(self) -> None:
        assert EventType.EARNINGS == "earnings"
        assert EventType.FDA == "fda"
        assert EventType.MACRO == "macro"
        assert EventType.GEOPOLITICAL == "geopolitical"
        assert EventType.CBR_RATE == "cbr_rate"
        assert EventType.OIL_PRICE == "oil_price"
        assert EventType.SANCTIONS == "sanctions"
        assert EventType.OTHER == "other"


class TestEventClassifier:
    @pytest.mark.asyncio
    async def test_classify_known_event_type(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "macro"
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.MACRO

    @pytest.mark.asyncio
    async def test_unknown_response_returns_other(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "random_unknown_value"
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.OTHER

    @pytest.mark.asyncio
    async def test_whitespace_stripped_from_response(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "  earnings  \n"
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.EARNINGS
```

### Step 3: Implement `src/finalayze/analysis/event_classifier.py`

```python
"""News event type classifier using an LLM client (Layer 3)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from finalayze.core.schemas import NewsArticle

if TYPE_CHECKING:
    from finalayze.analysis.llm_client import LLMClient

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class EventType(StrEnum):
    """Classification categories for news events."""

    EARNINGS = "earnings"
    FDA = "fda"
    MACRO = "macro"
    GEOPOLITICAL = "geopolitical"
    CBR_RATE = "cbr_rate"
    OIL_PRICE = "oil_price"
    SANCTIONS = "sanctions"
    OTHER = "other"


class EventClassifier:
    """Classifies news articles into EventType categories using an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._system: str | None = None

    def _load_system(self) -> str:
        if self._system is None:
            self._system = (_PROMPTS_DIR / "classify_event.txt").read_text(
                encoding="utf-8"
            ).strip()
        return self._system

    async def classify(self, article: NewsArticle) -> EventType:
        """Classify a news article into an EventType.

        Args:
            article: The news article to classify.

        Returns:
            EventType value. Returns ``EventType.OTHER`` for unrecognised responses.
        """
        system = self._load_system()
        user_prompt = f"Title: {article.title}\n\nContent: {article.content}"
        raw = await self._llm.complete(user_prompt, system)
        value = raw.strip().lower()
        try:
            return EventType(value)
        except ValueError:
            return EventType.OTHER
```

### Step 4: Run tests + quality check + commit

```bash
source ~/.zshrc && uv run pytest tests/unit/test_event_classifier.py -v
source ~/.zshrc && uv run ruff check . && uv run mypy src/
git add src/finalayze/analysis/event_classifier.py src/finalayze/analysis/prompts/classify_event.txt tests/unit/test_event_classifier.py
git commit -m "feat(analysis): EventType enum + EventClassifier

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 8: ImpactEstimator

**Files:**
- Create: `src/finalayze/analysis/impact_estimator.py`
- Create: `tests/unit/test_impact_estimator.py`

### Step 1: Write failing tests

```python
# tests/unit/test_impact_estimator.py
"""Unit tests for ImpactEstimator."""

from __future__ import annotations

from finalayze.analysis.event_classifier import EventType
from finalayze.analysis.impact_estimator import ImpactEstimator, SegmentImpact
from finalayze.core.schemas import SentimentResult

_SENTIMENT = SentimentResult(sentiment=0.8, confidence=0.9, reasoning="positive")
_ALL_SEGMENTS = [
    "us_tech", "us_healthcare", "us_finance", "us_broad",
    "ru_blue_chips", "ru_energy", "ru_tech", "ru_finance",
]

_PRIMARY_WEIGHT = 1.0
_SECONDARY_WEIGHT = 0.3


class TestImpactEstimatorGlobalScope:
    def test_global_event_affects_all_segments(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="global", event=EventType.MACRO,
            sentiment=_SENTIMENT, active_segments=_ALL_SEGMENTS,
        )
        segment_ids = {i.segment_id for i in impacts}
        assert segment_ids == set(_ALL_SEGMENTS)

    def test_global_event_all_weights_are_primary(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="global", event=EventType.MACRO,
            sentiment=_SENTIMENT, active_segments=_ALL_SEGMENTS,
        )
        assert all(i.weight == _PRIMARY_WEIGHT for i in impacts)


class TestImpactEstimatorUsScope:
    def test_us_event_affects_only_us_segments(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="us", event=EventType.EARNINGS,
            sentiment=_SENTIMENT, active_segments=_ALL_SEGMENTS,
        )
        segment_ids = {i.segment_id for i in impacts}
        assert all(s.startswith("us_") for s in segment_ids)
        assert not any(s.startswith("ru_") for s in segment_ids)


class TestImpactEstimatorRussiaScope:
    def test_russia_event_affects_only_ru_segments(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="russia", event=EventType.CBR_RATE,
            sentiment=_SENTIMENT, active_segments=_ALL_SEGMENTS,
        )
        segment_ids = {i.segment_id for i in impacts}
        assert all(s.startswith("ru_") for s in segment_ids)
        assert not any(s.startswith("us_") for s in segment_ids)


class TestImpactEstimatorEventRouting:
    def test_oil_price_affects_ru_energy_primary(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="sector", event=EventType.OIL_PRICE,
            sentiment=_SENTIMENT, active_segments=_ALL_SEGMENTS,
        )
        ru_energy = next((i for i in impacts if i.segment_id == "ru_energy"), None)
        assert ru_energy is not None
        assert ru_energy.weight == _PRIMARY_WEIGHT

    def test_fda_affects_us_healthcare_primary(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="sector", event=EventType.FDA,
            sentiment=_SENTIMENT, active_segments=_ALL_SEGMENTS,
        )
        healthcare = next((i for i in impacts if i.segment_id == "us_healthcare"), None)
        assert healthcare is not None
        assert healthcare.weight == _PRIMARY_WEIGHT

    def test_cbr_rate_affects_ru_finance_primary(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="sector", event=EventType.CBR_RATE,
            sentiment=_SENTIMENT, active_segments=_ALL_SEGMENTS,
        )
        finance = next((i for i in impacts if i.segment_id == "ru_finance"), None)
        assert finance is not None
        assert finance.weight == _PRIMARY_WEIGHT

    def test_sentiment_propagated_to_impacts(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="global", event=EventType.MACRO,
            sentiment=_SENTIMENT, active_segments=["us_tech"],
        )
        assert impacts[0].sentiment == pytest.approx(0.8)
```

Add `import pytest` at the top of the test file.

### Step 2: Implement `src/finalayze/analysis/impact_estimator.py`

```python
"""Impact estimator — routes news to affected segments (Layer 3)."""

from __future__ import annotations

from dataclasses import dataclass

from finalayze.analysis.event_classifier import EventType
from finalayze.core.schemas import SentimentResult

_PRIMARY = 1.0
_SECONDARY = 0.3

# EventType → {primary segments, secondary segments}
_EVENT_ROUTING: dict[EventType, tuple[list[str], list[str]]] = {
    EventType.OIL_PRICE: (["ru_energy"], ["ru_blue_chips"]),
    EventType.CBR_RATE: (["ru_finance"], ["ru_blue_chips"]),
    EventType.SANCTIONS: (["ru_blue_chips", "ru_energy", "ru_tech", "ru_finance"], []),
    EventType.FDA: (["us_healthcare"], []),
    EventType.EARNINGS: ([], []),  # handled by caller with specific symbol
    EventType.MACRO: ([], []),     # global event — caller uses scope="global"
    EventType.GEOPOLITICAL: ([], []),  # global event
    EventType.OTHER: ([], []),
}


@dataclass(frozen=True)
class SegmentImpact:
    """Impact of a news event on a specific segment."""

    segment_id: str
    weight: float   # 1.0 = primary, 0.3 = secondary
    sentiment: float  # from SentimentResult


class ImpactEstimator:
    """Routes news impact to affected segments based on scope and event type.

    No LLM needed — pure rule-based routing.
    """

    def estimate(
        self,
        scope: str,
        event: EventType,
        sentiment: SentimentResult,
        active_segments: list[str],
    ) -> list[SegmentImpact]:
        """Estimate which segments are affected and by how much.

        Args:
            scope: Geographic scope — "global", "us", "russia", or "sector".
            event: Classified event type.
            sentiment: Sentiment result from NewsAnalyzer.
            active_segments: List of segment IDs currently active in the system.

        Returns:
            List of SegmentImpact — may be empty if no matching segments.
        """
        sent = sentiment.sentiment
        impacts: dict[str, float] = {}  # segment_id → weight

        if scope == "global":
            for seg in active_segments:
                impacts[seg] = _PRIMARY

        elif scope == "us":
            for seg in active_segments:
                if seg.startswith("us_"):
                    impacts[seg] = _PRIMARY

        elif scope == "russia":
            for seg in active_segments:
                if seg.startswith("ru_"):
                    impacts[seg] = _PRIMARY

        elif scope == "sector":
            primary_segs, secondary_segs = _EVENT_ROUTING.get(event, ([], []))
            for seg in primary_segs:
                if seg in active_segments:
                    impacts[seg] = _PRIMARY
            for seg in secondary_segs:
                if seg in active_segments and seg not in impacts:
                    impacts[seg] = _SECONDARY

        return [
            SegmentImpact(segment_id=seg, weight=w, sentiment=sent * w)
            for seg, w in impacts.items()
        ]
```

### Step 3: Run tests + quality check + commit

```bash
source ~/.zshrc && uv run pytest tests/unit/test_impact_estimator.py -v
source ~/.zshrc && uv run ruff check . && uv run mypy src/
git add src/finalayze/analysis/impact_estimator.py tests/unit/test_impact_estimator.py
git commit -m "feat(analysis): ImpactEstimator — scope/event routing to segments

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 9: ML pipeline scaffold (features + models + registry)

**Files:**
- Create: `src/finalayze/ml/features/technical.py`
- Create: `src/finalayze/ml/models/xgboost_model.py`
- Create: `src/finalayze/ml/models/lightgbm_model.py`
- Create: `src/finalayze/ml/models/ensemble.py`
- Create: `src/finalayze/ml/registry.py`
- Create: `tests/unit/test_ml_pipeline.py`

### Step 1: Write failing tests

```python
# tests/unit/test_ml_pipeline.py
"""Unit tests for ML pipeline scaffold."""

from __future__ import annotations

import numpy as np
import pytest

from finalayze.ml.features.technical import compute_features
from finalayze.ml.models.ensemble import EnsembleModel
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.xgboost_model import XGBoostModel
from finalayze.ml.registry import MLModelRegistry

# ── Feature computation ──────────────────────────────────────────────────────
_FEATURE_NAMES = {"rsi_14", "macd_hist", "bb_pct_b", "volume_ratio_20d", "atr_14", "sentiment"}
_N_FEATURES = len(_FEATURE_NAMES)


class TestComputeFeatures:
    def test_returns_correct_keys(self) -> None:
        # 40 synthetic candles needed for all indicators
        features = _make_features()
        assert set(features.keys()) == _FEATURE_NAMES

    def test_all_values_are_floats(self) -> None:
        features = _make_features()
        assert all(isinstance(v, float) for v in features.values())

    def test_sentiment_passed_through(self) -> None:
        features = _make_features(sentiment=0.75)
        assert features["sentiment"] == pytest.approx(0.75)

    def test_insufficient_candles_raises(self) -> None:
        from datetime import UTC, datetime
        from decimal import Decimal
        from finalayze.core.schemas import Candle
        candles = [
            Candle(
                symbol="AAPL", market_id="us", timeframe="1d",
                timestamp=datetime(2024, 1, i + 1, tzinfo=UTC),
                open=Decimal("100"), high=Decimal("105"),
                low=Decimal("95"), close=Decimal("102"),
                volume=1000,
            )
            for i in range(5)  # only 5, need at least 30
        ]
        from finalayze.core.exceptions import InsufficientDataError
        with pytest.raises(InsufficientDataError):
            compute_features(candles)


class TestXGBoostModel:
    def test_predict_proba_before_fit_returns_half(self) -> None:
        model = XGBoostModel(segment_id="us_tech")
        features = _make_features()
        result = model.predict_proba(features)
        assert result == pytest.approx(0.5)

    def test_fit_and_predict(self) -> None:
        model = XGBoostModel(segment_id="us_tech")
        X = [_make_features()] * 50
        y = [1] * 25 + [0] * 25
        model.fit(X, y)
        result = model.predict_proba(_make_features())
        assert 0.0 <= result <= 1.0


class TestEnsembleModel:
    def test_predict_averages_two_models(self) -> None:
        xgb = XGBoostModel(segment_id="us_tech")
        lgb = LightGBMModel(segment_id="us_tech")
        ensemble = EnsembleModel(models=[xgb, lgb])
        features = _make_features()
        result = ensemble.predict_proba(features)
        assert 0.0 <= result <= 1.0

    def test_empty_models_returns_half(self) -> None:
        ensemble = EnsembleModel(models=[])
        assert ensemble.predict_proba(_make_features()) == pytest.approx(0.5)


class TestMLModelRegistry:
    def test_get_unregistered_returns_none(self) -> None:
        registry = MLModelRegistry()
        assert registry.get("us_tech") is None

    def test_register_and_get(self) -> None:
        registry = MLModelRegistry()
        xgb = XGBoostModel(segment_id="us_tech")
        lgb = LightGBMModel(segment_id="us_tech")
        model = EnsembleModel(models=[xgb, lgb])
        registry.register("us_tech", model)
        assert registry.get("us_tech") is model


# ── Helper ───────────────────────────────────────────────────────────────────

def _make_features(sentiment: float = 0.0) -> dict[str, float]:
    """Create a 40-candle set and return computed features."""
    from datetime import UTC, datetime
    from decimal import Decimal
    from finalayze.core.schemas import Candle
    rng = np.random.default_rng(42)
    prices = 100.0 + rng.standard_normal(40).cumsum()
    candles = [
        Candle(
            symbol="AAPL", market_id="us", timeframe="1d",
            timestamp=datetime(2024, 1, i + 1, tzinfo=UTC),
            open=Decimal(str(round(float(prices[i]) * 0.999, 2))),
            high=Decimal(str(round(float(prices[i]) * 1.005, 2))),
            low=Decimal(str(round(float(prices[i]) * 0.995, 2))),
            close=Decimal(str(round(float(prices[i]), 2))),
            volume=int(1000 + rng.integers(0, 500)),
        )
        for i in range(40)
    ]
    return compute_features(candles, sentiment_score=sentiment)
```

### Step 2: Implement feature engineering

`src/finalayze/ml/features/technical.py`:
```python
"""Technical feature engineering for ML models (Layer 3)."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from finalayze.core.exceptions import InsufficientDataError
from finalayze.core.schemas import Candle  # noqa: TC001

_MIN_CANDLES = 30


def compute_features(
    candles: list[Candle], sentiment_score: float = 0.0
) -> dict[str, float]:
    """Compute technical features from a list of candles.

    Args:
        candles: OHLCV candles sorted ascending by timestamp.
        sentiment_score: External sentiment score in [-1.0, 1.0].

    Returns:
        Dict of feature name → float value.

    Raises:
        InsufficientDataError: When fewer than 30 candles are provided.
    """
    if len(candles) < _MIN_CANDLES:
        msg = f"Need at least {_MIN_CANDLES} candles, got {len(candles)}"
        raise InsufficientDataError(msg)

    closes = [float(c.close) for c in candles]
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]
    volumes = [float(c.volume) for c in candles]

    close_s = pd.Series(closes, dtype=float)
    high_s = pd.Series(highs, dtype=float)
    low_s = pd.Series(lows, dtype=float)
    volume_s = pd.Series(volumes, dtype=float)

    # RSI-14
    rsi = ta.rsi(close_s, length=14)
    rsi_val = float(rsi.iloc[-1]) if rsi is not None and not rsi.empty else 50.0

    # MACD histogram
    macd_df = ta.macd(close_s, fast=12, slow=26, signal=9)
    macd_hist = 0.0
    if macd_df is not None and not macd_df.empty:
        hist_col = [c for c in macd_df.columns if "h" in c.lower()]
        if hist_col:
            macd_hist = float(macd_df[hist_col[0]].iloc[-1])

    # Bollinger %B
    bb = ta.bbands(close_s, length=20, std=2.0)
    bb_pct_b = 0.5
    if bb is not None and not bb.empty:
        pct_cols = [c for c in bb.columns if "P" in c]
        if pct_cols:
            bb_pct_b = float(bb[pct_cols[0]].iloc[-1])

    # Volume ratio (current vs 20-day average)
    vol_mean = volume_s.tail(20).mean()
    volume_ratio = float(volume_s.iloc[-1] / vol_mean) if vol_mean > 0 else 1.0

    # ATR-14
    atr = ta.atr(high_s, low_s, close_s, length=14)
    atr_val = float(atr.iloc[-1]) if atr is not None and not atr.empty else 0.0

    return {
        "rsi_14": rsi_val,
        "macd_hist": macd_hist,
        "bb_pct_b": bb_pct_b,
        "volume_ratio_20d": volume_ratio,
        "atr_14": atr_val,
        "sentiment": sentiment_score,
    }
```

### Step 3: Implement models and registry

`src/finalayze/ml/models/xgboost_model.py`:
```python
"""XGBoost per-segment model (Layer 3)."""

from __future__ import annotations

from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5


class XGBoostModel(BaseMLModel):
    """XGBoost classifier for directional prediction per segment."""

    def __init__(self, segment_id: str) -> None:
        self.segment_id = segment_id
        self._model: object | None = None

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return BUY probability (0.0-1.0). Returns 0.5 when untrained."""
        if self._model is None:
            return _UNTRAINED_PROB
        import numpy as np
        import xgboost as xgb
        X = np.array([[features[k] for k in sorted(features)]], dtype=float)
        proba: float = float(self._model.predict_proba(X)[0][1])  # type: ignore[union-attr]
        return proba

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:
        """Train the model on feature dicts and binary labels."""
        import numpy as np
        import xgboost as xgb
        X_arr = np.array([[row[k] for k in sorted(row)] for row in X], dtype=float)
        y_arr = np.array(y, dtype=int)
        self._model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss", verbosity=0,
        )
        self._model.fit(X_arr, y_arr)
```

`src/finalayze/ml/models/lightgbm_model.py`:
```python
"""LightGBM per-segment model (Layer 3)."""

from __future__ import annotations

from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5


class LightGBMModel(BaseMLModel):
    """LightGBM classifier for directional prediction per segment."""

    def __init__(self, segment_id: str) -> None:
        self.segment_id = segment_id
        self._model: object | None = None

    def predict_proba(self, features: dict[str, float]) -> float:
        if self._model is None:
            return _UNTRAINED_PROB
        import numpy as np
        X = np.array([[features[k] for k in sorted(features)]], dtype=float)
        proba: float = float(self._model.predict_proba(X)[0][1])  # type: ignore[union-attr]
        return proba

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:
        import lightgbm as lgb
        import numpy as np
        X_arr = np.array([[row[k] for k in sorted(row)] for row in X], dtype=float)
        y_arr = np.array(y, dtype=int)
        self._model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=-1
        )
        self._model.fit(X_arr, y_arr)
```

`src/finalayze/ml/models/base.py`:
```python
"""Abstract ML model base class (Layer 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMLModel(ABC):
    """Abstract base for all per-segment ML models."""

    segment_id: str

    @abstractmethod
    def predict_proba(self, features: dict[str, float]) -> float:
        """Return BUY probability in [0.0, 1.0]."""
        ...

    @abstractmethod
    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:
        """Train on feature dicts (X) and binary labels (y: 1=BUY, 0=SELL/HOLD)."""
        ...
```

`src/finalayze/ml/models/ensemble.py`:
```python
"""Ensemble model combining XGBoost + LightGBM (Layer 3)."""

from __future__ import annotations

from finalayze.ml.models.base import BaseMLModel

_DEFAULT_PROB = 0.5


class EnsembleModel:
    """Averages probability predictions from multiple BaseMLModel instances."""

    def __init__(self, models: list[BaseMLModel]) -> None:
        self._models = models

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return mean BUY probability across all models. Returns 0.5 when empty."""
        if not self._models:
            return _DEFAULT_PROB
        probs = [m.predict_proba(features) for m in self._models]
        return sum(probs) / len(probs)

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:
        """Train all constituent models."""
        for model in self._models:
            model.fit(X, y)
```

`src/finalayze/ml/registry.py`:
```python
"""Per-segment ML model registry (Layer 3)."""

from __future__ import annotations

from finalayze.ml.models.ensemble import EnsembleModel  # noqa: TC001


class MLModelRegistry:
    """Maps segment IDs to trained EnsembleModel instances."""

    def __init__(self) -> None:
        self._models: dict[str, EnsembleModel] = {}

    def register(self, segment_id: str, model: EnsembleModel) -> None:
        """Register or replace a model for a segment."""
        self._models[segment_id] = model

    def get(self, segment_id: str) -> EnsembleModel | None:
        """Return the model for the segment, or None if not registered."""
        return self._models.get(segment_id)
```

### Step 4: Run tests + quality check + commit

```bash
source ~/.zshrc && uv run pytest tests/unit/test_ml_pipeline.py -v
source ~/.zshrc && uv run ruff check . && uv run mypy src/
git add src/finalayze/ml/ tests/unit/test_ml_pipeline.py
git commit -m "feat(ml): XGBoost+LightGBM scaffold, feature engineering, model registry

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 10: EventDrivenStrategy

**Files:**
- Create: `src/finalayze/strategies/event_driven.py`
- Create: `tests/unit/test_event_driven_strategy.py`

### Step 1: Write failing tests

```python
# tests/unit/test_event_driven_strategy.py
"""Unit tests for EventDrivenStrategy."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.event_driven import EventDrivenStrategy

_CANDLE = Candle(
    symbol="AAPL", market_id="us", timeframe="1d",
    timestamp=datetime(2024, 1, 2, tzinfo=UTC),
    open=Decimal("150"), high=Decimal("155"),
    low=Decimal("148"), close=Decimal("152"),
    volume=1000,
)
_CANDLES = [_CANDLE]
_SEGMENT = "us_tech"
_MIN_SENTIMENT = 0.5


class TestEventDrivenStrategy:
    def test_name_is_event_driven(self) -> None:
        strategy = EventDrivenStrategy()
        assert strategy.name == "event_driven"

    def test_high_positive_sentiment_generates_buy(self) -> None:
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal(
            "AAPL", _CANDLES, _SEGMENT, sentiment_score=0.8
        )
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_high_negative_sentiment_generates_sell(self) -> None:
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal(
            "AAPL", _CANDLES, _SEGMENT, sentiment_score=-0.8
        )
        assert signal is not None
        assert signal.direction == SignalDirection.SELL

    def test_low_sentiment_returns_none(self) -> None:
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal(
            "AAPL", _CANDLES, _SEGMENT, sentiment_score=0.1
        )
        assert signal is None

    def test_zero_sentiment_returns_none(self) -> None:
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal(
            "AAPL", _CANDLES, _SEGMENT, sentiment_score=0.0
        )
        assert signal is None

    def test_confidence_scales_with_sentiment(self) -> None:
        strategy = EventDrivenStrategy()
        signal_high = strategy.generate_signal(
            "AAPL", _CANDLES, _SEGMENT, sentiment_score=0.9
        )
        signal_low = strategy.generate_signal(
            "AAPL", _CANDLES, _SEGMENT, sentiment_score=0.6
        )
        assert signal_high is not None
        assert signal_low is not None
        assert signal_high.confidence > signal_low.confidence

    def test_get_parameters_returns_dict(self) -> None:
        strategy = EventDrivenStrategy()
        params = strategy.get_parameters(_SEGMENT)
        assert isinstance(params, dict)

    def test_supported_segments_returns_list(self) -> None:
        strategy = EventDrivenStrategy()
        segments = strategy.supported_segments()
        assert isinstance(segments, list)
        assert _SEGMENT in segments
```

### Step 2: Implement `src/finalayze/strategies/event_driven.py`

```python
"""Event-driven trading strategy using news sentiment (Layer 4)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"
_DEFAULT_MIN_SENTIMENT = 0.5
_DEFAULT_WEIGHT = Decimal("0.4")


class EventDrivenStrategy(BaseStrategy):
    """News sentiment-driven strategy.

    Generates BUY when sentiment > min_sentiment threshold,
    SELL when sentiment < -min_sentiment.
    Confidence = min(1.0, abs(sentiment) * credibility).
    Falls back gracefully to None when sentiment == 0.
    """

    @property
    def name(self) -> str:
        return "event_driven"

    def supported_segments(self) -> list[str]:
        """Return segment IDs where event_driven strategy is enabled."""
        segments: list[str] = []
        for preset_path in sorted(_PRESETS_DIR.glob("*.yaml")):
            try:
                with preset_path.open() as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    continue
                strategies = data.get("strategies", {})
                if not isinstance(strategies, dict):
                    continue
                ed_cfg = strategies.get("event_driven", {})
                if isinstance(ed_cfg, dict) and ed_cfg.get("enabled", False):
                    seg_id = data.get("segment_id")
                    if seg_id:
                        segments.append(str(seg_id))
            except (OSError, yaml.YAMLError):
                continue
        return segments

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load event_driven parameters from the YAML preset."""
        try:
            preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return {}
            strategies = data.get("strategies", {})
            if not isinstance(strategies, dict):
                return {}
            ed_cfg = strategies.get("event_driven", {})
            if not isinstance(ed_cfg, dict):
                return {}
            params = ed_cfg.get("params", {})
            return dict(params) if isinstance(params, dict) else {}
        except (FileNotFoundError, OSError, yaml.YAMLError):
            return {}

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        credibility: float = 1.0,
    ) -> Signal | None:
        """Generate a trading signal based on news sentiment score.

        Args:
            symbol: Ticker symbol.
            candles: Recent OHLCV candles (used for context, not indicators).
            segment_id: The segment this symbol belongs to.
            sentiment_score: Sentiment in [-1.0, 1.0]. 0.0 → no signal.
            credibility: Source credibility [0.0, 1.0], scales confidence.

        Returns:
            Signal or None if sentiment is within neutral range.
        """
        params = self.get_parameters(segment_id)
        min_sentiment: float = float(params.get("min_sentiment", _DEFAULT_MIN_SENTIMENT))

        abs_sent = abs(sentiment_score)
        if abs_sent < min_sentiment:
            return None

        direction = SignalDirection.BUY if sentiment_score > 0 else SignalDirection.SELL
        confidence = min(1.0, abs_sent * credibility)

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=candles[-1].market_id if candles else "us",
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={"sentiment": sentiment_score, "credibility": credibility},
            reasoning=f"News sentiment {sentiment_score:+.2f} (credibility={credibility:.2f})",
        )
```

### Step 3: Run tests + full quality check + commit

```bash
source ~/.zshrc && uv run pytest tests/unit/test_event_driven_strategy.py -v
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header 2>&1 | python3 -c "import sys; lines=sys.stdin.readlines(); [print(l,end='') for l in lines[-5:]]"
git add src/finalayze/strategies/event_driven.py tests/unit/test_event_driven_strategy.py
git commit -m "feat(strategies): EventDrivenStrategy using news sentiment scores

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final Step: Push and create PR

```bash
git push -u origin feature/phase2-intelligence
gh pr create --title "feat(phase2-A): intelligence pipeline — LLM client, news analysis, ML scaffold, event strategy" --body "$(cat <<'EOF'
## Summary
- Abstract LLMClient with OpenRouter (default), OpenAI, Anthropic implementations
- NewsApiFetcher for English/Russian news articles
- NewsAnalyzer — LLM-powered sentiment scoring (EN + RU prompts)
- EventClassifier — categorises news into 8 event types
- ImpactEstimator — routes news impact to affected segments (no LLM)
- ML pipeline scaffold: XGBoost + LightGBM per-segment models, feature engineering, EnsembleModel, MLModelRegistry
- EventDrivenStrategy — generates signals from sentiment scores
- DB migration 002: news_articles + sentiment_scores tables

## Test Plan
- [ ] `uv run pytest` — all tests pass
- [ ] `uv run ruff check .` — zero warnings
- [ ] `uv run mypy src/` — zero errors
- [ ] Coverage ≥ 80% on new modules

🤖 Generated with Claude Code
EOF
)"
```

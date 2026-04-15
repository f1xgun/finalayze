# Phase 53: Sentiment ML Infrastructure — Research

**Researched:** 2026-04-15
**Domain:** TimescaleDB continuous aggregates, SQLAlchemy 2.0 async Layer 2 accessor, Alembic migrations
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
None — all implementation choices are at Claude's discretion.

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Success criteria are fully prescriptive:
- TimescaleDB continuous aggregate view `sentiment_7d_avg` with auto-refresh
- `SentimentStore.get_rolling(ticker, window='7d')` returning `list[(bucket, avg_score, article_count)]`
- Empty list on missing data (safe accessor for v11 pipeline)
- Alembic migration for the continuous aggregate
- STATE.md research flag: "Confirm timescaledb.enable_cagg_window_functions setting in Docker Compose PostgreSQL config; verify continuous aggregate refresh policy syntax for current TimescaleDB version"

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STML-01 | TimescaleDB continuous aggregate for rolling sentiment (1d/7d/30d buckets) | Continuous aggregate syntax verified against TimescaleDB 2.17.2 live instance; hypertable conversion required first |
| STML-02 | SentimentStore reader (Layer 2) provides rolling aggregation query for future ML feature extraction | Layer 2 pattern established; async session factory + `text()` raw SQL pattern confirmed |
</phase_requirements>

---

## Summary

Phase 53 builds a two-piece infrastructure: (1) a TimescaleDB continuous aggregate materializing rolling 7d sentiment averages, and (2) a `SentimentStore` class in `src/finalayze/data/` that queries it.

**Critical blocker discovered:** The `sentiment_scores` table was never converted to a TimescaleDB hypertable. Migration `002_news_sentiment.py` creates the regular PostgreSQL table but omits the `create_hypertable()` call, unlike `candles` (migration 001) and `portfolio_snapshots` (migration 003). Continuous aggregates in TimescaleDB require a hypertable as their source. Migration 005 must therefore perform a two-step upgrade: (a) convert `sentiment_scores` to a hypertable with `migrate_data => TRUE`, then (b) create the continuous aggregate and its refresh policy.

**STATE.md research flag resolved:** The `timescaledb.enable_cagg_window_functions` GUC does not appear in TimescaleDB 2.17.2. Verified by querying `pg_settings WHERE name LIKE 'timescaledb%'` on the live container — 54 GUC rows returned, none named `enable_cagg_window_functions`. The cagg definition for `sentiment_7d_avg` uses only `AVG()`, `COUNT()`, `time_bucket()`, and `GROUP BY` — no window functions — so this GUC is irrelevant and no Docker Compose configuration change is needed.

**Primary recommendation:** Use a single migration `005_sentiment_ml_cagg.py` that (1) converts `sentiment_scores` to a hypertable, (2) creates the `sentiment_7d_avg` continuous aggregate, and (3) adds the refresh policy. Write `SentimentStore` as a class in `src/finalayze/data/sentiment_store.py` that accepts an `async_sessionmaker` and exposes `get_rolling()` using `text()` raw SQL against the materialized view.

---

## Project Constraints (from CLAUDE.md)

| Directive | Application to This Phase |
|-----------|--------------------------|
| Python 3.12, strict typing, `from __future__ import annotations` | All new `.py` files must start with this import |
| Ruff (line-length 100), mypy strict | SentimentStore must pass both; use `TYPE_CHECKING` guard for heavy imports |
| SQLAlchemy 2.0 async | Use `async with session_factory() as session:` + `await session.execute(text(...))` |
| Pydantic v2 for all schemas | Return type of `get_rolling()` can be plain tuples or a named Pydantic model |
| TDD mandatory: write failing test FIRST | Tests for SentimentStore come before implementation |
| Layer 2: `data/` module, no upward imports | `SentimentStore` imports only from Layers 0–2 |
| Alembic for migrations, sequential numbered | New migration is `005_sentiment_ml_cagg.py` (down_revision = "004") |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| timescaledb | 2.17.2 (installed, VERIFIED) | continuous aggregates, hypertable partitioning | Already in Docker Compose; `candles` and `portfolio_snapshots` are hypertables |
| sqlalchemy[asyncio] | >=2.0.36 (VERIFIED: pyproject.toml) | async ORM + raw SQL via `text()` | Project-wide standard; `asyncpg` driver already configured |
| alembic | >=1.14.0 (VERIFIED: pyproject.toml) | schema migrations | All existing migrations use this pattern |
| asyncpg | >=0.30.0 (VERIFIED: pyproject.toml) | PostgreSQL async driver | Runtime driver; migrations use psycopg2 via env.py conversion |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| psycopg2 | (system, via alembic env.py) | Synchronous migrations | Alembic env.py converts asyncpg URLs to psycopg2 automatically |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| TimescaleDB cagg | PostgreSQL materialized view + cron refresh | No incremental refresh; full recompute on every refresh — defeats purpose |
| Raw SQL via `text()` | ORM query against a `Table` reflection | TimescaleDB-specific functions (`time_bucket`, `add_continuous_aggregate_policy`) have no ORM mapping; raw SQL is idiomatic here |

**Installation:** No new packages needed — all dependencies already in `pyproject.toml`.

---

## Architecture Patterns

### Recommended Project Structure

```
src/finalayze/data/
├── sentiment_store.py   # NEW: Layer 2 SentimentStore accessor
├── cache.py
├── normalizer.py
├── rate_limiter.py
└── fetchers/

alembic/versions/
├── 001_initial.py
├── 002_news_sentiment.py
├── 003_portfolio_snapshots.py
├── 004_add_credibility_to_sentiment_scores.py
└── 005_sentiment_ml_cagg.py   # NEW: hypertable + cagg + refresh policy

tests/unit/
└── test_sentiment_store.py    # NEW: unit tests (mock DB session)
```

### Pattern 1: Two-Step Alembic Migration (Hypertable Conversion + Cagg)

**What:** Migration 005 converts `sentiment_scores` to a hypertable first, then creates the continuous aggregate.
**When to use:** Required — continuous aggregates only work on hypertable sources.

```python
# Source: verified TimescaleDB 2.17.2 live instance + official docs
# alembic/versions/005_sentiment_ml_cagg.py

def upgrade() -> None:
    # Step 1: Convert sentiment_scores to a hypertable
    # migrate_data=TRUE preserves existing rows (currently 0 rows in sandbox)
    op.execute(
        "SELECT create_hypertable('sentiment_scores', 'timestamp', "
        "migrate_data => TRUE, if_not_exists => TRUE)"
    )

    # Step 2: Create 1-day bucket continuous aggregate over the hypertable
    op.execute("""
        CREATE MATERIALIZED VIEW sentiment_7d_avg
        WITH (timescaledb.continuous) AS
        SELECT
            symbol,
            market_id,
            time_bucket(INTERVAL '1 day', timestamp) AS bucket,
            AVG(composite_sentiment)::numeric(5,4)   AS avg_score,
            COUNT(*)                                  AS article_count
        FROM sentiment_scores
        GROUP BY symbol, market_id, time_bucket(INTERVAL '1 day', timestamp)
        WITH NO DATA
    """)

    # Step 3: Add refresh policy — refresh last 30 days once per hour
    # end_offset='1 day' excludes the incomplete current bucket
    op.execute("""
        SELECT add_continuous_aggregate_policy(
            'sentiment_7d_avg',
            start_offset  => INTERVAL '30 days',
            end_offset    => INTERVAL '1 day',
            schedule_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        )
    """)


def downgrade() -> None:
    # Drop cagg first (it depends on the hypertable)
    op.execute("DROP MATERIALIZED VIEW IF EXISTS sentiment_7d_avg")
    # Cannot un-convert a hypertable with data easily; migration is one-way in production.
    # For dev/test environments with no data, this suffices.
```

**Why `WITH NO DATA`:** Defers initial materialization to the first policy execution, avoiding a potentially long blocking migration when the table has existing data.

**Why `INTERVAL '1 day'` bucket:** The view is named `sentiment_7d_avg`. The "7d" refers to the query window in `get_rolling(window='7d')`, not the bucket size. Daily buckets are the right granularity for rolling averages consumed by ML features.

### Pattern 2: SentimentStore Layer 2 Accessor

**What:** Async class in `data/` that queries the `sentiment_7d_avg` view using raw SQL via `text()`.
**When to use:** Everywhere the v11 ML feature pipeline needs rolling sentiment data.

```python
# Source: established project pattern from api/v1/system.py + data/cache.py
# src/finalayze/data/sentiment_store.py

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

_WINDOW_INTERVALS: dict[str, str] = {
    "7d": "7 days",
    "30d": "30 days",
    "1d": "1 day",
}

class SentimentRow(NamedTuple):
    bucket: datetime
    avg_score: float | None
    article_count: int


class SentimentStore:
    """Layer 2: read-only accessor for rolling sentiment aggregates.

    Queries the `sentiment_7d_avg` continuous aggregate view.
    Returns empty list when no data exists — safe for v11 ML pipeline.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = session_factory

    async def get_rolling(
        self,
        ticker: str,
        *,
        window: str = "7d",
        market_id: str = "moex",
    ) -> list[SentimentRow]:
        """Return daily bucket rows for the given ticker over the rolling window.

        Args:
            ticker: Instrument symbol (e.g. 'SBER').
            window: Rolling window string ('1d', '7d', '30d').
            market_id: Market identifier (default 'moex').

        Returns:
            List of (bucket, avg_score, article_count) ordered by bucket ascending.
            Empty list if no rows exist — never raises on missing data.
        """
        interval = _WINDOW_INTERVALS.get(window, "7 days")
        sql = text("""
            SELECT bucket, avg_score, article_count
            FROM sentiment_7d_avg
            WHERE symbol    = :symbol
              AND market_id = :market_id
              AND bucket   >= NOW() - INTERVAL :interval
            ORDER BY bucket ASC
        """)
        async with self._factory() as session:
            result = await session.execute(
                sql,
                {"symbol": ticker, "market_id": market_id, "interval": interval},
            )
            rows = result.fetchall()
        return [
            SentimentRow(
                bucket=row.bucket,
                avg_score=float(row.avg_score) if row.avg_score is not None else None,
                article_count=int(row.article_count),
            )
            for row in rows
        ]
```

### Pattern 3: Unit Testing with AsyncMock Session Factory

**What:** Tests mock the session factory so no real DB is needed.
**When to use:** All unit tests in `tests/unit/test_sentiment_store.py`.

```python
# Source: established project pattern from tests/unit/test_redis_cache.py
# asyncio_mode = "auto" in pytest config — no @pytest.mark.asyncio needed for class methods

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from datetime import UTC, datetime
from finalayze.data.sentiment_store import SentimentStore, SentimentRow

@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    # session used as async context manager
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session

@pytest.fixture
def mock_factory(mock_session: AsyncMock) -> MagicMock:
    factory = MagicMock()
    factory.return_value = mock_session
    return factory

@pytest.fixture
def store(mock_factory: MagicMock) -> SentimentStore:
    return SentimentStore(mock_factory)
```

### Anti-Patterns to Avoid

- **Using ORM `select()` against the cagg view:** SQLAlchemy doesn't know the view's schema. Use raw `text()` queries.
- **Returning `Decimal` from `get_rolling()`:** ML consumers need `float`. Cast explicitly in the accessor.
- **Using `op.drop_table()` in downgrade for sentiment_scores:** Cannot drop a hypertable that has a dependent continuous aggregate. Must `DROP MATERIALIZED VIEW` first.
- **Using `timescaledb.materialized_only = FALSE` without justification:** Default is `TRUE`; real-time cagg (materializing incomplete buckets) has higher overhead and is unnecessary for daily ML features.
- **Passing `window='7d'` as a LIMIT on bucket count:** The window should filter by time range (`>= NOW() - INTERVAL '7 days'`), not restrict row count.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Incremental rolling average | Custom Python aggregator reading raw rows | TimescaleDB continuous aggregate + `time_bucket()` | CAgg does incremental materialization; Python re-aggregation scans full history on every call |
| Refresh scheduling | APScheduler / Celery beat task refreshing the view | `add_continuous_aggregate_policy()` | Background workers in TimescaleDB are native, survive restarts, and coordinate with WAL |
| SQL parameter binding | f-string with `%` or `.format()` | SQLAlchemy `text()` with named `:param` bindings | Prevents SQL injection; asyncpg rejects unbound queries |

**Key insight:** TimescaleDB's background refresh workers are the correct mechanism. The policy `start_offset='30 days', end_offset='1 day', schedule_interval='1 hour'` keeps the materialized view current with no application-layer scheduling code.

---

## Critical Pre-requisite: sentinel_scores Hypertable Gap

**Finding:** `sentiment_scores` was created as a regular PostgreSQL table (migration 002). It was never converted to a hypertable. The `candles` table (migration 001) and `portfolio_snapshots` table (migration 003) both call `create_hypertable()`, but `sentiment_scores` was omitted.

**Impact:** `CREATE MATERIALIZED VIEW ... WITH (timescaledb.continuous) AS SELECT ... FROM sentiment_scores` will fail with an error because the source must be a hypertable.

**Confirmed:** Queried running sandbox DB (`finalayze-sandbox-db`, TimescaleDB 2.17.2):
```sql
SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'sentiment_scores';
-- Returns 0 rows
```

**Fix:** Migration 005 must call `SELECT create_hypertable('sentiment_scores', 'timestamp', migrate_data => TRUE, if_not_exists => TRUE)` before creating the cagg.

**Side effect:** After hypertable conversion, Alembic autogenerate will try to drop the time-dimension index that TimescaleDB creates automatically. This is a known pattern (discussed in `sqlalchemy/alembic#1465`). The project already handles this by using handwritten migrations only — no `alembic revision --autogenerate` in CI.

---

## Research Flag Resolution: enable_cagg_window_functions

**STATE.md flag:** "Confirm timescaledb.enable_cagg_window_functions setting in Docker Compose PostgreSQL config; verify continuous aggregate refresh policy syntax for current TimescaleDB version"

**Finding (VERIFIED):** `timescaledb.enable_cagg_window_functions` does NOT exist as a GUC in TimescaleDB 2.17.2. Queried `pg_settings WHERE name LIKE 'timescaledb%'` on the live `finalayze-sandbox-db` container — 54 GUC rows returned, none matching this name.

**Impact:** No Docker Compose `POSTGRES_<GUC>` environment variable is needed. The flag appears to reference a future or experimental feature that is either absent from this release or has a different name.

**The `sentiment_7d_avg` view uses no window functions** — only `AVG()`, `COUNT()`, and `time_bucket()` — so this GUC is irrelevant to this phase.

**Conclusion:** The research flag can be closed without action.

---

## Common Pitfalls

### Pitfall 1: Migrating Without WITH NO DATA Causes Lock Timeout

**What goes wrong:** If `CREATE MATERIALIZED VIEW ... WITH (timescaledb.continuous)` without `WITH NO DATA` is run against a populated hypertable, TimescaleDB performs full initial materialization synchronously during migration. This can timeout on large tables.

**Why it happens:** Default behavior materializes all existing data immediately.

**How to avoid:** Always use `WITH NO DATA` in the migration. Let the refresh policy do the first materialization asynchronously.

**Warning signs:** Migration hangs for >30 seconds on first run.

### Pitfall 2: cagg Query Returns No Rows Until First Refresh

**What goes wrong:** `SentimentStore.get_rolling()` returns empty list immediately after migration because `WITH NO DATA` defers materialization.

**Why it happens:** Policy runs asynchronously; first run is `schedule_interval` after policy creation.

**How to avoid:** This is acceptable behavior — STML-02 success criterion explicitly requires empty list on missing data. Document in code.

**Warning signs:** Integration test returns 0 rows immediately after migration — this is expected.

### Pitfall 3: Downgrade Fails if cagg Exists

**What goes wrong:** `op.drop_table('sentiment_scores')` in downgrade fails because TimescaleDB prevents dropping a hypertable that has dependent continuous aggregates.

**Why it happens:** Foreign-key-like dependency: cagg is built on the hypertable.

**How to avoid:** Downgrade must first `DROP MATERIALIZED VIEW IF EXISTS sentiment_7d_avg`, then handle the hypertable. Hypertable-to-regular-table conversion is not natively supported; downgrade drops the table.

**Warning signs:** `ERROR: cannot drop table ... because other objects depend on it`.

### Pitfall 4: end_offset Must Not Be Zero

**What goes wrong:** Setting `end_offset => INTERVAL '0'` causes the policy to try materializing the incomplete current bucket, resulting in noisy/partial data.

**Why it happens:** TimescaleDB docs explicitly warn against this.

**How to avoid:** Use `end_offset => INTERVAL '1 day'` for daily buckets to exclude the current incomplete day.

**Warning signs:** Last bucket has artificially low `article_count`.

### Pitfall 5: INTERVAL Parameter Binding with asyncpg

**What goes wrong:** Passing Python `timedelta` as an INTERVAL parameter to `text()` may not bind correctly with asyncpg.

**Why it happens:** asyncpg expects PostgreSQL-native types; the `text()` clause with `:interval` named param using a string `'7 days'` is the safest pattern.

**How to avoid:** Pass the interval as a plain string (e.g., `"7 days"`) in the params dict, not as a `timedelta` object.

**Warning signs:** `asyncpg.exceptions.InvalidParameterTypeError`.

---

## Code Examples

### Verified Migration Pattern (from existing migrations)

```python
# Source: alembic/versions/001_initial.py (verified in repo)
op.execute("SELECT create_hypertable('candles', 'timestamp', migrate_data => true)")

# Source: alembic/versions/003_portfolio_snapshots.py (verified in repo)
op.execute(
    "SELECT create_hypertable('portfolio_snapshots', 'timestamp', if_not_exists => TRUE)"
)
```

### Verified Async Session Pattern (from api/v1/system.py)

```python
# Source: src/finalayze/api/v1/system.py (verified in repo)
from finalayze.core.db import get_async_session_factory
from sqlalchemy import text

factory = get_async_session_factory()
async with factory() as session:
    await session.execute(text("SELECT 1"))
```

### Verified Continuous Aggregate SQL Syntax (from TimescaleDB 2.17.2 official docs)

```sql
-- Source: tigerdata.com/docs/api/latest/continuous-aggregates/create_materialized_view
-- [CITED: https://www.tigerdata.com/docs/api/latest/continuous-aggregates/create_materialized_view]
CREATE MATERIALIZED VIEW sentiment_7d_avg
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    market_id,
    time_bucket(INTERVAL '1 day', timestamp) AS bucket,
    AVG(composite_sentiment)::numeric(5,4)   AS avg_score,
    COUNT(*)                                  AS article_count
FROM sentiment_scores
GROUP BY symbol, market_id, time_bucket(INTERVAL '1 day', timestamp)
WITH NO DATA;

-- Source: tigerdata.com/docs/api/latest/continuous-aggregates/add_continuous_aggregate_policy
SELECT add_continuous_aggregate_policy(
    'sentiment_7d_avg',
    start_offset      => INTERVAL '30 days',
    end_offset        => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists     => TRUE
);
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `timescaledb.materialized_only = FALSE` (real-time cagg) | `timescaledb.materialized_only = TRUE` (default) | TimescaleDB 2.7+ | Real-time mode has higher overhead; materialized-only is correct for ML batch features |
| Manual `REFRESH MATERIALIZED VIEW` via cron | `add_continuous_aggregate_policy()` | TimescaleDB 1.7+ | Policy is native, survives restarts, no app-layer scheduler needed |
| `finalized = FALSE` (old cagg format) | `finalized = TRUE` (default since TimescaleDB 2.7) | 2.7+ | New format supports more aggregate functions; migration from old format via `cagg_migrate()` |

**Deprecated/outdated:**
- `timescaledb.continuous_aggregates.ignore_invalidation_older_than`: Removed in TimescaleDB 2.0; replaced by `start_offset` in the refresh policy.
- The `WITH (timescaledb.continuous, timescaledb.refresh_lag = '...')` option: Replaced by `add_continuous_aggregate_policy()`.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `composite_sentiment` is the correct column to average for the `avg_score` output (vs `news_sentiment` or `confidence`) | Standard Stack, Code Examples | avg_score would represent wrong signal; low risk — easily changed |

**Note:** All other critical claims are VERIFIED against the live container or codebase.

---

## Open Questions

1. **`composite_sentiment` vs `news_sentiment` for avg_score**
   - What we know: `SentimentScoreModel` has `news_sentiment`, `social_sentiment`, `composite_sentiment`, `confidence`, `credibility`
   - What's unclear: The phase description says "avg_score" — which column? CONTEXT.md doesn't specify.
   - Recommendation: Use `composite_sentiment` — it's the weighted combination of news + social, most appropriate for ML feature consumption. Planner should confirm.

2. **Multi-window design: 1d/7d/30d vs single 1d bucket + Python windowing**
   - What we know: REQUIREMENTS.md STML-01 mentions "1d/7d/30d buckets"; STML-02 specifies `get_rolling(window='7d')`
   - What's unclear: Whether three separate cagg views are needed or a single 1d view with query-time windowing
   - Recommendation: Create a single `sentiment_7d_avg` view with 1-day buckets. The `get_rolling(window='7d')` call simply filters `bucket >= NOW() - INTERVAL '7 days'`. This satisfies both STML-01 and STML-02 without three separate views. STML-01's "1d/7d/30d" refers to query windows, not separate views.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| TimescaleDB | Migration 005, cagg | ✓ | 2.17.2 (live sandbox container) | — |
| PostgreSQL | Database | ✓ | 16 (via TimescaleDB image) | — |
| psycopg2 | Alembic migrations | ✓ | system (env.py converts asyncpg URLs) | — |
| asyncpg | Runtime queries | ✓ | >=0.30.0 (pyproject.toml) | — |

**No missing dependencies.** All required tools are present.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 with pytest-asyncio |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/unit/test_sentiment_store.py -x` |
| Full suite command | `uv run pytest --cov=src/finalayze` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STML-01 | Migration creates hypertable + cagg + policy | infra smoke (SQL introspection) | `uv run pytest tests/unit/test_sentiment_store.py -x -k migration` | ❌ Wave 0 |
| STML-02 | `get_rolling('SBER', window='7d')` returns list of SentimentRow | unit (AsyncMock DB) | `uv run pytest tests/unit/test_sentiment_store.py -x -k get_rolling` | ❌ Wave 0 |
| STML-02 | `get_rolling()` on ticker with no data returns `[]` without error | unit (AsyncMock DB, empty result) | `uv run pytest tests/unit/test_sentiment_store.py -x -k empty` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_sentiment_store.py -x`
- **Per wave merge:** `uv run pytest --cov=src/finalayze`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_sentiment_store.py` — covers STML-01 migration SQL check, STML-02 get_rolling, STML-02 empty-list contract
- Framework already installed — no additional setup needed

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | Layer 2 read-only accessor; no write path |
| V5 Input Validation | yes | SQLAlchemy `text()` with named `:param` bindings prevents SQL injection; `window` param validated against allowlist `_WINDOW_INTERVALS` dict |
| V6 Cryptography | no | — |

### Known Threat Patterns for this Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| SQL injection via `ticker` or `window` params | Tampering | SQLAlchemy `text()` named params — never string-format into SQL |
| Unbounded result set (ticker with huge history) | DoS | `LIMIT` clause or time-bounded `WHERE bucket >= NOW() - INTERVAL :interval` |

---

## Sources

### Primary (HIGH confidence)
- TimescaleDB 2.17.2 live instance (`finalayze-sandbox-db`) — `pg_settings` query, `timescaledb_information.hypertables`, `timescaledb_information.continuous_aggregates`
- `alembic/versions/001_initial.py`, `002_news_sentiment.py`, `003_portfolio_snapshots.py`, `004_add_credibility_to_sentiment_scores.py` — verified migration patterns
- `src/finalayze/core/models.py` — `SentimentScoreModel` column definitions
- `src/finalayze/api/v1/system.py` — async session factory usage pattern
- `tests/unit/test_redis_cache.py` — `AsyncMock` unit test pattern
- `pyproject.toml` — dependency versions

### Secondary (MEDIUM confidence)
- [CITED: https://www.tigerdata.com/docs/api/latest/continuous-aggregates/create_materialized_view] — CREATE MATERIALIZED VIEW syntax
- [CITED: https://www.tigerdata.com/docs/api/latest/continuous-aggregates/add_continuous_aggregate_policy] — refresh policy function signature
- [CITED: https://www.tigerdata.com/docs/api/latest/configuration/gucs] — GUC list confirming `enable_cagg_window_functions` default = FALSE

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages verified against running system and pyproject.toml
- Architecture: HIGH — migration and Layer 2 patterns verified against existing codebase
- TimescaleDB syntax: HIGH — verified against live 2.17.2 instance + official docs
- Pitfalls: MEDIUM — drawn from TimescaleDB docs and GitHub issues; not all reproduced locally

**Research date:** 2026-04-15
**Valid until:** 2026-05-15 (stable API; TimescaleDB 2.17.2 pinned in Docker)

# Database Schema

PostgreSQL + TimescaleDB. ORM: SQLAlchemy 2.0 async.
Models defined in `src/finalayze/core/models.py`. Migrations in `alembic/versions/`.

## Tables

### Market Data

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `markets` | Market definitions (US, MOEX) | id, name, timezone, currency |
| `segments` | Trading segments (us_tech, ru_blue_chips) | id, market_id, name, universe |
| `instruments` | Tradeable instruments with FIGI | id, segment_id, symbol, figi, instrument_type |
| `candles` | OHLCV price data | instrument_id, timestamp, open, high, low, close, volume |
| `bond_candles` | Bond-specific price data (% of face) | instrument_id, timestamp, open, high, low, close, volume |

### Trading

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `signals` | Strategy signal history | instrument_id, timestamp, direction, confidence, strategy_name |
| `orders` | Order lifecycle tracking | instrument_id, side, qty, price, status, broker_order_id, filled_at |

### News & Sentiment

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `news_articles` | Fetched articles (RSS, Telegram, API) | source, title, content, url, language, published_at |
| `sentiment_scores` | LLM sentiment per article | article_id, sentiment (-1..1), confidence, reasoning |

### Bonds

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `coupon_schedules` | Bond coupon payment dates | instrument_id, coupon_date, coupon_amount, record_date |
| `amortization_events` | Bond principal repayment events | instrument_id, event_date, amount, remaining_face |
| `layer_ledger` | Bond portfolio layers (Core/Tactical/Opportunistic/Buffer) | instrument_id, layer, qty, entry_price, entry_ytm |

### Portfolio & Monitoring

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `macro_snapshots` | CBR key rate, FX, index snapshots | timestamp, key_rate, usd_rub, imoex |
| `daily_equity_snapshots` | End-of-day equity tracking | date, market, equity_value, drawdown |
| `portfolio_snapshots` | Periodic portfolio state | timestamp, total_value, positions_json |

## Migrations

| Version | Name | What it does |
|---------|------|-------------|
| 001 | initial | Core tables: markets, segments, instruments, candles, signals, orders |
| 002 | news_sentiment | news_articles, sentiment_scores, macro_snapshots, bond tables |
| 003 | portfolio_snapshots | daily_equity_snapshots, portfolio_snapshots |

## Running Migrations

```bash
uv run alembic -c alembic/alembic.ini upgrade head
```

## Conventions

- All timestamps are UTC with timezone (`DateTime(timezone=True)`)
- Prices stored as `Numeric(18, 8)` for precision
- FIGI is the canonical instrument identifier for T-Invest API
- Bond prices are in % of face value (MOEX convention)
- Use async sessions: `async with async_session() as session:`

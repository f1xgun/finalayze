# Deployment Guide

## Prerequisites

- Docker Engine 24+ and Docker Compose v2
- At least 2 GB RAM and 10 GB disk
- Valid API keys for Finnhub, NewsAPI, and broker accounts (Alpaca and/or Tinkoff)
- A strong random `FINALAYZE_API_KEY` for REST API authentication
- A `REDIS_PASSWORD` for production Redis

## Environment Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Fill in all required values in `.env`:
   - `FINALAYZE_MODE` — start with `sandbox` for testing, `real` for live trading
   - `FINALAYZE_API_KEY` — strong random string (e.g. `openssl rand -hex 32`)
   - `REDIS_PASSWORD` — strong random string for Redis auth
   - `POSTGRES_PASSWORD` — database password (default: `secret`)
   - Broker credentials (`FINALAYZE_ALPACA_API_KEY`, `FINALAYZE_TINKOFF_TOKEN`, etc.)
   - Data feed keys (`FINALAYZE_FINNHUB_API_KEY`, `FINALAYZE_NEWSAPI_API_KEY`)

3. Production Redis URL is auto-constructed in `docker-compose.prod.yml`:
   ```
   FINALAYZE_REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
   ```

## First Deploy

```bash
# Build the production image
docker build -f docker/Dockerfile.prod -t finalayze:prod .

# Start the full stack (postgres, redis, app, nginx)
docker compose -f docker/docker-compose.prod.yml up -d

# Verify all services are running
docker compose -f docker/docker-compose.prod.yml ps

# Check application health
curl http://localhost/api/v1/health
```

The entrypoint script automatically runs `alembic upgrade head` before starting
the API server, so database migrations are applied on every deployment.

## Database Migrations

Migrations run automatically on container start via `docker/entrypoint.sh`.

To run migrations manually:
```bash
docker compose -f docker/docker-compose.prod.yml exec app \
    uv run alembic -c alembic/alembic.ini upgrade head
```

To check current migration state:
```bash
docker compose -f docker/docker-compose.prod.yml exec app \
    uv run alembic -c alembic/alembic.ini current
```

The `alembic/env.py` reads `FINALAYZE_DATABASE_URL` from the environment and
converts `+asyncpg` to synchronous `psycopg2` for migration execution.

## Redis Configuration

Production Redis requires a password set via `REDIS_PASSWORD` in `.env`.
The `docker-compose.prod.yml` passes this to both the Redis `--requirepass`
flag and the application's `FINALAYZE_REDIS_URL`.

Redis is used for:
- Event bus (Redis Streams): market data, signals, execution events
- Candle cache (5 min TTL)
- Sentiment cache (30 min TTL)

## Updating

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker compose -f docker/docker-compose.prod.yml up -d --build

# Migrations run automatically on container start
```

## Monitoring Stack

Start the Prometheus + Alertmanager stack alongside the production services:
```bash
docker compose -f docker/docker-compose.prod.yml up -d
docker compose -f docker-compose.monitoring.yml up -d
```

Both compose files share the `finalayze_finalayze_net` network. Prometheus
scrapes `/metrics` from the app container on port 8000 (nginx blocks external
access to `/metrics`).

## Backup Strategy

### Database
```bash
# Create a backup
docker compose -f docker/docker-compose.prod.yml exec postgres \
    pg_dump -U finalayze finalayze > backup_$(date +%Y%m%d).sql

# Restore from backup
docker compose -f docker/docker-compose.prod.yml exec -i postgres \
    psql -U finalayze finalayze < backup_20260228.sql
```

### Redis
Redis data is persisted to the `redisdata` Docker volume. For backup:
```bash
docker compose -f docker/docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD BGSAVE
```

## Scaling

- Adjust `UVICORN_WORKERS` in `.env` (default: 2). Recommended: number of CPU cores.
- Nginx handles rate limiting at 30 req/min per IP on `/api/` endpoints.
- The app container runs as non-root user `finalayze` (uid 1000).

## Broker Credential Management

- Never commit `.env` to git — it is in `.gitignore`
- Store credentials in a secrets manager for production deployments
- Use `FINALAYZE_ALPACA_PAPER=true` and `FINALAYZE_TINKOFF_SANDBOX=true` for testing
- Switching to `FINALAYZE_MODE=real` requires `FINALAYZE_REAL_TOKEN` and API confirmation

## Health Check Endpoints

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `GET /api/v1/health` | No | Liveness check (Docker HEALTHCHECK) |
| `GET /api/v1/health/feeds` | No | Data feed status |
| `GET /api/v1/system/status` | Yes | Full system status with uptime |
| `GET /api/v1/system/errors` | Yes | Last 100 errors |

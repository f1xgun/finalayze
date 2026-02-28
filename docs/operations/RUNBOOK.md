# Operational Runbook

## System Startup

```bash
# Start infrastructure
docker compose -f docker/docker-compose.prod.yml up -d

# Verify health
curl http://localhost/api/v1/health

# Start monitoring (optional)
docker compose -f docker-compose.monitoring.yml up -d

# Check all containers are healthy
docker compose -f docker/docker-compose.prod.yml ps
```

## System Shutdown

```bash
# Graceful shutdown (waits for in-flight requests)
docker compose -f docker/docker-compose.prod.yml down

# Shutdown monitoring
docker compose -f docker-compose.monitoring.yml down

# Full cleanup including volumes (DESTRUCTIVE)
# docker compose -f docker/docker-compose.prod.yml down -v
```

## Emergency Stop

When immediate halt of all trading is required:

1. **Set mode to DEBUG** (stops all trading cycles):
   ```bash
   curl -X POST http://localhost/api/v1/mode \
       -H "X-API-Key: $FINALAYZE_API_KEY" \
       -H "Content-Type: application/json" \
       -d '{"mode": "debug"}'
   ```

2. **Verify mode changed**:
   ```bash
   curl http://localhost/api/v1/mode \
       -H "X-API-Key: $FINALAYZE_API_KEY"
   ```

3. **If API is unresponsive**, stop the container:
   ```bash
   docker compose -f docker/docker-compose.prod.yml stop app
   ```

4. **Review positions** after stopping:
   ```bash
   curl http://localhost/api/v1/portfolio \
       -H "X-API-Key: $FINALAYZE_API_KEY"
   ```

## Circuit Breaker Recovery

When `CircuitBreakerHalted` alert fires:

1. **Check which market triggered**:
   ```bash
   curl http://localhost/api/v1/risk/circuit-breakers \
       -H "X-API-Key: $FINALAYZE_API_KEY"
   ```

2. **Analyze drawdown**: Review portfolio P&L and positions.

3. **Recovery options**:
   - Wait for daily reset (automatic at `FINALAYZE_DAILY_RESET_HOUR_UTC`)
   - Manual override via API if drawdown is acceptable:
     ```bash
     curl -X POST http://localhost/api/v1/risk/circuit-breakers/reset \
         -H "X-API-Key: $FINALAYZE_API_KEY" \
         -H "Content-Type: application/json" \
         -d '{"market": "us"}'
     ```

4. **Post-incident**: Review the trades that caused the drawdown.

## Mode Switching

### Test to Sandbox
```bash
curl -X POST http://localhost/api/v1/mode \
    -H "X-API-Key: $FINALAYZE_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"mode": "sandbox"}'
```

### Sandbox to Real
Requires `FINALAYZE_REAL_TOKEN` to be configured:
```bash
curl -X POST http://localhost/api/v1/mode \
    -H "X-API-Key: $FINALAYZE_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"mode": "real", "confirm_token": "<FINALAYZE_REAL_TOKEN>"}'
```

### Real to Debug (Emergency)
```bash
curl -X POST http://localhost/api/v1/mode \
    -H "X-API-Key: $FINALAYZE_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"mode": "debug"}'
```

## Database Operations

### Check migration status
```bash
docker compose -f docker/docker-compose.prod.yml exec app \
    uv run alembic -c alembic/alembic.ini current
```

### Create backup
```bash
docker compose -f docker/docker-compose.prod.yml exec postgres \
    pg_dump -U finalayze finalayze > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Restore from backup
```bash
docker compose -f docker/docker-compose.prod.yml exec -i postgres \
    psql -U finalayze finalayze < backup_file.sql
```

## Log Analysis

```bash
# Application logs
docker compose -f docker/docker-compose.prod.yml logs app --tail=100

# Follow logs in real-time
docker compose -f docker/docker-compose.prod.yml logs -f app

# Filter for errors
docker compose -f docker/docker-compose.prod.yml logs app 2>&1 | grep -i error

# Nginx access logs
docker compose -f docker/docker-compose.prod.yml logs nginx --tail=50
```

## Daily Reconciliation Checklist

1. Compare positions reported by `/api/v1/portfolio` with broker dashboards
2. Verify trade count in system matches broker trade history
3. Check P&L alignment between system and broker accounts
4. Review any rejected orders in `/api/v1/system/errors`
5. Confirm circuit breakers reset after daily reset cycle
6. Check news feed is active (articles fetched in last 30 min)
7. Verify ML model predictions are being generated

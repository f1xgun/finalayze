# Monitoring

## Purpose
Runtime health monitoring, anomaly detection, sandbox validation, and go/no-go checks for the trading system.

## Layer
Layer 6 -- Monitoring. Imports from orchestration (L5), api (L6), and all lower layers. Consumed by api/ endpoints and main.py lifespan.

Import rule: monitoring/ may import from L0-L5 and L6 (api/). Must NOT be imported by L0-L4.

## Key Files
- `health_monitor.py` -- HealthMonitor: tracks feed freshness, cycle health, component status. Alerts via TelegramAlerter on degradation.
- `sandbox_monitor.py` -- SandboxMonitorService: monitors sandbox trading performance, persists metrics to TimescaleDB, delegates anomaly checks to AnomalyDetector.
- `anomaly_detector.py` -- AnomalyDetector: statistical anomaly detection on portfolio metrics (drawdown spikes, unusual returns, volume anomalies). Per-metric cooldown to avoid alert spam.
- `go_no_go.py` -- GoNoGoReporter: evaluates 8 criteria against sandbox metrics with PROCEED/DEFER/ABORT verdict. Thresholds loaded from config/gate_thresholds.yaml.

## Public API
- `HealthMonitor` -- feed timestamp tracking, cycle health reporting, broker connectivity checks
- `SandboxMonitorService` -- sandbox performance monitoring, per-cycle metric persistence
- `AnomalyDetector` -- anomaly detection and alerting with z-score and threshold rules
- `GoNoGoReporter` -- production readiness validation with configurable gate thresholds
- `GateThresholds` -- threshold configuration (from YAML)
- `GateReport` -- structured go/no-go evaluation result
- `GateVerdict` -- PROCEED / DEFER / ABORT enum
- `CycleMetrics` -- per-cycle metric dataclass
- `HealthCheckResult` -- health check result dataclass

## Contracts
- Input: TradingLoop reference (for health checks), TelegramAlerter (for alerts), BrokerRouter (for connectivity)
- Output: Health status dicts, anomaly alerts, go/no-go verdicts
- Import rule: monitoring/ may import from L0-L5 and L6 (api/). Must NOT be imported by L0-L4.

## Testing
- Test location: tests/unit/test_health_monitor.py, tests/unit/test_sandbox_monitor.py
- Run: uv run pytest tests/unit/test_health_monitor.py tests/unit/test_sandbox_monitor.py -v

---
title: Multi-Instance Sandbox A/B Testing
trigger_condition: When basic experiment framework (registry + Streamlit UI) is operational and first experiments run successfully via backtest
planted_date: 2026-04-07
---

# Multi-Instance Sandbox A/B Testing

## Idea

Run multiple sandbox instances in parallel, each with different strategy/risk configurations,
to validate experiment results beyond backtesting.

## Why Wait

- Need experiment registry and UI first (otherwise no way to track parallel runs)
- Docker compose currently single-instance — needs orchestration layer
- SandboxMonitor needs `experiment_id` field to separate metrics across instances
- Resource cost: each instance needs its own DB connections, gRPC channels, API quota

## When to Trigger

- Experiment framework handles backtest-based experiments end-to-end
- At least 3 experiments completed and validated via backtest
- Team wants to validate backtest-winning proposals in live conditions

## Implementation Sketch

- Docker Compose profiles or Kubernetes for parallel instances
- Shared PostgreSQL with `experiment_id` partitioning
- Shared Grafana/Prometheus with instance labels
- Streamlit UI shows side-by-side live metrics for A vs B vs A+B
- Auto-shutdown after configured experiment duration
- Budget cap per experiment instance (absolute max notional value)

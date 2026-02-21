# Data Flow

## Overview

This document describes the flow of data through the Finalayze system, from
ingestion to trade execution.

## Primary Data Flows

### 1. Market Data Flow

```
External Sources              Internal Processing              Storage
+-------------------+
| Alpaca WebSocket  |--+
+-------------------+  |   +------------------+    +-------------------+
                       +-->| Data Normalizer  |--->| TimescaleDB       |
+-------------------+  |   | (data/)          |    | (OHLCV hypertable)|
| Tinkoff gRPC      |--+   +--------+---------+    +-------------------+
+-------------------+            |
                                 v
                          +------+--------+
                          | Event Bus     |
                          | MARKET_UPDATE |
                          +------+--------+
                                 |
                    +------------+------------+
                    |            |            |
                    v            v            v
              +---------+  +---------+  +---------+
              |Strategy |  |   ML    |  | Redis   |
              |Engine   |  |Pipeline |  | Cache   |
              +---------+  +---------+  +---------+
```

### 2. News & Sentiment Flow

```
+-------------------+    +-------------------+    +-------------------+
| News APIs         |--->| News Fetcher      |--->| Raw Article Store |
| (RSS, REST)       |    | (data/)           |    | (PostgreSQL)      |
+-------------------+    +--------+----------+    +-------------------+
                                  |
                                  v
                         +--------+----------+
                         | LLM Analyzer      |
                         | (Claude Sonnet)   |
                         | - Sentiment score |
                         | - Fact-check      |
                         | - Entity extract  |
                         +--------+----------+
                                  |
                                  v
                         +--------+----------+    +-------------------+
                         | Event Bus         |--->| Sentiment Store   |
                         | NEWS_SENTIMENT    |    | (PostgreSQL)      |
                         +--------+----------+    +-------------------+
                                  |
                                  v
                         +--------+----------+
                         | Strategy Engine   |
                         | (event_driven)    |
                         +-------------------+
```

### 3. Signal Generation Flow

```
+-------------------+    +-------------------+    +-------------------+
| Technical         |    | ML Ensemble       |    | News Sentiment    |
| Indicators        |    | (per segment)     |    | Score             |
| (pandas-ta)       |    | XGB+LGBM+LSTM    |    | (Claude)          |
+--------+----------+    +--------+----------+    +--------+----------+
         |                        |                        |
         v                        v                        v
+--------+------------------------+------------------------+--------+
|                     Signal Combiner (strategies/)                 |
|  - Momentum signal (RSI + MACD weight)                           |
|  - Mean reversion signal (Bollinger Bands weight)                |
|  - Event-driven signal (news sentiment weight)                   |
|  - Pairs trading signal (cointegration weight)                   |
+--------+----------------------------------------------------------+
         |
         v
+--------+----------+
| Combined Signal   |
| per instrument    |
| (direction, size, |
|  confidence)      |
+--------+----------+
         |
         v
+--------+----------+
| Risk Manager      |
| (risk/)           |
| - Position sizing |
| - Exposure check  |
| - Drawdown check  |
+--------+----------+
         |
         v
+--------+----------+
| Order Request     |
| (approved/denied) |
+-------------------+
```

### 4. Execution Flow

```
+-------------------+
| Order Request     |
| (from Risk Mgr)  |
+--------+----------+
         |
         v
+--------+----------+
| Execution Router  |
| (execution/)      |
+--------+----------+
         |
    +----+----+
    |         |
    v         v
+---+---+ +---+---+
|Alpaca | |Tinkoff|
|Broker | |Broker |
+---+---+ +---+---+
    |         |
    v         v
+---+---------+---+
| Fill Reconciler |
| - Slippage      |
| - Commission    |
+--------+--------+
         |
         v
+--------+--------+
| Portfolio Update|----> TimescaleDB (trade log)
| Event Bus       |----> Redis (position cache)
| TRADE_EXECUTED  |----> Dashboard (WebSocket)
+-----------------+
```

## Event Bus Events

| Event | Producer | Consumers |
|---|---|---|
| `MARKET_UPDATE` | Data Normalizer | Strategy Engine, ML Pipeline, Redis Cache |
| `NEWS_SENTIMENT` | LLM Analyzer | Strategy Engine (event-driven), Sentiment Store |
| `SIGNAL_GENERATED` | Strategy Engine | Risk Manager |
| `ORDER_APPROVED` | Risk Manager | Execution Router |
| `ORDER_REJECTED` | Risk Manager | Dashboard, Logging |
| `TRADE_EXECUTED` | Execution Router | Portfolio Manager, Dashboard, Trade Log |
| `POSITION_UPDATED` | Portfolio Manager | Strategy Engine, Risk Manager |
| `MODEL_RETRAINED` | ML Pipeline | Strategy Engine |

## Data Refresh Rates

| Data Type | Mode: sandbox/real | Mode: test | Mode: debug |
|---|---|---|---|
| OHLCV (1min) | Real-time WebSocket | Historical replay | Fixture data |
| OHLCV (daily) | End-of-day batch | Historical replay | Fixture data |
| News articles | Polling every 5 min | Historical set | Fixture data |
| Sentiment scores | On news arrival | Pre-computed | Fixture data |
| ML predictions | Every 15 min | On-demand | Mock scores |
| Portfolio snapshot | On every trade | On every trade | On every trade |

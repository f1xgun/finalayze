"""Tests for MetricsCollector — verifies all static methods exist and don't raise."""

from __future__ import annotations

from finalayze.api.metrics import MetricsCollector


def test_set_portfolio_equity() -> None:
    MetricsCollector.set_portfolio_equity("us", 10000.0)


def test_set_circuit_breaker_level() -> None:
    MetricsCollector.set_circuit_breaker_level("us", 0)
    MetricsCollector.set_circuit_breaker_level("moex", 2)


def test_record_trade() -> None:
    MetricsCollector.record_trade("us", "buy", 5.0, 0.1)


def test_record_rejection() -> None:
    MetricsCollector.record_rejection("us", "insufficient_cash")


def test_set_strategy_win_rate() -> None:
    MetricsCollector.set_strategy_win_rate("us", "momentum", 0.55)


def test_record_signal() -> None:
    MetricsCollector.record_signal("us", "momentum", "BUY")


def test_set_ml_retrain_timestamp() -> None:
    MetricsCollector.set_ml_retrain_timestamp("us_tech_xgb", 1700000000.0)


def test_observe_ml_prediction_latency() -> None:
    MetricsCollector.observe_ml_prediction_latency("us_tech_xgb", 0.05)


def test_observe_feed_latency() -> None:
    MetricsCollector.observe_feed_latency("us", "finnhub", 0.2)


def test_set_news_feed_timestamp() -> None:
    MetricsCollector.set_news_feed_timestamp("global", 1700000000.0)


def test_set_usd_rub_rate() -> None:
    MetricsCollector.set_usd_rub_rate(89.5)


def test_set_portfolio_equity_rub() -> None:
    MetricsCollector.set_portfolio_equity_rub(8950000.0)

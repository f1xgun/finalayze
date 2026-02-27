"""Prometheus business metrics collector for Finalayze.

Layer 6 -- API layer. Exposes static methods so callers need no instance.

Usage:
    from finalayze.api.metrics import MetricsCollector

    MetricsCollector.set_portfolio_equity("us", 12500.0)
    MetricsCollector.record_trade("us", "buy", slippage_bps=3.0, fill_latency_seconds=0.12)
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Portfolio ──────────────────────────────────────────────────────────────────
portfolio_equity_usd = Gauge(
    "finalayze_portfolio_equity_usd",
    "Portfolio equity in USD",
    ["market"],
)
portfolio_equity_pct_change = Gauge(
    "finalayze_portfolio_equity_pct_change",
    "Portfolio equity percentage change",
    ["market", "period"],
)
daily_pnl_usd = Gauge(
    "finalayze_daily_pnl_usd",
    "Daily P&L in USD",
    ["market"],
)
drawdown_pct = Gauge(
    "finalayze_drawdown_pct",
    "Current drawdown as a fraction (0.0-1.0)",
    ["market"],
)
max_drawdown_pct = Gauge(
    "finalayze_max_drawdown_pct",
    "Rolling 30d maximum drawdown as a fraction",
    ["market"],
)

# ── Positions ─────────────────────────────────────────────────────────────────
open_positions_count = Gauge(
    "finalayze_open_positions_count",
    "Number of open positions",
    ["market"],
)

# ── Circuit breakers ──────────────────────────────────────────────────────────
circuit_breaker_level = Gauge(
    "finalayze_circuit_breaker_level",
    "Circuit breaker level (0=normal 1=caution 2=halted 3=liquidate)",
    ["market"],
)

# ── Execution ─────────────────────────────────────────────────────────────────
trades_total = Counter(
    "finalayze_trades_total",
    "Cumulative filled orders",
    ["market", "side"],
)
trade_slippage_bps = Histogram(
    "finalayze_trade_slippage_bps",
    "Trade slippage in basis points",
    ["market"],
    buckets=[0, 1, 2, 5, 10, 20, 50, 100],
)
order_fill_latency_seconds = Histogram(
    "finalayze_order_fill_latency_seconds",
    "Order fill latency in seconds",
    ["market"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
order_rejection_total = Counter(
    "finalayze_order_rejection_total",
    "Order rejections by reason",
    ["market", "reason"],
)

# ── Strategy ──────────────────────────────────────────────────────────────────
strategy_win_rate = Gauge(
    "finalayze_strategy_win_rate",
    "Rolling win rate over last 100 trades",
    ["market", "strategy"],
)
strategy_signal_count = Counter(
    "finalayze_strategy_signal_count",
    "Cumulative signals generated",
    ["market", "strategy", "dir"],
)

# ── ML model health ───────────────────────────────────────────────────────────
ml_model_last_retrain_timestamp = Gauge(
    "finalayze_ml_model_last_retrain_timestamp",
    "Unix timestamp of last model retrain",
    ["model"],
)
ml_model_prediction_latency_seconds = Histogram(
    "finalayze_ml_model_prediction_latency_seconds",
    "ML model prediction latency in seconds",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)

# ── Data feed health ──────────────────────────────────────────────────────────
market_data_feed_latency_seconds = Histogram(
    "finalayze_market_data_feed_latency_seconds",
    "Market data fetch latency in seconds",
    ["market", "src"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
news_feed_last_article_timestamp = Gauge(
    "finalayze_news_feed_last_article_timestamp",
    "Unix timestamp of last processed news article",
    ["scope"],
)

# ── Currency ──────────────────────────────────────────────────────────────────
usd_rub_rate = Gauge(
    "finalayze_usd_rub_rate",
    "USD/RUB exchange rate",
)
portfolio_equity_rub = Gauge(
    "finalayze_portfolio_equity_rub",
    "MOEX portfolio equity in RUB",
)


class MetricsCollector:
    """Facade for updating all Prometheus business metrics.

    All methods are static; prometheus_client objects are module-level singletons
    and are thread-safe by design.
    """

    @staticmethod
    def set_portfolio_equity(market: str, equity_usd: float) -> None:
        """Set portfolio equity in USD for the given market."""
        portfolio_equity_usd.labels(market=market).set(equity_usd)

    @staticmethod
    def set_daily_pnl(market: str, pnl_usd: float) -> None:
        """Set daily P&L in USD for the given market."""
        daily_pnl_usd.labels(market=market).set(pnl_usd)

    @staticmethod
    def set_drawdown(market: str, pct: float) -> None:
        """Set current drawdown fraction for the given market."""
        drawdown_pct.labels(market=market).set(pct)

    @staticmethod
    def set_max_drawdown(market: str, pct: float) -> None:
        """Set rolling 30d max drawdown fraction for the given market."""
        max_drawdown_pct.labels(market=market).set(pct)

    @staticmethod
    def set_open_positions(market: str, count: int) -> None:
        """Set number of open positions for the given market."""
        open_positions_count.labels(market=market).set(count)

    @staticmethod
    def set_circuit_breaker_level(market: str, level: int) -> None:
        """Set circuit breaker level (0=normal, 1=caution, 2=halted, 3=liquidate)."""
        circuit_breaker_level.labels(market=market).set(level)

    @staticmethod
    def record_trade(
        market: str,
        side: str,
        slippage_bps: float,
        fill_latency_seconds: float,
    ) -> None:
        """Increment trade counter and record slippage/latency histograms."""
        trades_total.labels(market=market, side=side).inc()
        trade_slippage_bps.labels(market=market).observe(slippage_bps)
        order_fill_latency_seconds.labels(market=market).observe(fill_latency_seconds)

    @staticmethod
    def record_rejection(market: str, reason: str) -> None:
        """Increment order rejection counter."""
        order_rejection_total.labels(market=market, reason=reason).inc()

    @staticmethod
    def set_strategy_win_rate(market: str, strategy: str, win_rate: float) -> None:
        """Set rolling win rate for a strategy."""
        strategy_win_rate.labels(market=market, strategy=strategy).set(win_rate)

    @staticmethod
    def record_signal(market: str, strategy: str, direction: str) -> None:
        """Increment signal counter for a strategy and direction."""
        strategy_signal_count.labels(market=market, strategy=strategy, dir=direction).inc()

    @staticmethod
    def set_ml_retrain_timestamp(model: str, ts: float) -> None:
        """Set the Unix timestamp of the last model retrain."""
        ml_model_last_retrain_timestamp.labels(model=model).set(ts)

    @staticmethod
    def observe_ml_prediction_latency(model: str, latency_seconds: float) -> None:
        """Record ML model prediction latency."""
        ml_model_prediction_latency_seconds.labels(model=model).observe(latency_seconds)

    @staticmethod
    def observe_feed_latency(market: str, src: str, latency_seconds: float) -> None:
        """Record market data feed fetch latency."""
        market_data_feed_latency_seconds.labels(market=market, src=src).observe(latency_seconds)

    @staticmethod
    def set_news_feed_timestamp(scope: str, ts: float) -> None:
        """Set the Unix timestamp of the last processed news article."""
        news_feed_last_article_timestamp.labels(scope=scope).set(ts)

    @staticmethod
    def set_usd_rub_rate(rate: float) -> None:
        """Set the USD/RUB exchange rate."""
        usd_rub_rate.set(rate)

    @staticmethod
    def set_portfolio_equity_rub(equity_rub: float) -> None:
        """Set MOEX portfolio equity in RUB."""
        portfolio_equity_rub.set(equity_rub)

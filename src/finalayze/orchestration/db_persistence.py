"""Database persistence for fire-and-forget writes from trading loop.

Extracted from trading_loop.py to improve modularity and testability.
Handles news articles, sentiment scores, orders, signals, and equity snapshots.
All operations are fire-and-forget to prevent crashes in the trading loop (PERSIST-05).
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import uuid

    from config.settings import Settings
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from finalayze.analysis.news_impact_analyzer import NewsImpactResult
    from finalayze.core.schemas import NewsArticle
    from finalayze.execution.broker_base import OrderRequest
    from finalayze.execution.simulated_broker import StopLossState

_log = structlog.get_logger()

_DB_PERSIST_TIMEOUT = 120  # seconds — generous for fire-and-forget writes


class TradingPersistence:
    """Manages all fire-and-forget database persistence operations.

    Designed to be called from the TradingLoop's background event loop.
    All operations use asyncio.run_coroutine_threadsafe for cross-loop safety.
    """

    def __init__(
        self,
        db_url: str | None,
        async_loop: asyncio.AbstractEventLoop | None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize persistence layer with optional DB URL.

        Args:
            db_url: Database URL for async SQLAlchemy engine. If None, persistence is disabled.
            async_loop: Event loop for executing async operations. Should be the background loop
                used by TradingLoop._run_async().
            settings: Settings object for accessing mode and other configuration. Optional.
        """
        self._db_url = db_url
        self._async_loop = async_loop
        self._settings = settings
        self._bg_session_factory: async_sessionmaker[AsyncSession] | None = None

    def _get_bg_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Return a session factory bound to the background event loop.

        asyncpg connections are pinned to the event loop where they were first
        used.  The global ``get_async_session_factory()`` creates its engine on
        the FastAPI (uvicorn) loop, so using it from the background loop that
        ``_run_async`` manages causes ``RuntimeError: Future attached to a
        different loop``.  This method lazily creates a *separate* engine for
        the background loop, avoiding the cross-loop conflict.
        """
        if self._bg_session_factory is None:
            from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession  # noqa: PLC0415
            from sqlalchemy.ext.asyncio import (  # noqa: PLC0415
                async_sessionmaker as _async_sessionmaker,
            )
            from sqlalchemy.ext.asyncio import (  # noqa: PLC0415
                create_async_engine as _create_async_engine,
            )

            if self._db_url is None:
                raise RuntimeError("db_url not set; cannot create session factory")

            engine = _create_async_engine(
                self._db_url,
                echo=False,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=2,
                pool_timeout=30,
                pool_recycle=1800,
            )
            self._bg_session_factory = _async_sessionmaker(
                engine, class_=_AsyncSession, expire_on_commit=False
            )
        return self._bg_session_factory

    def _run_async(self, coro: Any, *, timeout: int = 30) -> Any:
        """Execute a coroutine on the background event loop.

        Uses asyncio.run_coroutine_threadsafe to safely cross loop boundaries.

        Args:
            coro: Coroutine to execute.
            timeout: Timeout in seconds (default 30).

        Returns:
            Result of the coroutine.

        Raises:
            TimeoutError: If the coroutine exceeds the timeout.
            RuntimeError: If the async loop is not available.
        """
        if self._async_loop is None or self._async_loop.is_closed():
            raise RuntimeError("async_loop not available; cannot persist to DB")
        future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        return future.result(timeout=timeout)

    def _persist_to_db(self, coro: Any, *, table: str, **ctx: Any) -> None:
        """Fire-and-forget DB write. Never crashes the trading loop (PERSIST-05)."""
        if self._db_url is None:
            _log.debug("db_persist_skipped", table=table, reason="db_url not set", **ctx)
            return
        try:
            self._run_async(coro, timeout=_DB_PERSIST_TIMEOUT)
            _log.debug("db_persist_ok", table=table, **ctx)
        except Exception:
            from finalayze.api.metrics import db_write_failures  # noqa: PLC0415

            db_write_failures.labels(table=table).inc()
            _log.warning("db_persist_failed", table=table, **ctx, exc_info=True)

    async def _persist_to_db_async(self, coro: Any, *, table: str, **ctx: Any) -> None:
        """Async variant of _persist_to_db for use in async contexts (PERSIST-05)."""
        if self._db_url is None:
            _log.debug("db_persist_skipped", table=table, reason="db_url not set", **ctx)
            return
        try:
            await coro
            _log.debug("db_persist_ok", table=table, **ctx)
        except Exception:
            from finalayze.api.metrics import db_write_failures  # noqa: PLC0415

            db_write_failures.labels(table=table).inc()
            _log.warning("db_persist_failed", table=table, **ctx, exc_info=True)

    async def _persist_news_article_async(
        self,
        article: NewsArticle,
        impact_result: NewsImpactResult | None,
    ) -> None:
        """Persist a news article to the news_articles table (PERSIST-03)."""
        from finalayze.core.models import NewsArticleModel  # noqa: PLC0415

        content_hash = hashlib.sha256(article.content.encode()).hexdigest()[:32]
        factory = self._get_bg_session_factory()
        async with factory() as session:
            row = NewsArticleModel(
                source=article.source[:50],
                title=article.title,
                summary=article.content[:500] if article.content else None,
                content=content_hash,
                url=article.url or None,
                language=getattr(article, "language", "en"),
                published_at=article.published_at,
                symbols=list(impact_result.direct_tickers) if impact_result else [],
                affected_segments=(
                    [s.sector for s in impact_result.affected_sectors] if impact_result else []
                ),
                raw_sentiment=(
                    Decimal(str(round(impact_result.sentiment, 4))) if impact_result else None
                ),
                credibility_score=(
                    Decimal(str(round(impact_result.confidence, 4))) if impact_result else None
                ),
                is_processed=impact_result is not None,
            )
            session.add(row)
            await session.commit()

    async def _persist_sentiment_batch_async(
        self,
        ticker_scores: dict[str, float],
        market_id: str,
        confidence: float,
    ) -> None:
        """Persist sentiment scores for a batch of tickers (PERSIST-04)."""
        from finalayze.core.models import SentimentScoreModel  # noqa: PLC0415

        now = datetime.now(UTC)
        factory = self._get_bg_session_factory()
        async with factory() as session:
            for ticker, score in ticker_scores.items():
                row = SentimentScoreModel(
                    symbol=ticker,
                    market_id=market_id,
                    timestamp=now,
                    news_sentiment=Decimal(str(round(score, 4))),
                    composite_sentiment=Decimal(str(round(score, 4))),
                    confidence=Decimal(str(round(confidence, 4))),
                )
                session.add(row)
            await session.commit()

    async def _persist_order_async(
        self,
        order: OrderRequest,
        result: Any,
        market_id: str,
    ) -> None:
        """Persist a filled/rejected order to the orders table."""
        from finalayze.core.models import OrderModel  # noqa: PLC0415

        mode = self._settings.mode.value if self._settings else "test"
        factory = self._get_bg_session_factory()
        async with factory() as session:
            row = OrderModel(
                broker=market_id,
                broker_order_id=result.order_id or None,
                symbol=order.symbol,
                market_id=market_id,
                side=order.side,
                order_type="market",
                quantity=order.quantity,
                currency="RUB" if market_id.startswith(("moex", "ru_")) else "USD",
                status="filled" if result.filled else "rejected",
                filled_quantity=result.quantity if result.filled else Decimal(0),
                filled_avg_price=result.fill_price,
                filled_at=datetime.now(UTC) if result.filled else None,
                submitted_at=datetime.now(UTC),
                mode=mode,
            )
            session.add(row)
            await session.commit()

    async def _persist_signal_async(self, signal: Any) -> None:
        """Persist a generated signal to the signals table."""
        from finalayze.core.models import SignalModel  # noqa: PLC0415

        mode = self._settings.mode.value if self._settings else "test"
        factory = self._get_bg_session_factory()
        async with factory() as session:
            row = SignalModel(
                strategy_name=signal.strategy_name,
                symbol=signal.symbol,
                market_id=signal.market_id,
                segment_id=signal.segment_id,
                direction=signal.direction.value,
                confidence=Decimal(str(round(signal.confidence, 4))),
                features=signal.features or None,
                reasoning=signal.reasoning,
                created_at=datetime.now(UTC),
                mode=mode,
            )
            session.add(row)
            await session.commit()

    def _persist_equity_snapshots(
        self,
        baselines: dict[str, Decimal],
        now: datetime,
    ) -> None:
        """Persist equity snapshots to DB asynchronously."""
        if self._db_url is None:
            _log.debug("equity_snapshot_persist_skipped", reason="db_url not set")
            return
        try:
            self._run_async(
                self._persist_snapshots_async(baselines, now),
            )
        except Exception:
            _log.warning("equity_snapshot_persist_failed", exc_info=True)

    async def _persist_snapshots_async(
        self,
        baselines: dict[str, Decimal],
        now: datetime,
    ) -> None:
        """Async helper to persist equity snapshots to TimescaleDB.

        Creates one DailyEquitySnapshot row per market_id. Currency is
        determined from market_id prefix (moex/ru_ -> RUB, else USD).
        """
        from finalayze.core.models import DailyEquitySnapshot  # noqa: PLC0415

        factory = self._get_bg_session_factory()
        async with factory() as session:
            for market_id, equity in baselines.items():
                currency = (
                    "RUB" if market_id.startswith("moex") or market_id.startswith("ru_") else "USD"
                )
                snapshot = DailyEquitySnapshot(
                    timestamp=now,
                    market_id=market_id,
                    equity=equity,
                    currency=currency,
                )
                session.add(snapshot)
            await session.commit()
        _log.info(
            "equity_snapshots_persisted",
            markets=list(baselines.keys()),
            count=len(baselines),
        )

    # Public wrapper methods for convenient calling from TradingLoop

    def persist_news_article(
        self,
        article: NewsArticle,
        impact_result: NewsImpactResult | None,
    ) -> None:
        """Fire-and-forget wrapper for news article persistence."""
        self._persist_to_db(
            self._persist_news_article_async(article, impact_result),
            table="news_articles",
        )

    def persist_sentiment_batch(
        self,
        ticker_scores: dict[str, float],
        market_id: str,
        confidence: float,
    ) -> None:
        """Fire-and-forget wrapper for sentiment batch persistence."""
        if not ticker_scores:
            return
        self._persist_to_db(
            self._persist_sentiment_batch_async(ticker_scores, market_id, confidence),
            table="sentiment_scores",
            market=market_id,
            tickers=len(ticker_scores),
        )

    def persist_order(
        self,
        order: OrderRequest,
        result: Any,
        market_id: str,
    ) -> None:
        """Fire-and-forget wrapper for order persistence."""
        self._persist_to_db(
            self._persist_order_async(order, result, market_id),
            table="orders",
            symbol=order.symbol,
            market=market_id,
        )

    def persist_signal(self, signal: Any) -> None:
        """Fire-and-forget wrapper for signal persistence."""
        self._persist_to_db(
            self._persist_signal_async(signal),
            table="signals",
            symbol=signal.symbol,
            strategy=signal.strategy_name,
        )

    def persist_equity_snapshots(
        self,
        baselines: dict[str, Decimal],
        now: datetime,
    ) -> None:
        """Fire-and-forget wrapper for equity snapshot persistence."""
        self._persist_equity_snapshots(baselines, now)

    async def _persist_stop_snapshots_async(
        self,
        states: dict[str, StopLossState],
        market_ids: dict[str, str],
        prices: dict[str, Decimal],
        now: datetime,
        event_type: str,
    ) -> None:
        """Write one StopLossEventModel row per entry in ``states`` (STOP-03)."""
        from finalayze.core.models import StopLossEventModel  # noqa: PLC0415

        factory = self._get_bg_session_factory()
        async with factory() as session:
            for sym, state in states.items():
                row = StopLossEventModel(
                    timestamp=now,
                    symbol=sym,
                    market_id=market_ids.get(sym, ""),
                    event_type=event_type,
                    entry_price=state.entry_price,
                    current_stop=state.current_stop,
                    highest_price=state.highest_price,
                    atr_value=state.atr_value,
                    activation_atr=state.activation_atr,
                    trail_atr=state.trail_atr,
                    trail_activated=state.trail_activated,
                    current_price=prices.get(sym),
                )
                session.add(row)
            await session.commit()

    def persist_stop_snapshots(
        self,
        states: dict[str, StopLossState],
        market_ids: dict[str, str],
        prices: dict[str, Decimal],
        now: datetime,
        event_type: str = "snapshot",
    ) -> None:
        """Fire-and-forget write of stop-loss state snapshots (PERSIST-05, STOP-03).

        Mirrors ``persist_equity_snapshots``:
          - No exception escapes.
          - ``db_write_failures.labels(table='stop_loss_events')`` increments on
            error.
          - Silently skips when ``self._db_url`` is None.
          - Does NOT affect ``_consecutive_equity_errors`` (lives on TradingLoop,
            not here).

        Args:
            states: symbol -> StopLossState snapshot to persist.
            market_ids: symbol -> market_id ("us" | "moex") for each row.
            prices: symbol -> current price (for chart overlay); optional per sym.
            now: Timestamp to write.
            event_type: One of ``'snapshot' | 'entry' | 'trigger' | 'activation' |
                'exit'``.
        """
        if not states:
            return
        # Note: structlog reserves the ``event`` positional arg for the log
        # message, so we forward the event_type under a different ctx key
        # (``event_kind``) to avoid ``TypeError: multiple values for 'event'``
        # in the ``db_persist_skipped`` / ``db_persist_ok`` / ``db_persist_failed``
        # debug/warning log lines.
        self._persist_to_db(
            self._persist_stop_snapshots_async(states, market_ids, prices, now, event_type),
            table="stop_loss_events",
            event_kind=event_type,
            count=len(states),
        )

    async def _persist_alert_async(
        self,
        alert_id: uuid.UUID,
        timestamp: datetime,
        alert_type: str,
        priority: str,
        symbol: str | None,
        market_id: str | None,
        message: str,
        parent_id: uuid.UUID | None,
        delivery_status: str,
        alert_metadata: dict[str, object] | None,
    ) -> None:
        """Write one AlertModel row. Internal async body for persist_alert.

        Phase 57-01 (ALRT-03). Used by the Phase 57-02 alerter write hook
        (`TelegramAlerter._send` / `_send_sync`) to record every outbound
        Telegram message into the `alerts` hypertable.
        """
        from finalayze.core.models import AlertModel  # noqa: PLC0415

        factory = self._get_bg_session_factory()
        async with factory() as session:
            row = AlertModel(
                id=alert_id,
                timestamp=timestamp,
                alert_type=alert_type,
                priority=priority,
                symbol=symbol,
                market_id=market_id,
                message=message,
                parent_id=parent_id,
                delivery_status=delivery_status,
                alert_metadata=alert_metadata,
            )
            session.add(row)
            await session.commit()

    def persist_alert(
        self,
        alert_id: uuid.UUID,
        timestamp: datetime,
        alert_type: str,
        priority: str,
        message: str,
        *,
        symbol: str | None = None,
        market_id: str | None = None,
        parent_id: uuid.UUID | None = None,
        delivery_status: str = "queued",
        alert_metadata: dict[str, object] | None = None,
    ) -> None:
        """Fire-and-forget alert persistence (PERSIST-05 envelope).

        Never raises: DB failures increment ``db_write_failures`` counter +
        log warning. Used by TelegramAlerter write-hook (Phase 57 ALRT-03).
        """
        # Forward `alert_type` under the `alert_type_key` ctx kwarg so the
        # `db_persist_skipped/ok/failed` log lines do not collide with the
        # structlog-reserved `event` positional arg (mirrors `event_kind`
        # precedent in persist_stop_snapshots).
        self._persist_to_db(
            self._persist_alert_async(
                alert_id, timestamp, alert_type, priority, symbol, market_id,
                message, parent_id, delivery_status, alert_metadata,
            ),
            table="alerts",
            alert_type_key=alert_type,
            symbol=symbol or "",
        )

    async def _update_alert_status_async(
        self,
        alert_id: uuid.UUID,
        timestamp: datetime,
        delivery_status: str,
    ) -> None:
        """Update delivery_status for an existing alert row (D-05).

        Called by the alerter write hook AFTER the httpx response is observed:
        ``sent`` on success, ``failed`` on transport error or rate limit.
        """
        from sqlalchemy import update  # noqa: PLC0415

        from finalayze.core.models import AlertModel  # noqa: PLC0415

        factory = self._get_bg_session_factory()
        async with factory() as session:
            await session.execute(
                update(AlertModel)
                .where(
                    AlertModel.id == alert_id,
                    AlertModel.timestamp == timestamp,
                )
                .values(delivery_status=delivery_status),
            )
            await session.commit()

    def update_alert_status(
        self,
        alert_id: uuid.UUID,
        timestamp: datetime,
        delivery_status: str,
    ) -> None:
        """Fire-and-forget status update. Never blocks the Telegram send path."""
        self._persist_to_db(
            self._update_alert_status_async(alert_id, timestamp, delivery_status),
            table="alerts",
            op="status_update",
            status=delivery_status,
        )

    async def persist_sentiment_batch_async(
        self,
        ticker_scores: dict[str, float],
        market_id: str,
        confidence: float,
    ) -> None:
        """Async wrapper for sentiment batch persistence (for async contexts)."""
        if not ticker_scores:
            return
        await self._persist_to_db_async(
            self._persist_sentiment_batch_async(ticker_scores, market_id, confidence),
            table="sentiment_scores",
            market=market_id,
            tickers=len(ticker_scores),
        )

    async def persist_news_article_async(
        self,
        article: NewsArticle,
        impact_result: NewsImpactResult | None,
    ) -> None:
        """Async wrapper for news article persistence (for async contexts)."""
        await self._persist_to_db_async(
            self._persist_news_article_async(article, impact_result),
            table="news_articles",
        )

"""Layer 2 read-only accessor for rolling sentiment aggregates.

Queries the ``sentiment_7d_avg`` continuous aggregate view created by
migration 005. Returns empty list when no data exists -- safe for the
v11 ML feature pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from sqlalchemy import text

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_WINDOW_INTERVALS: dict[str, str] = {
    "1d": "1 day",
    "7d": "7 days",
    "30d": "30 days",
}


class SentimentRow(NamedTuple):
    """Single row from the sentiment_7d_avg continuous aggregate."""

    bucket: datetime
    avg_score: float | None
    article_count: int


class SentimentStore:
    """Read-only accessor for rolling sentiment aggregates.

    Queries the ``sentiment_7d_avg`` continuous aggregate view.
    Returns empty list when no data exists -- never raises on missing data.
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
            ticker: Instrument symbol (e.g. ``'SBER'``).
            window: Rolling window string (``'1d'``, ``'7d'``, ``'30d'``).
                Invalid values fall back to ``'7d'``.
            market_id: Market identifier (default ``'moex'``).

        Returns:
            List of ``SentimentRow(bucket, avg_score, article_count)`` ordered
            by bucket ascending. Empty list if no rows exist.
        """
        interval = _WINDOW_INTERVALS.get(window, "7 days")
        sql = text(
            "SELECT bucket, avg_score, article_count "
            "FROM sentiment_7d_avg "
            "WHERE symbol    = :symbol "
            "  AND market_id = :market_id "
            "  AND bucket   >= NOW() - CAST(:interval AS INTERVAL) "
            "ORDER BY bucket ASC"
        )
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

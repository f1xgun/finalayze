"""Cached macro context with scheduled refresh (Layer 2).

Sync-only — no async. APScheduler BackgroundScheduler calls sync functions.
Future LiveMacroContextProvider with httpx must use asyncio.to_thread().
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from finalayze.core.models import MacroSnapshotModel
from finalayze.data.fetchers.cbr import CBR_MEETINGS, MacroContextProvider, MacroSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

_log = structlog.get_logger()

_DEFAULT_HISTORY_SIZE = 252


class MacroCacheService:
    """Cached macro context with daily refresh and CBR-day force-refresh.

    Stores rolling history for future ML feature engineering.
    Optionally persists snapshots to TimescaleDB when db_session_factory is provided.
    """

    def __init__(
        self,
        provider: MacroContextProvider,
        history_size: int = _DEFAULT_HISTORY_SIZE,
        db_session_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._provider = provider
        self._snapshot: MacroSnapshot | None = None
        self._last_refresh: datetime | None = None
        self._history: deque[MacroSnapshot] = deque(maxlen=history_size)
        self._db_session_factory = db_session_factory
        self._persist_task: asyncio.Task[None] | None = None

    def refresh(self) -> MacroSnapshot:
        """Fetch fresh macro snapshot. Called by scheduler. SYNC."""
        self._snapshot = self._provider.get_snapshot(as_of=datetime.now(tz=UTC).date())
        self._last_refresh = datetime.now(tz=UTC)
        self._history.append(self._snapshot)
        _log.info(
            "macro_cache_refreshed",
            key_rate=str(self._snapshot.key_rate),
            ruonia=str(self._snapshot.ruonia_7d_avg),
        )

        # Persist to DB if session factory is provided
        if self._db_session_factory is not None:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    self._persist_task = loop.create_task(
                        self._persist_snapshot(self._snapshot)
                    )
                except RuntimeError:
                    # No running loop — create one for the write
                    asyncio.run(self._persist_snapshot(self._snapshot))
            except Exception:
                _log.warning(
                    "macro_snapshot_persist_failed",
                    key_rate=str(self._snapshot.key_rate),
                )

        return self._snapshot

    async def _persist_snapshot(self, snapshot: MacroSnapshot) -> None:
        """Persist a MacroSnapshot to the database.

        Creates a MacroSnapshotModel and commits it via the async session factory.
        """
        now = datetime.now(tz=UTC)
        model = MacroSnapshotModel(
            timestamp=now,
            key_rate=snapshot.key_rate,
            ruonia_7d_avg=snapshot.ruonia_7d_avg,
            cpi_yoy=snapshot.cpi_yoy,
            last_cbr_decision=snapshot.last_cbr_decision,
            breakeven_inflation=snapshot.breakeven_inflation,
            yield_curve=(
                {k: str(v) for k, v in snapshot.yield_curve.items()}
                if snapshot.yield_curve
                else None
            ),
            usdrub=snapshot.usdrub,
            ofzin_indexation_coefficient=snapshot.ofzin_indexation_coefficient,
        )
        session = await self._db_session_factory()
        session.add(model)
        await session.commit()
        _log.info("macro_snapshot_persisted", timestamp=str(now))

    def get(self) -> MacroSnapshot | None:
        """Return cached snapshot. None only before first refresh."""
        return self._snapshot

    def get_history(self) -> list[MacroSnapshot]:
        """Return rolling macro history (for ML feature engineering)."""
        return list(self._history)

    def is_cbr_meeting_day(self) -> bool:
        """Check if today is a CBR key rate decision day.

        Uses direct calendar check (not days_to_next_cbr which has off-by-one).
        """
        today = datetime.now(tz=UTC).date()
        return any(m.date == today for m in CBR_MEETINGS)

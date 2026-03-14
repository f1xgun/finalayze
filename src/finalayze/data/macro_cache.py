"""Cached macro context with scheduled refresh (Layer 2).

Sync-only — no async. APScheduler BackgroundScheduler calls sync functions.
Future LiveMacroContextProvider with httpx must use asyncio.to_thread().
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime

import structlog

from finalayze.data.fetchers.cbr import CBR_MEETINGS, MacroContextProvider, MacroSnapshot

_log = structlog.get_logger()

_DEFAULT_HISTORY_SIZE = 252


class MacroCacheService:
    """Cached macro context with daily refresh and CBR-day force-refresh.

    Stores rolling history for future ML feature engineering.
    """

    def __init__(
        self,
        provider: MacroContextProvider,
        history_size: int = _DEFAULT_HISTORY_SIZE,
    ) -> None:
        self._provider = provider
        self._snapshot: MacroSnapshot | None = None
        self._last_refresh: datetime | None = None
        self._history: deque[MacroSnapshot] = deque(maxlen=history_size)

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
        return self._snapshot

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

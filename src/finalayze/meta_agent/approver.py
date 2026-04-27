"""Telegram /approve handler for FIX-severity meta-agent decisions (Phase 58-04).

SPEC §Requirement 7 + AC #12 + #17. Owns three responsibilities:

  1. ``handle_approve(short8, *, chat_id)`` — invoked by the Telegram
     webhook (``api/telegram_bot.py::handle_approve``) when the operator
     replies ``/approve <id8>``. Looks up the matching ``agent_decisions``
     row, validates state (severity='FIX', status='sent', created_at
     within ``approve_ttl_minutes``), and flips status to 'approved'
     (dispatches ``executor.execute_fix_spawn(decision)``) or 'expired'
     (TTL exceeded). Wrapped in fire-and-forget envelope (D-15) so
     persistence failure logs without raising.
  2. ``expire_overdue_fix_decisions()`` — invoked at the START of each
     meta-agent tick (``runner.run_one_tick()``) per D-14. Issues a
     single SQL UPDATE flipping FIX rows older than 30 min from 'sent'
     to 'expired'. Idempotent + bounded.
  3. ``_lookup_by_short8(short8)`` — async helper that queries the
     hypertable for the most-recent FIX 'sent' row whose UUID starts
     with the given short8 prefix.

Layer 6 — imports L0/L4 only:
  - ``finalayze.core.models.MetaAgentDecisionModel`` (L0 ORM)
  - ``finalayze.orchestration.db_persistence.TradingPersistence`` (L4)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from finalayze.meta_agent.executor import ActionExecutor
    from finalayze.orchestration.db_persistence import TradingPersistence

_log = structlog.get_logger()

# Module-level constants (PLR2004).
_DEFAULT_APPROVE_TTL_MINUTES = 30
_FIX_SEVERITY = "FIX"
_SENT_STATUS = "sent"
_APPROVED_STATUS = "approved"
_EXPIRED_STATUS = "expired"


def _now_utc() -> datetime:
    """Indirection for testability — tests monkeypatch this to control 'now'.

    A free function (not a method) so monkeypatch.setattr can replace it
    on the module without surgery on every approver instance.
    """
    return datetime.now(UTC)


class MetaAgentApprover:
    """Telegram /approve dispatcher + TTL sweep for FIX decisions.

    Constructor injects:
      - ``executor``: ``ActionExecutor`` — recipient of
        ``execute_fix_spawn(decision)`` on a successful approval.
      - ``persistence``: ``TradingPersistence`` — for status flips
        (``update_decision_status``) AND the sweep query (uses the
        background session factory directly).
      - ``approve_ttl_minutes``: TTL for /approve replies (default 30
        per SPEC line 64).

    Layer 6 — does NOT directly touch trading-critical paths.
    """

    def __init__(
        self,
        *,
        executor: ActionExecutor | Any,
        persistence: TradingPersistence | Any,
        approve_ttl_minutes: int = _DEFAULT_APPROVE_TTL_MINUTES,
    ) -> None:
        self._executor = executor
        self._persistence = persistence
        self._approve_ttl_minutes = approve_ttl_minutes

    # ── public API ────────────────────────────────────────────────────────

    async def handle_approve(self, short8: str, *, chat_id: str) -> None:
        """Process one /approve <short8> command from the Telegram webhook.

        SPEC AC #12 state machine:
          - Lookup → None: log ``meta_agent_approve_unknown_decision_id``,
            return.
          - State mismatch (severity!=FIX OR status!=sent): log
            ``meta_agent_approve_state_mismatch``, return.
          - Age > TTL: flip status='expired', return (NO spawn dispatch).
          - Else: flip status='approved', then await
            ``executor.execute_fix_spawn(row)``.

        D-15 fire-and-forget envelope: any exception is caught + logged
        as ``meta_agent_approve_persist_failed``; the webhook caller
        always returns 200 OK.
        """
        try:
            row = await self._lookup_by_short8(short8)
            if row is None:
                _log.info(
                    "meta_agent_approve_unknown_decision_id",
                    short8=short8,
                    chat_id=chat_id,
                )
                return

            # State mismatch — already approved/expired/rejected, or not FIX.
            if row.severity != _FIX_SEVERITY or row.status != _SENT_STATUS:
                _log.info(
                    "meta_agent_approve_state_mismatch",
                    short8=short8,
                    chat_id=chat_id,
                    severity_key=row.severity,
                    status_key=row.status,
                    decision_id_key=str(row.id),
                )
                return

            # TTL check — `created_at` is the canonical reply window per
            # SPEC line 64. Use the same monotonic-ish boundary as the
            # sweep query so a row sweeping in the next tick is consistent
            # with what handle_approve sees here.
            age = _now_utc() - row.created_at
            if age > timedelta(minutes=self._approve_ttl_minutes):
                _log.warning(
                    "meta_agent_approve_window_expired",
                    short8=short8,
                    chat_id=chat_id,
                    decision_id_key=str(row.id),
                    age_seconds=age.total_seconds(),
                )
                self._persistence.update_decision_status(
                    decision_id=row.id,
                    timestamp=row.timestamp,
                    status=_EXPIRED_STATUS,
                )
                return

            # Happy path: flip 'approved', dispatch spawn.
            _log.info(
                "meta_agent_approve_accepted",
                short8=short8,
                chat_id=chat_id,
                decision_id_key=str(row.id),
                age_seconds=age.total_seconds(),
            )
            self._persistence.update_decision_status(
                decision_id=row.id,
                timestamp=row.timestamp,
                status=_APPROVED_STATUS,
            )
            await self._executor.execute_fix_spawn(row)
        except Exception:
            _log.warning(
                "meta_agent_approve_persist_failed",
                short8=short8,
                chat_id=chat_id,
                exc_info=True,
            )

    async def expire_overdue_fix_decisions(self) -> int:
        """Sweep: flip FIX 'sent' rows older than ``approve_ttl_minutes``
        to 'expired'. Single bounded UPDATE; idempotent.

        SPEC AC #17 / D-14 — invoked at the START of each meta-agent
        tick (``MetaAgentRunner.run_one_tick()``) BEFORE snapshot
        collection.

        Returns the affected row count (0 when no rows match).
        """
        from sqlalchemy import text, update  # noqa: PLC0415

        from finalayze.core.models import MetaAgentDecisionModel  # noqa: PLC0415

        factory = self._persistence._get_bg_session_factory()
        async with factory() as session:
            stmt = (
                update(MetaAgentDecisionModel)
                .where(
                    MetaAgentDecisionModel.severity == _FIX_SEVERITY,
                    MetaAgentDecisionModel.status == _SENT_STATUS,
                    MetaAgentDecisionModel.created_at
                    <= text(
                        f"NOW() - INTERVAL '{self._approve_ttl_minutes} minutes'",
                    ),
                )
                .values(status=_EXPIRED_STATUS)
            )
            result = await session.execute(stmt)
            await session.commit()

        # ``CursorResult.rowcount`` exists on SQLAlchemy execute() results for
        # UPDATE / DELETE statements but is typed as a Union in stubs; cast
        # via getattr so mypy is happy with the test fakes (which set it
        # directly on a fake _FakeResult).
        rowcount = getattr(result, "rowcount", 0) or 0
        affected = int(rowcount)
        _log.info("meta_agent_approve_sweep", affected_count=affected)
        return affected

    # ── private helpers ──────────────────────────────────────────────────

    async def _lookup_by_short8(self, short8: str) -> Any | None:
        """Query agent_decisions for the most-recent FIX 'sent' row whose
        UUID starts with ``short8`` (8 hex chars).

        Returns the ORM row (or None). Uses the LIKE prefix match on the
        text-cast UUID (TimescaleDB-friendly; no UUID range query needed
        at the 2/day cap level — SPEC line 395 risk note).
        """
        from sqlalchemy import cast, desc, select  # noqa: PLC0415
        from sqlalchemy.types import String  # noqa: PLC0415

        from finalayze.core.models import MetaAgentDecisionModel  # noqa: PLC0415

        factory = self._persistence._get_bg_session_factory()
        async with factory() as session:
            stmt = (
                select(MetaAgentDecisionModel)
                .where(
                    MetaAgentDecisionModel.severity == _FIX_SEVERITY,
                    MetaAgentDecisionModel.status == _SENT_STATUS,
                    cast(MetaAgentDecisionModel.id, String).like(f"{short8}%"),
                )
                .order_by(desc(MetaAgentDecisionModel.created_at))
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

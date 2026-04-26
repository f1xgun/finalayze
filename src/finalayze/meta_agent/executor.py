"""ActionExecutor — Telegram-action leg of the meta-agent (Phase 58-02, META-05).

When ``meta_agent_dry_run=False`` and severity ∈ ``{WATCH, INVESTIGATE, FIX}``,
the executor sends one Telegram alert via the existing ``TelegramAlerter``
(Phase 57 persist-before-send envelope), enforces
``meta_agent_max_telegram_alerts_per_day``, and stamps the agent_decisions
row with ``status='sent'`` plus ``decision_metadata['telegram_alert_id']`` (or
``status='queued_capped'`` when capped).

The dry-run short-circuit is the FIRST line of ``execute()`` (PATTERNS AP-10):
no Telegram send, no persistence update, no session open before the gate.

Subprocess spawning for INVESTIGATE/FIX lands in Plans 58-03 / 58-04 — this
module is the Telegram-leg only.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from finalayze.meta_agent.classifier import Severity

if TYPE_CHECKING:
    from collections.abc import Callable

    from config.settings import Settings

    from finalayze.core.models import MetaAgentDecisionModel
    from finalayze.orchestration.db_persistence import TradingPersistence

_log = structlog.get_logger()


# SPEC §Requirement 5 mapping — severity to AlertPriority for Telegram routing.
# IntEnum values: CRITICAL=0, IMPORTANT=1, INFO=2 (lower = higher priority).
def _severity_to_priority(severity: str) -> Any:
    """Map a meta-agent severity to the Phase 57 AlertPriority IntEnum.

    Late import via PLC0415 keeps the executor module-load light — the
    enum lives in ``finalayze.api.alerts`` (Layer 0 alerter package).
    """
    from finalayze.api.alerts import AlertPriority  # noqa: PLC0415

    if severity == Severity.FIX.value:
        return AlertPriority.CRITICAL
    if severity == Severity.INVESTIGATE.value:
        return AlertPriority.IMPORTANT
    # WATCH (and any unexpected non-HEALTHY value, defensively).
    return AlertPriority.INFO


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one ``ActionExecutor.execute()`` call.

    ``skipped=True`` means no Telegram side-effect (dry-run, severity below
    threshold, or daily cap hit). ``reason`` records *why* in
    {``"dry_run"``, ``"severity_below_threshold"``, ``"telegram_cap_hit"``}.
    ``telegram_alert_id`` is populated only on a successful send.
    """

    skipped: bool
    reason: str | None
    telegram_alert_id: uuid.UUID | None


class ActionExecutor:
    """Telegram-action dispatcher for the meta-agent (Phase 58-02 META-05).

    Constructor injects ``settings`` (carries ``meta_agent_dry_run`` and
    ``meta_agent_max_telegram_alerts_per_day``), ``alerter``
    (``TelegramAlerter`` — sends Telegram + records the persist-before-send
    envelope), and ``persistence`` (``TradingPersistence`` — fire-and-forget
    ``update_decision_status`` envelope). Layer 6 — does NOT touch
    trading-critical paths.
    """

    def __init__(
        self,
        *,
        settings: Settings | Any,
        alerter: Any,
        persistence: TradingPersistence | Any,
        session_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._alerter = alerter
        self._persistence = persistence
        # Optional override for tests; production uses persistence's
        # background session factory.
        self._session_factory = session_factory

    def _open_session(self) -> Any:
        """Open an async session for the cap query.

        Production: delegates to ``persistence._get_bg_session_factory()``
        (mirrors the persist_alert envelope). Tests inject a fake via
        ``session_factory`` constructor kwarg or by monkeypatching this
        method directly.
        """
        if self._session_factory is not None:
            return self._session_factory()
        factory = self._persistence._get_bg_session_factory()
        return factory()

    async def execute(
        self,
        decision: MetaAgentDecisionModel | Any,
    ) -> ExecutionResult:
        """Dispatch one decision through the Telegram action leg.

        FIRST line is the dry-run gate (AP-10): zero side effects before
        the guard. Subsequent severity / cap / send branches land in
        Tasks 58-02-03 → 58-02-06.
        """
        # AP-10: dry-run short-circuit MUST be the first executable line.
        # No I/O, no session open, no log lookup before this gate.
        if self._settings.meta_agent_dry_run:
            _log.info(
                "meta_agent_executor_dry_run_skipped",
                decision_id_key=str(decision.id),
                severity_key=decision.severity,
            )
            return ExecutionResult(
                skipped=True, reason="dry_run", telegram_alert_id=None,
            )

        # SPEC §Requirement 5: only WATCH/INVESTIGATE/FIX trigger Telegram.
        # HEALTHY decisions are persisted (status='queued' from runner) and
        # the executor returns without action. The runner's persist already
        # captured the row; we do NOT touch persistence here.
        if decision.severity == Severity.HEALTHY.value:
            _log.info(
                "meta_agent_executor_severity_below_threshold",
                decision_id_key=str(decision.id),
                severity_key=decision.severity,
            )
            return ExecutionResult(
                skipped=True,
                reason="severity_below_threshold",
                telegram_alert_id=None,
            )

        # SPEC AC #8 + #9 — Telegram path with daily cap.
        # Open a background session for the cap query (PERSIST-05 envelope).
        async with self._open_session() as session:
            count = await self._telegram_count_today(session)

        # Cap enforcement branch lands in Task 58-02-06.

        # Build the message body — operator-friendly summary + rationale.
        message = self._build_message(decision)
        priority = _severity_to_priority(decision.severity)
        alert_type = f"meta_agent_{decision.severity}"

        ok, alert_id = await self._alerter._send(
            message,
            alert_type=alert_type,
            priority=priority,
        )
        if not ok or alert_id is None:
            _log.warning(
                "meta_agent_executor_telegram_send_failed",
                decision_id_key=str(decision.id),
                severity_key=decision.severity,
            )
            self._persistence.update_decision_status(
                decision_id=decision.id,
                timestamp=decision.timestamp,
                status="failed",
                outcome="telegram_send_failed",
            )
            return ExecutionResult(
                skipped=True, reason="telegram_send_failed", telegram_alert_id=None,
            )

        # Single UPDATE that flips status and stamps decision_metadata
        # atomically at the single-writer level (Task 58-02-01b).
        self._persistence.update_decision_status(
            decision_id=decision.id,
            timestamp=decision.timestamp,
            status="sent",
            metadata_patch={"telegram_alert_id": str(alert_id)},
        )
        _log.info(
            "meta_agent_executor_telegram_sent",
            decision_id_key=str(decision.id),
            severity_key=decision.severity,
            telegram_alert_id=str(alert_id),
            count_before=count,
        )
        return ExecutionResult(
            skipped=False, reason=None, telegram_alert_id=alert_id,
        )

    @staticmethod
    def _build_message(decision: MetaAgentDecisionModel | Any) -> str:
        """Compose the Telegram message body for one decision.

        Format: ``[meta-agent <SEVERITY>] <summary>\\n<rationale>``. Kept
        dependency-free so 58-03/58-04 can extend without breaking the
        Phase 57 escape conventions (D-09).
        """
        return (
            f"[meta-agent {decision.severity}] {decision.summary}\n{decision.rationale}"
        )

    async def _telegram_count_today(self, session: Any) -> int:
        """Return the count of meta-agent Telegram alerts already sent in the
        current UTC day (CONTEXT D-13 / SPEC AC #9 / RESEARCH §11.2).

        Cap resets at 00:00 UTC via ``date_trunc('day', NOW() AT TIME ZONE
        'UTC')`` — TIMESTAMPTZ columns auto-convert. The LIKE pattern
        ``meta_agent_%`` matches the ``alert_type`` value the executor
        passes to ``TelegramAlerter._send`` (Task 58-02-05).
        """
        from sqlalchemy import func, select, text  # noqa: PLC0415

        from finalayze.core.models import AlertModel  # noqa: PLC0415

        stmt = select(func.count()).select_from(AlertModel).where(
            AlertModel.alert_type.like("meta_agent_%"),
            AlertModel.timestamp >= text("date_trunc('day', NOW() AT TIME ZONE 'UTC')"),
        )
        result = await session.execute(stmt)
        return int(result.scalar_one())

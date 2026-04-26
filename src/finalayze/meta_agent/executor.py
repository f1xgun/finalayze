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

if TYPE_CHECKING:
    from config.settings import Settings

    from finalayze.core.models import MetaAgentDecisionModel
    from finalayze.orchestration.db_persistence import TradingPersistence

_log = structlog.get_logger()


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
    ) -> None:
        self._settings = settings
        self._alerter = alerter
        self._persistence = persistence

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

        # Subsequent branches added by Tasks 58-02-03 → 58-02-06.
        return ExecutionResult(
            skipped=True, reason="not_implemented", telegram_alert_id=None,
        )

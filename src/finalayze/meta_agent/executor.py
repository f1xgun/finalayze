"""ActionExecutor — Telegram + investigate-spawn legs (Phase 58-02 + 58-03).

58-02 (Telegram leg): when ``meta_agent_dry_run=False`` and severity ∈
``{WATCH, INVESTIGATE, FIX}``, send one Telegram alert via the existing
``TelegramAlerter`` (Phase 57 persist-before-send envelope), enforce
``meta_agent_max_telegram_alerts_per_day``, and stamp the agent_decisions
row with ``status='sent'`` plus ``decision_metadata['telegram_alert_id']``
(or ``status='queued_capped'`` when capped).

58-03 (investigate spawn): a separate ``execute_investigate_spawn(decision)``
entry point that, after the cap check, transitions the row through
``'spawned' → 'completed'/'failed'/'rejected'`` while invoking
``spawner.spawn_readonly(...)``. The runner dispatches this as a fire-and-
forget task tracked on ``self._spawn_tasks`` (RUF006).

Dry-run short-circuit on ``execute()`` is the FIRST line (PATTERNS AP-10):
no Telegram send, no persistence update, no session open before the gate.

FIX-spawn pipeline (worktree, /approve, allow-list validator) lands in 58-04.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from finalayze.meta_agent.classifier import Severity
from finalayze.meta_agent.exceptions import MetaAgentDeniedPathError, MetaAgentWorktreeError
from finalayze.meta_agent.path_validator import validate_fix_prompt
from finalayze.meta_agent.skill_loader import SkillSpec, load_skill
from finalayze.meta_agent.spawner import SpawnOutcome, spawn_fix, spawn_readonly
from finalayze.meta_agent.worktree import create_fix_worktree

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable

    from config.settings import Settings

    from finalayze.core.models import MetaAgentDecisionModel
    from finalayze.orchestration.db_persistence import TradingPersistence

_log = structlog.get_logger()

# ── 58-03 module-level constants (PLR2004) ─────────────────────────────────
_INVESTIGATE_SKILL_PATH = (
    Path(__file__).resolve().parents[3]
    / ".claude"
    / "skills"
    / "meta-agent-investigate"
    / "SKILL.md"
)
_OUTCOME_TEXT_MAX_BYTES = 64 * 1024  # D-06 cap on persisted outcome text
_INVEST_TIMEOUT_S = 300  # SPEC §Requirement 6 — 300s investigate timeout

# 58-04 module-level constants.
_FIX_SKILL_PATH = (
    Path(__file__).resolve().parents[3] / ".claude" / "skills" / "meta-agent-fix" / "SKILL.md"
)
_FIX_TIMEOUT_S = 600  # SPEC §Requirement 7 — 600s fix-spawn timeout
_FIX_SHORT8_LEN = 8


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
                skipped=True,
                reason="dry_run",
                telegram_alert_id=None,
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

        # SPEC AC #9: cap is "at most N per day" — strict >= so the (cap+1)th
        # decision in a UTC day records 'queued_capped' with zero side
        # effects (no Telegram, no metadata stamp).
        if count >= self._settings.meta_agent_max_telegram_alerts_per_day:
            _log.warning(
                "meta_agent_executor_telegram_cap_hit",
                decision_id_key=str(decision.id),
                severity_key=decision.severity,
                count=count,
                cap=self._settings.meta_agent_max_telegram_alerts_per_day,
            )
            self._persistence.update_decision_status(
                decision_id=decision.id,
                timestamp=decision.timestamp,
                status="queued_capped",
            )
            return ExecutionResult(
                skipped=True,
                reason="telegram_cap_hit",
                telegram_alert_id=None,
            )

        # Build the message body — operator-friendly summary + rationale.
        message = self._build_message(decision)
        priority = _severity_to_priority(decision.severity)
        alert_type = f"meta_agent_{decision.severity}"

        # alerter._send uses an httpx.AsyncClient bound to the uvicorn event loop.
        # When called from _async_loop (meta-agent tick), bridge via wrap_future
        # so the HTTP request executes in the loop where the client was created.
        main_loop = getattr(self, "_main_loop", None)
        if main_loop is not None and main_loop.is_running():
            _coro = self._alerter._send(message, alert_type=alert_type, priority=priority)
            _fut = asyncio.run_coroutine_threadsafe(_coro, main_loop)
            ok, alert_id = await asyncio.wrap_future(_fut)
        else:
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
                skipped=True,
                reason="telegram_send_failed",
                telegram_alert_id=None,
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
            skipped=False,
            reason=None,
            telegram_alert_id=alert_id,
        )

    @staticmethod
    def _build_message(decision: MetaAgentDecisionModel | Any) -> str:
        """Compose the Telegram message body for one decision.

        Format (WATCH/INVESTIGATE): ``[meta-agent <SEVERITY>] <summary>\\n
        <rationale>``.

        Format (FIX, 58-04): adds a header banner with the decision_id
        short8 and the literal ``/approve <short8>`` instruction so the
        operator can copy-paste the reply (SPEC §Requirement 7 + AC #12).
        """
        if decision.severity == Severity.FIX.value:
            short8 = str(decision.id)[:_FIX_SHORT8_LEN]
            return (
                f"\U0001f6a8 [meta-agent FIX proposed]\n"
                f"decision_id={short8}\n"
                f"Reply /approve {short8} within 30 min to authorise.\n\n"
                f"Summary: {decision.summary}\n"
                f"Rationale: {decision.rationale}"
            )
        return f"[meta-agent {decision.severity}] {decision.summary}\n{decision.rationale}"

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

        stmt = (
            select(func.count())
            .select_from(AlertModel)
            .where(
                AlertModel.alert_type.like("meta_agent_%"),
                AlertModel.timestamp >= text("date_trunc('day', NOW() AT TIME ZONE 'UTC')"),
            )
        )
        result = await session.execute(stmt)
        return int(result.scalar_one())

    # ── 58-03 META-06: read-only investigate spawn ──────────────────────────

    async def execute_investigate_spawn(
        self,
        decision: MetaAgentDecisionModel | Any,
    ) -> None:
        """Dispatch a read-only investigation spawn for ``decision``.

        SPEC AC #10 + #11. State machine (SPEC line 57):
          1. Cap query (``_spawn_count_today_async('INVESTIGATE')``).
          2. If ``count >= settings.meta_agent_max_spawns_per_day``:
             flip ``status='rejected'`` with ``outcome='spawn_cap_exceeded'``
             and emit ``meta_agent_spawn_cap_exceeded``. Return.
          3. Else: flip ``status='spawned'`` BEFORE invoking the subprocess
             (so the next tick's cap query sees the in-flight row).
          4. Load ``meta-agent-investigate`` skill from
             ``.claude/skills/meta-agent-investigate/SKILL.md``.
          5. ``await spawner.spawn_readonly(prompt, decision_id=...,
             timeout_s=300)``.
          6. Build ``outcome_text`` = exit_code + stdout + stderr, truncated
             to 64 KiB. Flip ``status='completed'`` on exit_code==0 (and not
             timed_out), else ``'failed'``.

        NEVER raises — all failure modes flip the decision to ``'failed'``
        with a structlog warning. The runner (Plan 58-01 / 58-02) treats
        executor failures as non-fatal so the scheduler keeps ticking.
        """
        decision_id = decision.id
        timestamp = decision.timestamp

        # 1. Cap query.
        try:
            count = await self._persistence._spawn_count_today_async(
                "INVESTIGATE",
            )
        except Exception:
            _log.warning(
                "meta_agent_spawn_cap_query_failed",
                decision_id_key=str(decision_id),
                exc_info=True,
            )
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="failed",
                outcome="spawn_cap_query_failed",
            )
            return

        # 2. Cap exceeded → reject and return.
        cap = self._settings.meta_agent_max_spawns_per_day
        if count >= cap:
            _log.warning(
                "meta_agent_spawn_cap_exceeded",
                decision_id_key=str(decision_id),
                spawn_type="investigate",
                count=count,
                cap=cap,
            )
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="rejected",
                outcome="spawn_cap_exceeded",
            )
            return

        # 3. Mark 'spawned' BEFORE invoking — the next tick's cap query
        #    must see this row even if the spawn is still running.
        self._persistence.update_decision_status(
            decision_id=decision_id,
            timestamp=timestamp,
            status="spawned",
        )

        # 4. Load the investigate skill (system prompt + spawner directives).
        try:
            skill = load_skill(_INVESTIGATE_SKILL_PATH)
        except (FileNotFoundError, ValueError):
            _log.warning(
                "meta_agent_skill_load_failed",
                decision_id_key=str(decision_id),
                skill_path=str(_INVESTIGATE_SKILL_PATH),
                exc_info=True,
            )
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="failed",
                outcome="skill_missing",
            )
            return

        prompt = self._build_invest_prompt(decision, skill)

        # 5. Spawn — never raises (the spawner swallows everything except
        #    CancelledError, which the runner-level exception guard handles).
        try:
            outcome = await spawn_readonly(
                prompt,
                decision_id=decision_id,
                timeout_s=_INVEST_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            # Killswitch fired (Plan 58-05). Mark 'failed' and re-raise so
            # the cancellation propagates to the task supervisor.
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="failed",
                outcome="killed_by_killswitch",
            )
            raise
        except Exception:
            _log.warning(
                "meta_agent_spawn_failed",
                decision_id_key=str(decision_id),
                exc_info=True,
            )
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="failed",
                outcome="spawn_invocation_failed",
            )
            return

        # 6. Build outcome text (D-06: 64 KiB cap on persisted column).
        outcome_text = _build_outcome_text(outcome)
        terminal_status = (
            "completed" if outcome.exit_code == 0 and not outcome.timed_out else "failed"
        )
        self._persistence.update_decision_status(
            decision_id=decision_id,
            timestamp=timestamp,
            status=terminal_status,
            outcome=outcome_text,
        )

    @staticmethod
    def _build_invest_prompt(
        decision: MetaAgentDecisionModel | Any,
        skill: SkillSpec,
    ) -> str:
        """Build the user-turn prompt threaded into ``claude -p`` (D-10).

        Composes the skill's system-prompt body with a JSON-stringified
        snapshot summary derived from the decision row (severity, summary,
        rationale). The skill body is appended so the spawned CLI session
        carries both context.
        """
        import json  # noqa: PLC0415

        snapshot_payload = {
            "decision_id": str(decision.id),
            "timestamp": str(decision.timestamp),
            "severity": decision.severity,
            "summary": decision.summary,
            "rationale": decision.rationale,
        }
        return (
            f"{skill.system_prompt}\n\n"
            f"## Snapshot\n\n"
            f"```json\n{json.dumps(snapshot_payload, indent=2)}\n```\n"
        )

    # ── 58-04 META-07: FIX-spawn pipeline (worktree + validator + cap) ─────

    async def execute_fix_spawn(
        self,
        decision: MetaAgentDecisionModel | Any,
    ) -> None:
        """Dispatch the FIX-spawn pipeline (called by approver after /approve).

        SPEC AC #12 + #13 + #14 state machine:
          1. Cap query (``_spawn_count_today_async('FIX')``).
             - If ``count >= settings.meta_agent_max_fix_spawns_per_day``:
               flip ``status='rejected', outcome='fix_spawn_cap_exceeded'``.
               Return.
          2. Load the meta-agent-fix skill (path validator pulls
             ``skill.denied_paths``).
          3. Build the fix prompt.
          4. ``validate_fix_prompt(prompt, denied_paths=...)``.
             - On ``MetaAgentDeniedPathError``: flip
               ``status='rejected', outcome='denied_path:<exc>'``. Return.
          5. ``create_fix_worktree(short8)``.
             - On ``MetaAgentWorktreeError``: flip
               ``status='failed', outcome='worktree_create_failed:<exc>'``.
               Return.
          6. Mark ``status='spawned'``.
          7. ``await spawn_fix(prompt, decision_id=..., cwd=worktree, ...)``.
          8. Flip ``status='completed'`` (exit=0 + not timed_out) or
             ``'failed'``; store outcome text.

        NEVER raises (except CancelledError, propagated from the spawner).
        """
        decision_id = decision.id
        timestamp = decision.timestamp
        short8 = str(decision_id)[:_FIX_SHORT8_LEN]

        # 1. Cap query.
        try:
            count = await self._persistence._spawn_count_today_async("FIX")
        except Exception:
            _log.warning(
                "meta_agent_fix_spawn_cap_query_failed",
                decision_id_key=str(decision_id),
                exc_info=True,
            )
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="failed",
                outcome="fix_spawn_cap_query_failed",
            )
            return

        cap = self._settings.meta_agent_max_fix_spawns_per_day
        if count >= cap:
            _log.warning(
                "meta_agent_fix_spawn_cap_exceeded",
                decision_id_key=str(decision_id),
                spawn_type="fix",
                count=count,
                cap=cap,
            )
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="rejected",
                outcome="fix_spawn_cap_exceeded",
            )
            return

        # 2. Load the fix skill.
        try:
            skill = load_skill(_FIX_SKILL_PATH)
        except (FileNotFoundError, ValueError):
            _log.warning(
                "meta_agent_fix_skill_load_failed",
                decision_id_key=str(decision_id),
                skill_path=str(_FIX_SKILL_PATH),
                exc_info=True,
            )
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="failed",
                outcome="fix_skill_missing",
            )
            return

        # 3. Build the prompt.
        prompt = self._build_fix_prompt(decision, skill)

        # 4. Pre-spawn validator — scan only the decision-derived content
        # (summary + rationale), not the trusted skill body which intentionally
        # lists denied paths as examples of what NOT to touch.
        decision_payload = f"{decision.summary}\n{decision.rationale}"
        try:
            validate_fix_prompt(decision_payload, denied_paths=skill.denied_paths)
        except MetaAgentDeniedPathError as exc:
            _log.warning(
                "meta_agent_fix_denied_path",
                decision_id_key=str(decision_id),
                exc_info=False,
                error=str(exc),
            )
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="rejected",
                outcome=f"denied_path:{exc}",
            )
            return

        # 5. Create the worktree.
        try:
            worktree = create_fix_worktree(short8)
        except MetaAgentWorktreeError as exc:
            _log.warning(
                "meta_agent_fix_worktree_failed",
                decision_id_key=str(decision_id),
                error=str(exc),
            )
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="failed",
                outcome=f"worktree_create_failed:{exc}",
            )
            return

        # 6. Mark 'spawned' BEFORE invoking — next tick's cap query sees this.
        self._persistence.update_decision_status(
            decision_id=decision_id,
            timestamp=timestamp,
            status="spawned",
        )

        # 7. Spawn — never raises (except CancelledError, propagated).
        try:
            outcome = await spawn_fix(
                prompt,
                decision_id=decision_id,
                cwd=worktree,
                allowed_paths=skill.allowed_paths,
                denied_paths=skill.denied_paths,
                timeout_s=_FIX_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="failed",
                outcome="killed_by_killswitch",
            )
            raise
        except Exception:
            _log.warning(
                "meta_agent_fix_spawn_failed",
                decision_id_key=str(decision_id),
                exc_info=True,
            )
            self._persistence.update_decision_status(
                decision_id=decision_id,
                timestamp=timestamp,
                status="failed",
                outcome="fix_spawn_invocation_failed",
            )
            return

        # 8. Build outcome text + terminal status.
        outcome_text = _build_outcome_text(outcome)
        terminal_status = (
            "completed" if outcome.exit_code == 0 and not outcome.timed_out else "failed"
        )
        self._persistence.update_decision_status(
            decision_id=decision_id,
            timestamp=timestamp,
            status=terminal_status,
            outcome=outcome_text,
        )

    @staticmethod
    def _build_fix_prompt(
        decision: MetaAgentDecisionModel | Any,
        skill: SkillSpec,
    ) -> str:
        """Build the user-turn prompt for the FIX spawn.

        Mirrors ``_build_invest_prompt`` but adds an explicit "FIX" banner
        and the allowed-path list so the LLM knows the path constraints
        upfront. The pre-spawn validator (``validate_fix_prompt``) checks
        this prompt against ``skill.denied_paths`` before any subprocess
        is created.
        """
        import json  # noqa: PLC0415

        snapshot_payload = {
            "decision_id": str(decision.id),
            "timestamp": str(decision.timestamp),
            "severity": decision.severity,
            "summary": decision.summary,
            "rationale": decision.rationale,
            "allowed_paths": skill.allowed_paths,
        }
        return (
            f"{skill.system_prompt}\n\n"
            f"## FIX Snapshot\n\n"
            f"```json\n{json.dumps(snapshot_payload, indent=2)}\n```\n"
        )


def _build_outcome_text(outcome: SpawnOutcome) -> str:
    """Format a SpawnOutcome into the ``decision.outcome`` text column.

    Format: ``<exit_code=N>\\n<stdout>\\n---\\n<stderr>`` (SPEC line 57).
    Truncated to 64 KiB total (D-06) with marker.
    """
    text = (
        f"<exit_code={outcome.exit_code} timed_out={outcome.timed_out} "
        f"killed_by_killswitch={outcome.killed_by_killswitch}>\n"
        f"{outcome.stdout}\n---\n{outcome.stderr}"
    )
    if len(text) <= _OUTCOME_TEXT_MAX_BYTES:
        return text
    # Truncate at the byte limit, keep the marker.
    return text[:_OUTCOME_TEXT_MAX_BYTES] + "\n[truncated_at=64KiB]"

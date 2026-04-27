"""Real-CLI smoke test for `spawn_readonly` against the operator's `claude` binary.

Closes the argv-drift coverage gap noted in `/gsd-add-tests 58`: every test in
`tests/unit/meta_agent/test_spawner.py` monkey-patches
`asyncio.create_subprocess_exec`, so a Claude CLI flag rename or argv ordering
change would silently break production while every unit test stays green.

This test is gated on the `claude` binary being on `$PATH` (operator's local
Max-subscription install). CI runners without the binary skip cleanly.

The test invokes `spawn_readonly` with a trivial prompt that should exit
quickly. It does NOT validate model output — it validates that the spawner
constructs a working argv that the real CLI accepts (no `unknown flag`,
no `argument required`, no immediate non-zero exit from argv parsing).
"""

from __future__ import annotations

import asyncio
import shutil
import uuid

import pytest

from finalayze.meta_agent.spawner import spawn_readonly

pytestmark = pytest.mark.integration

_CLAUDE_BIN = shutil.which("claude")
_SMOKE_PROMPT = "Reply with exactly the literal string OK and nothing else."
_SMOKE_TIMEOUT_S = 60


@pytest.mark.skipif(
    _CLAUDE_BIN is None,
    reason="claude CLI not on PATH; smoke test requires the operator's Max-subscription install",
)
def test_spawn_readonly_invokes_real_claude_cli() -> None:
    """`spawn_readonly` must construct argv that the real `claude` CLI accepts.

    Pass criteria (any of):
      - exit_code == 0 with non-empty stdout (CLI accepted argv and produced output)
      - exit_code != 0 BUT stderr does NOT contain argv-parse error markers
        (the CLI ran, may have failed for runtime reasons like rate-limit or
        auth, but the spawner's argv construction is sound)

    Hard fail criteria (the bug this test catches):
      - stderr contains 'unknown flag', 'unknown option', 'unrecognized argument',
        'usage:', 'error: argument', or 'invalid choice' — these signal argv
        drift between spawner and CLI
      - timed_out is True with no captured output (CLI hung at argv parse)
    """
    decision_id = uuid.uuid4()

    outcome = asyncio.run(
        spawn_readonly(
            prompt=_SMOKE_PROMPT,
            decision_id=decision_id,
            timeout_s=_SMOKE_TIMEOUT_S,
        )
    )

    argv_parse_error_markers = (
        "unknown flag",
        "unknown option",
        "unrecognized argument",
        "usage:",
        "error: argument",
        "invalid choice",
    )
    stderr_lower = outcome.stderr.lower()
    detected_markers = [m for m in argv_parse_error_markers if m in stderr_lower]
    assert not detected_markers, (
        f"Spawner argv drift detected — claude CLI rejected the argv. "
        f"Markers found: {detected_markers}. stderr: {outcome.stderr[:500]!r}"
    )

    assert not (outcome.timed_out and not outcome.stdout and not outcome.stderr), (
        "spawn timed out with zero output — likely argv parse hang. "
        f"timed_out={outcome.timed_out}, exit_code={outcome.exit_code}"
    )

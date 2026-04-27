"""Tests for meta_agent.path_validator (Phase 58-04 Task 01).

SPEC AC #13 — pre-spawn validator rejects FIX prompts referencing
``src/finalayze/risk/``, ``src/finalayze/execution/``, ``src/finalayze/core/``.
Substring scan; case-sensitive.
"""

from __future__ import annotations

import pytest

# Module-level constants (PLR2004).
_DENIED_PATHS = [
    "src/finalayze/risk/",
    "src/finalayze/execution/",
    "src/finalayze/core/",
]
_ALLOWED_PATHS = [
    "src/finalayze/strategies/presets/",
    "config/segments.py",
]


def test_validate_fix_prompt_rejects_risk_path() -> None:
    """SPEC AC #13: a prompt referencing ``src/finalayze/risk/manager.py``
    raises ``MetaAgentDeniedPathError`` with the offending path in the message.
    """
    from finalayze.meta_agent.exceptions import MetaAgentDeniedPathError
    from finalayze.meta_agent.path_validator import validate_fix_prompt

    with pytest.raises(MetaAgentDeniedPathError) as exc_info:
        validate_fix_prompt(
            "Edit src/finalayze/risk/manager.py to fix the cap",
            denied_paths=_DENIED_PATHS,
        )
    # Offending path appears in the exception message.
    assert "src/finalayze/risk/" in str(exc_info.value), (
        f"expected denied path in message, got {exc_info.value!r}"
    )


def test_validate_fix_prompt_rejects_execution_and_core_paths() -> None:
    """SPEC AC #13: all three denied substrings (risk/, execution/, core/)
    are rejected independently.
    """
    from finalayze.meta_agent.exceptions import MetaAgentDeniedPathError
    from finalayze.meta_agent.path_validator import validate_fix_prompt

    for denied in _DENIED_PATHS:
        prompt = f"Please modify {denied}some_file.py"
        with pytest.raises(MetaAgentDeniedPathError):
            validate_fix_prompt(prompt, denied_paths=_DENIED_PATHS)


def test_validate_fix_prompt_accepts_presets_path() -> None:
    """SPEC AC #13: a prompt referencing the allow-listed
    ``src/finalayze/strategies/presets/`` path passes the validator.
    """
    from finalayze.meta_agent.path_validator import validate_fix_prompt

    # Should not raise.
    result = validate_fix_prompt(
        "Tweak src/finalayze/strategies/presets/momentum.yaml threshold",
        denied_paths=_DENIED_PATHS,
    )
    assert result is None


def test_validate_fix_prompt_accepts_segments_config() -> None:
    """SPEC AC #13: ``config/segments.py`` (the other allow-listed path) passes."""
    from finalayze.meta_agent.path_validator import validate_fix_prompt

    result = validate_fix_prompt(
        "Adjust config/segments.py weight for ru_finance",
        denied_paths=_DENIED_PATHS,
    )
    assert result is None


def test_validate_fix_prompt_with_empty_denied_list_accepts_anything() -> None:
    """Edge case: empty denied list means no rejection (defensive default)."""
    from finalayze.meta_agent.path_validator import validate_fix_prompt

    result = validate_fix_prompt(
        "Edit src/finalayze/risk/manager.py",
        denied_paths=[],
    )
    assert result is None


def test_validate_fix_prompt_is_case_sensitive() -> None:
    """SPEC AC #13: substring match is case-sensitive — uppercase variant
    does NOT trigger rejection (paths are real paths, not free text).
    """
    from finalayze.meta_agent.path_validator import validate_fix_prompt

    # Uppercase variant should NOT be rejected by the case-sensitive scan.
    result = validate_fix_prompt(
        "Edit SRC/FINALAYZE/RISK/manager.py",
        denied_paths=_DENIED_PATHS,
    )
    assert result is None

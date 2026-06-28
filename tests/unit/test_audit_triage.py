"""Tests for the autonomous-audit risk-class triage (safety core).

The gate is default-risky: only docs/tests/uv.lock changes auto-merge; anything
touching strategy/risk/ML/execution/core/config/migrations/CI escalates.
"""

from __future__ import annotations

from scripts.audit_triage import RiskClass, classify_change


def test_docs_only_is_safe() -> None:
    v = classify_change(["docs/audit/report.md", "README.md", "src/finalayze/core/AGENTS.md"])
    assert v.risk_class is RiskClass.SAFE


def test_tests_only_is_safe() -> None:
    v = classify_change(["tests/unit/test_foo.py", "tests/integration/test_bar.py"])
    assert v.risk_class is RiskClass.SAFE


def test_lockfile_refresh_is_safe() -> None:
    assert classify_change(["uv.lock"]).risk_class is RiskClass.SAFE


def test_src_change_is_risky() -> None:
    v = classify_change(["src/finalayze/risk/position_sizing.py"])
    assert v.risk_class is RiskClass.RISKY
    assert v.offending_path == "src/finalayze/risk/position_sizing.py"


def test_mixed_safe_and_risky_is_risky() -> None:
    # One risky path among safe ones must tip the whole change to risky.
    v = classify_change(["docs/x.md", "src/finalayze/strategies/momentum.py", "tests/test_x.py"])
    assert v.risk_class is RiskClass.RISKY
    assert "strategies" in (v.offending_path or "")


def test_config_and_migrations_are_risky() -> None:
    assert classify_change(["config/segments.py"]).risk_class is RiskClass.RISKY
    assert classify_change(["alembic/versions/015_x.py"]).risk_class is RiskClass.RISKY


def test_ci_and_pyproject_are_risky() -> None:
    # CI gates and tool/dep config must never auto-merge (could weaken the gates).
    assert classify_change([".github/workflows/ci.yml"]).risk_class is RiskClass.RISKY
    assert classify_change(["pyproject.toml"]).risk_class is RiskClass.RISKY


def test_empty_change_is_risky() -> None:
    # Never a silent no-op auto-merge.
    assert classify_change([]).risk_class is RiskClass.RISKY


def test_cli_exit_codes() -> None:
    from scripts.audit_triage import main

    assert main(["docs/x.md"]) == 0
    assert main(["src/finalayze/execution/tinkoff_broker.py"]) == 2

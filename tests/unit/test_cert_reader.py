"""Phase 87: the read-only binding-cert reader -- HONEST, derived-from-the-real-cert, fail-closed.

Anti-hollow contract: every metric assertion reads the REAL committed cert summary.json and compares
the surfaced value against it -- never a hardcoded fixture. HARD_FAIL must be surfaced as HARD_FAIL,
the full-window best-naive bar must be the equity benchmark (NOT the deposit -- the deposit wins
only the high_rate sub-window), and NO fabricated rate threshold may appear in the framing.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from finalayze.backtest import cert_reader
from finalayze.backtest.cert_reader import (
    CertDecision,
    load_latest_cert,
    parse_cert_json,
    select_latest_cert_dir,
)
from finalayze.core.exceptions import CertNotFoundError

_LATEST_DIRNAME = "allocation-gate-73-20260622T220628Z"


def _committed_cert() -> dict:
    """Load the real latest committed cert JSON (the source of truth for the anti-hollow tests)."""
    return json.loads((select_latest_cert_dir() / "summary.json").read_text(encoding="utf-8"))


def test_default_iter_dir_resolves_to_repo_results() -> None:
    """The DEFAULT (un-injected) _ITER_DIR points at <repo>/results/iterations (parents[3]), and
    select_latest_cert_dir() with no arg finds the real committed cert (the parents[2] path bug)."""
    assert cert_reader._ITER_DIR.is_dir()
    assert cert_reader._ITER_DIR.name == "iterations"
    assert cert_reader._ITER_DIR.parent.name == "results"
    assert select_latest_cert_dir().name == _LATEST_DIRNAME


def test_select_latest_cert_dir_picks_most_recent() -> None:
    """The lexicographically/chronologically last allocation-gate-73-* dir is selected; stable."""
    chosen = select_latest_cert_dir()
    assert chosen.name == _LATEST_DIRNAME
    assert select_latest_cert_dir() == chosen  # deterministic across calls


def test_parse_cert_json_reads_required_keys() -> None:
    """parse_cert_json returns the dict with all required keys + native JSON types."""
    data = parse_cert_json(select_latest_cert_dir())
    for key in cert_reader._REQUIRED_KEYS:
        assert key in data
    assert isinstance(data["per_profile"]["balanced"]["sharpe"], float)
    assert isinstance(data["phase_verdict"], str)


def test_surfaced_phase_verdict_equals_committed_cert() -> None:
    """ANTI-HOLLOW: the surfaced verdict BYTE-matches the committed cert -- it is not a literal."""
    decision = load_latest_cert()
    assert decision.phase_verdict == _committed_cert()["phase_verdict"]


def test_hard_fail_not_softened() -> None:
    """When HARD_FAIL the headline says so and NEVER claims a pass/beat (constraint 6)."""
    decision = load_latest_cert()
    if decision.phase_verdict == "HARD_FAIL":
        assert "does not beat" in decision.headline
        low = decision.headline.lower()
        assert "pass" not in low
        assert "beats" not in low
        assert "outperforms" not in low


def test_full_window_best_naive_is_equity_not_deposit() -> None:
    """The full-window best-naive bar is equity_100, NOT the deposit (the §1 honesty trap)."""
    cert = _committed_cert()
    decision = load_latest_cert()
    assert decision.best_naive_sharpe_full == cert["per_profile"]["balanced"]["best_naive_sharpe"]
    assert decision.best_naive_sharpe_full == cert["naive"]["equity_100_sharpe"]
    assert decision.best_naive_sharpe_full != cert["naive"]["deposit_100_sharpe"]


def test_high_rate_story_deposit_wins_per_regime() -> None:
    """The deposit-wins (+0.89) claim is a PER-REGIME (high_rate) row, sourced + sign-derived."""
    cert = _committed_cert()
    high = next(s for s in load_latest_cert().regime_stories if s.unit_key == "high_rate")
    hr = cert["per_regime"]["high_rate"]["balanced"]
    assert high.best_naive_sharpe == hr["best_naive_sharpe"]
    assert high.best_naive_sharpe > 0  # the deposit wins here...
    assert high.allocation_sharpe < 0  # ...while the allocator is negative
    assert high.unit_label == "high_rate"


def test_easing_story_all_sleeves_negative() -> None:
    """In the single easing cycle both the allocation and the best naive are negative (derived)."""
    cert = _committed_cert()
    easing = next(s for s in load_latest_cert().regime_stories if s.unit_key == "early_cut")
    assert easing.unit_label == "easing"
    assert easing.allocation_sharpe < 0
    assert easing.best_naive_sharpe < 0
    assert easing.allocation_sharpe == cert["per_regime"]["early_cut"]["balanced"]["sharpe"]


def test_escalation_passthrough() -> None:
    """escalation is read verbatim from the cert, not recomputed."""
    assert load_latest_cert().escalation == _committed_cert()["escalation"]


def test_n1_caveat_passthrough_and_in_framing() -> None:
    """n1_caveat is verbatim; when True the honest framing names the N=1 limitation."""
    decision = load_latest_cert()
    assert decision.n1_caveat == _committed_cert()["n1_caveat"]
    if decision.n1_caveat:
        assert "N=1" in decision.when_framing


def test_high_rate_caveat_verbatim() -> None:
    """The gate's high_rate caveat is surfaced byte-for-byte."""
    assert load_latest_cert().high_rate_caveat == _committed_cert()["high_rate_caveat"]


def test_when_framing_has_no_fabricated_rate_threshold() -> None:
    """The framing emits NO 'rates below X%' threshold (constraint 2) and is flagged qualitative."""
    framing = load_latest_cert().when_framing
    threshold = re.compile(r"below\s+\d+(\.\d+)?\s*%|rates?\s+(under|below)\s+\d", re.IGNORECASE)
    assert threshold.search(framing) is None
    assert "no rate threshold" in framing.lower()
    assert "qualitative" in framing.lower()


def test_cert_timestamp_and_staleness_from_dirname() -> None:
    """The timestamp parses from the dir name; staleness is computed against an injected today."""
    decision = load_latest_cert(today=date(2026, 7, 2))
    assert decision.cert_timestamp == "2026-06-22T22:06:28+00:00"
    assert decision.staleness_days == 10  # 2026-06-22 -> 2026-07-02


def test_load_latest_cert_fail_closed_when_no_dir(tmp_path: Path) -> None:
    """An empty/missing iter dir raises CertNotFoundError -- never fabricates numbers."""
    with pytest.raises(CertNotFoundError):
        load_latest_cert(iter_dir=tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(CertNotFoundError):
        load_latest_cert(iter_dir=tmp_path / "empty")


def test_parse_cert_json_fail_closed_malformed_or_missing_key(tmp_path: Path) -> None:
    """Malformed JSON or a missing required key fails closed with a diagnostic."""
    bad = tmp_path / "allocation-gate-73-20260101T000000Z"
    bad.mkdir()
    (bad / "summary.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CertNotFoundError, match="malformed"):
        parse_cert_json(bad)
    (bad / "summary.json").write_text(json.dumps({"git_sha": "x"}), encoding="utf-8")
    with pytest.raises(CertNotFoundError, match="missing required keys"):
        parse_cert_json(bad)


def test_returns_frozen_cert_decision() -> None:
    """load_latest_cert returns an immutable CertDecision (audit-safe)."""
    decision = load_latest_cert()
    assert isinstance(decision, CertDecision)
    with pytest.raises((AttributeError, TypeError)):
        decision.phase_verdict = "PASS"  # type: ignore[misc]


def test_cert_reader_does_not_import_frozen_allocator_or_gate() -> None:
    """Import-graph lock: the reader must NOT import the FROZEN gate/allocator (no re-run drift).

    Inspects actual import statements (AST), not docstring prose -- the reader only READS the
    committed artifact and must never pull the gate/allocator logic (which would risk a re-run).
    """
    import ast

    tree = ast.parse(Path(cert_reader.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    forbidden = ("allocation_gate", "orchestration.allocation", "run_allocation_gate")
    offenders = [m for m in imported if any(f in m for f in forbidden)]
    assert offenders == [], f"cert_reader must not import the frozen gate/allocator: {offenders}"

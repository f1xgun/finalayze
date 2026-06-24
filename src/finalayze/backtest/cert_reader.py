"""Read-only reader for the latest committed allocation-gate binding cert (Phase 87).

The FROZEN binding gate (``allocation_gate.py``, run by ``scripts/run_allocation_gate.py``) writes
a committed ``results/iterations/allocation-gate-73-<ts>/summary.json`` artifact carrying the honest
per-profile + per-regime verdicts (HARD_FAIL), the naive benchmarks, the regime split, the derived
escalation + n1_caveat. This module hydrates the LATEST committed cert into a frozen
``CertDecision`` so a read-only decision-support view can surface the verdict + the deposit-anchor
benchmark ALONGSIDE the allocator's recommendation.

HONESTY CONTRACT (the deliverable): every number + verdict is DERIVED from the real committed cert
file. There is NO pre-baked verdict literal and NO fabricated number anywhere -- a hardcoded verdict
is a fixture, not a measurement (the Phase 72/75 anti-hollow lesson). HARD_FAIL is surfaced AS
HARD_FAIL (never softened), and NO rate-threshold ("rates below X%") is ever produced -- the cert
computes none. This module imports NOTHING from the allocator/gate logic; it only reads the artifact
and raises an L0 ``CertNotFoundError`` fail-closed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finalayze.core.exceptions import CertNotFoundError

if TYPE_CHECKING:
    from collections.abc import Mapping

# <repo>/results/iterations -- this module is at <repo>/src/finalayze/backtest/cert_reader.py, so
# the repo root is parents[3] (backtest=0, finalayze=1, src=2, <repo>=3). NOTE: NOT parents[2]
# (=src/), which would point the reader at a non-existent src/results and always fail closed.
_ITER_DIR = Path(__file__).resolve().parents[3] / "results" / "iterations"
_RUN_PREFIX = "allocation-gate-73"
# dirname -> YYYYMMDDTHHMMSSZ (ISO-8601 basic; lexicographic order == chronological order).
_TS_RE = re.compile(rf"{_RUN_PREFIX}-(\d{{8}}T\d{{6}}Z)$")
_TS_FMT = "%Y%m%dT%H%M%SZ"

# The top-level keys the latest cert MUST carry (Phase 75 additive set + the high_rate caveat the
# reader reads verbatim). A missing key fails CLOSED -- never surface a partial / older-schema cert
# (the pre-Phase-75 June-12/13 certs lack per_regime/escalation/n1_caveat and are rejected).
_REQUIRED_KEYS = (
    "git_sha",
    "per_profile",
    "naive",
    "regime_split",
    "per_regime",
    "escalation",
    "n1_caveat",
    "phase_verdict",
    "high_rate_caveat",
)

# The representative middle profile sliced for the headline per-regime stories.
_REPRESENTATIVE_PROFILE = "balanced"
# Unit keys as emitted by the gate's regime_split (the JSON data contract; pinned by a test).
_HIGH_RATE_UNIT = "high_rate"
_EASING_UNIT = "early_cut"  # the post-first-cut binding unit
_EASING_LABEL = "easing"  # human-facing label for early_cut
_UNIT_LABELS = {_HIGH_RATE_UNIT: _HIGH_RATE_UNIT, _EASING_UNIT: _EASING_LABEL}

_HARD_FAIL = "HARD_FAIL"


@dataclass(frozen=True)
class RegimeStory:
    """One rate-regime sub-window's allocation-vs-best-naive benchmark story (read verbatim)."""

    unit_key: str  # "high_rate" | "early_cut" (raw cert key)
    unit_label: str  # "high_rate" | "easing"  (human label)
    window_start: str  # regime_split[unit][0] (ISO date, verbatim)
    window_end: str  # regime_split[unit][1]
    allocation_sharpe: float
    best_naive_sharpe: float
    allocation_sortino: float
    best_naive_sortino: float
    unit_verdict: str  # per_regime[unit][balanced]["verdict"]


@dataclass(frozen=True)
class CertDecision:
    """Latest binding cert verdict + per-regime benchmark stories + honest framing (Phase 87)."""

    # Provenance
    cert_path: str
    cert_timestamp: str  # ISO datetime parsed from the dir-name suffix
    git_sha: str
    staleness_days: int

    # Binding verdict -- sourced verbatim, never hardcoded
    phase_verdict: str
    escalation: str | None
    n1_caveat: bool

    # Full-window representative (BALANCED) metrics. best_naive is the BEST-of-three naive bar (here
    # equity_100), NOT the deposit -- the deposit only wins in the high_rate sub-window.
    alloc_sharpe_full: float
    best_naive_sharpe_full: float
    full_verdict: str

    # Per-regime stories (high_rate first, then easing if present)
    regime_stories: list[RegimeStory]

    # Operator-facing strings -- DERIVED inline from the fields above, never pre-baked
    headline: str
    when_framing: str

    # The gate's verbatim caveat (written to the JSON)
    high_rate_caveat: str


def select_latest_cert_dir(iter_dir: Path = _ITER_DIR) -> Path:
    """Return the most-recent committed ``allocation-gate-73-*`` dir, fail-closed.

    Sorts the matching sub-dirs by their ``YYYYMMDDTHHMMSSZ`` suffix (ISO-8601 basic ->
    lexicographic == chronological) and returns the LAST. Deterministic: the same filesystem state
    yields the same dir. Raises ``CertNotFoundError`` if *iter_dir* is missing or holds no matching
    dir -- never returns a fabricated path.
    """
    if not iter_dir.is_dir():
        msg = f"no cert directory at {iter_dir}; run scripts/run_allocation_gate.py to produce one"
        raise CertNotFoundError(msg)
    matches = sorted(d for d in iter_dir.iterdir() if d.is_dir() and _TS_RE.match(d.name))
    if not matches:
        msg = f"no committed {_RUN_PREFIX}-* cert under {iter_dir}"
        raise CertNotFoundError(msg)
    return matches[-1]


def parse_cert_json(cert_dir: Path) -> dict[str, Any]:
    """Read + validate ``{cert_dir}/summary.json``, fail-closed.

    Raises ``CertNotFoundError`` on a missing file, malformed JSON, a non-object payload, or ANY
    missing key in ``_REQUIRED_KEYS`` (so a partial / older-schema cert never surfaces silently).
    """
    path = cert_dir / "summary.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read cert summary at {path}: {exc}"
        raise CertNotFoundError(msg) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"malformed cert JSON at {path}: {exc}"
        raise CertNotFoundError(msg) from exc
    if not isinstance(data, dict):
        msg = f"cert payload at {path} is not a JSON object"
        raise CertNotFoundError(msg)
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        msg = f"cert at {path} is missing required keys {missing} (partial / older-schema cert)"
        raise CertNotFoundError(msg)
    return data


def _cert_timestamp(cert_dir: Path) -> datetime:
    """Parse the UTC timestamp from the dir-name suffix; fail-closed on a non-matching name."""
    match = _TS_RE.match(cert_dir.name)
    if match is None:
        msg = f"cert dir {cert_dir.name!r} does not carry a {_TS_FMT} timestamp suffix"
        raise CertNotFoundError(msg)
    return datetime.strptime(match.group(1), _TS_FMT).replace(tzinfo=UTC)


def _regime_story(data: Mapping[str, Any], unit_key: str) -> RegimeStory:
    """Build one RegimeStory from the cert's regime_split + per_regime (representative profile)."""
    window = data["regime_split"][unit_key]
    prof = data["per_regime"][unit_key][_REPRESENTATIVE_PROFILE]
    return RegimeStory(
        unit_key=unit_key,
        unit_label=_UNIT_LABELS.get(unit_key, unit_key),
        window_start=str(window[0]),
        window_end=str(window[1]),
        allocation_sharpe=float(prof["sharpe"]),
        best_naive_sharpe=float(prof["best_naive_sharpe"]),
        allocation_sortino=float(prof["sortino"]),
        best_naive_sortino=float(prof["best_naive_sortino"]),
        unit_verdict=str(prof["verdict"]),
    )


def _compose_headline(phase_verdict: str) -> str:
    """Headline DERIVED from phase_verdict -- HARD_FAIL is shown AS HARD_FAIL, never softened.

    The verdict literal comes from the cert; the surrounding sentence is a display wrapper. The
    bar is "its best benchmark (deposit-anchored)" -- precise, because the full-window best-naive is
    equity_100, while the deposit only wins the high_rate sub-window (the escalation is
    deposit-anchored). A PASS cert flips the headline automatically -- it is not a constant.
    """
    if phase_verdict == _HARD_FAIL:
        return (
            "HOLD DEPOSIT-HEAVY: the allocator does not beat its best benchmark (deposit-anchored) "
            f"(verdict: {phase_verdict})"
        )
    return f"Gate result: {phase_verdict}"


def _compose_when_framing(
    *, regime_stories: list[RegimeStory], n1_caveat: bool, escalation: str | None
) -> str:
    """Honest-qualitative "when do risk assets pay" -- NO fabricated rate threshold (constraint 2).

    Every number is a cert field (the high_rate Sharpe pair); the N=1 clause fires off n1_caveat and
    the redesign clause off escalation. No "rates below X%" string is ever produced -- the cert
    computes no key-rate cutoff, so the framing is explicitly qualitative.
    """
    all_hard_fail = all(s.unit_verdict == _HARD_FAIL for s in regime_stories)
    high_rate = next((s for s in regime_stories if s.unit_key == _HIGH_RATE_UNIT), None)
    deposit_won_high_rate = high_rate is not None and high_rate.best_naive_sharpe > 0.0

    if all_hard_fail and deposit_won_high_rate and high_rate is not None:
        parts = [
            "Risk assets have not beaten the deposit in either measured regime. ",
            f"In the high-rate plateau ({high_rate.window_start}..{high_rate.window_end}) the "
            f"deposit's risk-adjusted return was strongly positive (best-naive Sharpe "
            f"{high_rate.best_naive_sharpe:+.2f}) while the allocator was deeply negative "
            f"({high_rate.allocation_sharpe:+.2f}). ",
            "In the single observed easing cycle all sleeves were negative -- the allocator still "
            "trailed its best benchmark. ",
        ]
        if n1_caveat:
            parts.append("This easing read is N=1: suggestive, not statistically robust. ")
        if escalation == "deposit_anchor_vs_redesign":
            parts.append(
                "The recorded escalation is deposit-anchor-vs-redesign: anchor on the low-vol "
                "deposit for now; a redesign is the documented next step when conditions change. "
            )
        parts.append(
            "No rate threshold is available from the measurement -- the cert computes no key-rate "
            "cutoff; this is a qualitative regime read, not a 'rates below X%' rule."
        )
        return "".join(parts)
    return (
        "Per-regime outcomes (each verdict and Sharpe sourced from the cert above) determine when "
        "risk assets pay; no numeric rate threshold is computed by the measurement."
    )


def load_latest_cert(iter_dir: Path = _ITER_DIR, *, today: date | None = None) -> CertDecision:
    """Select -> parse -> hydrate the latest committed cert into a frozen ``CertDecision``.

    ``today`` is injectable (defaults to the real clock's date) so ``staleness_days`` is
    deterministic in tests. Raises ``CertNotFoundError`` on any failure (missing/empty dir,
    malformed JSON, missing key) so the surface layers can render a fail-closed empty state.
    """
    cert_dir = select_latest_cert_dir(iter_dir)
    data = parse_cert_json(cert_dir)
    ts = _cert_timestamp(cert_dir)
    if today is None:
        from finalayze.core.clock import RealClock  # noqa: PLC0415 -- avoid load-time coupling

        today = RealClock().now().date()
    staleness_days = (today - ts.date()).days

    stories = [_regime_story(data, _HIGH_RATE_UNIT)]
    if _EASING_UNIT in data["per_regime"]:
        stories.append(_regime_story(data, _EASING_UNIT))

    balanced = data["per_profile"][_REPRESENTATIVE_PROFILE]
    phase_verdict = str(data["phase_verdict"])
    escalation = data["escalation"]
    n1_caveat = bool(data["n1_caveat"])

    return CertDecision(
        cert_path=str(cert_dir / "summary.json"),
        cert_timestamp=ts.isoformat(),
        git_sha=str(data["git_sha"]),
        staleness_days=staleness_days,
        phase_verdict=phase_verdict,
        escalation=None if escalation is None else str(escalation),
        n1_caveat=n1_caveat,
        alloc_sharpe_full=float(balanced["sharpe"]),
        best_naive_sharpe_full=float(balanced["best_naive_sharpe"]),
        full_verdict=str(balanced["verdict"]),
        regime_stories=stories,
        headline=_compose_headline(phase_verdict),
        when_framing=_compose_when_framing(
            regime_stories=stories,
            n1_caveat=n1_caveat,
            escalation=None if escalation is None else str(escalation),
        ),
        high_rate_caveat=str(data["high_rate_caveat"]),
    )

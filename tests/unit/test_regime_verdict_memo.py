"""Memo-presence smoke test for the REGIME-05 decision memo (Phase 75 Plan 04).

A thin file-exists + content-grep check that the standalone strategic memo
``docs/research/regime_verdict_decision.md`` exists and structurally records the
recorded decision derived from the real Plan-03 cert:

  (a) both regime verdicts (mentions ``high_rate`` AND ``easing``),
  (b) the deposit-anchor lean (contains ``deposit``),
  (c) the documented-DEFERRED redesign branch (OFZ duration / fixed-coupon vs
      OFZ-PK floater),
  (d) the N=1 caveat (single observed easing cycle), and
  (e) the cert source (``results/iterations``).

The assertions are STRUCTURAL (presence), NOT prose-quality: the prose is
human-verified (Task 3 checkpoint) and the numbers are validated against the cert
(D-03a). This test only pins that the memo exists and references the required
decision elements.
"""

from __future__ import annotations

from pathlib import Path

# Repo root: file -> unit -> tests -> repo root (the established tests/unit/ pattern).
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_MEMO_PATH: Path = _REPO_ROOT / "docs" / "research" / "regime_verdict_decision.md"

# Required substrings (named constants — no magic literals). Each maps to one of the
# memo's load-bearing decision elements (D-02 / D-03 / D-04 / D-03a).
_REGIME_HIGH_RATE: str = "high_rate"  # (a) high-rate regime verdict referenced
_REGIME_EASING: str = "easing"  # (a) easing regime verdict referenced
_DEPOSIT_ANCHOR: str = "deposit"  # (b) the deposit-anchor lean (D-03)
_REDESIGN_OFZ: str = "ofz"  # (c) the deferred redesign branch (OFZ duration vs floater)
_REDESIGN_FLOATER: str = "floater"  # (c) OFZ-PK floater (the redesign contrast)
_N1_CAVEAT_SINGLE: str = "single"  # (d) single observed easing cycle (N=1 caveat, D-04)
_N1_CAVEAT_TOKEN: str = "n=1"  # (d) the explicit N=1 marker
_CERT_SOURCE: str = "results/iterations"  # (e) the cert source (D-03a anti-hollow link)


def test_memo_records_decision() -> None:
    """The decision memo exists and structurally records the derived decision.

    Reads ``docs/research/regime_verdict_decision.md`` and asserts each required
    substring is present (case-insensitive). FAILS (RED) until the memo is written
    in Task 2.
    """
    assert _MEMO_PATH.exists(), f"decision memo not found: {_MEMO_PATH}"
    text: str = _MEMO_PATH.read_text(encoding="utf-8")
    lowered: str = text.lower()

    # (a) both regime verdicts referenced.
    assert _REGIME_HIGH_RATE in lowered, "memo must reference the high_rate regime verdict"
    assert _REGIME_EASING in lowered, "memo must reference the easing regime verdict"
    # (b) the deposit-anchor lean.
    assert _DEPOSIT_ANCHOR in lowered, "memo must record the deposit-anchor lean (D-03)"
    # (c) the documented-DEFERRED redesign branch (OFZ duration vs OFZ-PK floater).
    assert _REDESIGN_OFZ in lowered, "memo must document the OFZ-duration redesign branch"
    assert _REDESIGN_FLOATER in lowered, "memo must contrast the OFZ-PK floater redesign branch"
    # (d) the N=1 caveat (single observed easing cycle).
    assert _N1_CAVEAT_TOKEN in lowered or _N1_CAVEAT_SINGLE in lowered, (
        "memo must record the N=1 / single-cycle easing caveat (D-04)"
    )
    # (e) the cert source (anti-hollow link, D-03a).
    assert _CERT_SOURCE in lowered, "memo must cite the results/iterations cert source (D-03a)"

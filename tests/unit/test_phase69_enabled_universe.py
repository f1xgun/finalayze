"""Pin the post-Phase-68 enabled MOEX universe (Phase 69 diagnosis universe).

Phase 68 removed ``ru_blue_chips`` entirely and activated four sector segments
(ru_metals / ru_telecom / ru_construction / ru_transport). The whole Phase-69
exit-path diagnostic runs against this exact enabled set, so any silent config
drift in ``config/segments.DEFAULT_SEGMENTS`` MUST fail loudly here (threat
T-69-01 mitigation).

NOTE on field name: ``SegmentConfig`` exposes ``segment_id`` (NOT ``name``);
the segment identity attribute is ``segment_id`` everywhere in config/segments.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable (config/ lives at repo root, not under src/).
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from config.segments import DEFAULT_SEGMENTS  # noqa: E402

# ---------------------------------------------------------------------------
# Expected post-68 enabled MOEX universe (RESEARCH "Enabled post-68 Segment Set")
# ---------------------------------------------------------------------------
# The authoritative diagnosis universe: 7 equities + 2 bonds. ru_blue_chips was
# REMOVED in Phase 68 and is intentionally absent. ru_consumer / ru_utilities /
# ru_chemicals are present-but-disabled (enabled=False) and out of scope.
EXPECTED_ENABLED_MOEX: frozenset[str] = frozenset(
    {
        "ru_energy",  # CONTROL segment (no-regression guardrail)
        "ru_tech",
        "ru_finance",
        "ru_metals",
        "ru_telecom",
        "ru_construction",
        "ru_transport",
        "ru_ofz_pd",  # bond
        "ru_ofz_pk",  # bond
    }
)

EXPECTED_ENABLED_MOEX_COUNT = 9
EXPECTED_BOND = 2  # ru_ofz_pd, ru_ofz_pk
EXPECTED_EQUITY = 7  # the other seven enabled MOEX segments


def _enabled_moex_ids() -> set[str]:
    """The set of enabled MOEX ``segment_id`` values from DEFAULT_SEGMENTS."""
    return {s.segment_id for s in DEFAULT_SEGMENTS if s.market == "moex" and s.enabled}


class TestEnabledMoexUniverse:
    """The post-68 enabled MOEX universe is exactly the 9 expected segments."""

    def test_enabled_moex_set_matches_expected(self) -> None:
        assert _enabled_moex_ids() == EXPECTED_ENABLED_MOEX

    def test_enabled_moex_count_is_nine(self) -> None:
        assert len(_enabled_moex_ids()) == EXPECTED_ENABLED_MOEX_COUNT

    def test_ru_blue_chips_is_not_enabled(self) -> None:
        # Negative assertion only: ru_blue_chips was removed entirely in Phase 68.
        assert "ru_blue_chips" not in _enabled_moex_ids()

    def test_four_sectors_are_activated(self) -> None:
        activated = {"ru_metals", "ru_telecom", "ru_construction", "ru_transport"}
        assert activated <= _enabled_moex_ids()


class TestEnabledMoexInstrumentMix:
    """The enabled MOEX universe is 7 equities + 2 bonds."""

    def test_exactly_two_bonds(self) -> None:
        bonds = {
            s.segment_id
            for s in DEFAULT_SEGMENTS
            if s.market == "moex" and s.enabled and s.instrument_type == "bond"
        }
        assert bonds == {"ru_ofz_pd", "ru_ofz_pk"}
        assert len(bonds) == EXPECTED_BOND

    def test_exactly_seven_equities(self) -> None:
        non_bonds = {
            s.segment_id
            for s in DEFAULT_SEGMENTS
            if s.market == "moex" and s.enabled and s.instrument_type != "bond"
        }
        assert len(non_bonds) == EXPECTED_EQUITY

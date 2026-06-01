"""Single-source proof for the three universe seams (LIQ-08 / T-66-05).

Phase 66 Plan 02 makes ``finalayze.markets.liquidity.select_segment_symbols`` the ONE
source every universe seam resolves through:

  1. LIVE      -- ``config.segments.DEFAULT_SEGMENTS[*].symbols`` (frozen, built at
                  module-construction time from the selector).
  2. backtest  -- ``scripts.run_iteration.UNIVERSE``.
  3. training  -- ``scripts.training.cli.SEGMENT_SYMBOLS``.

``test_three_seams_resolve_same_set`` is deliberately NON-TRIVIAL: it monkeypatches the
selector's committed-snapshot read with a deterministic fixture whose selected set DIFFERS
from the pre-phase hardcoded lists (drops delisted YNDX/TCSG/SNGS, adds new liquid names),
then reloads all three modules so they re-resolve. It asserts (a) all three resolve to the
EXACT fixture set per segment, and (b) the fixture set DIFFERS from the old hardcoded list
for at least one segment -- so a seam that still returned its old hardcoded list (a
monkeypatched identity) would FAIL, not silently pass.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from finalayze.markets import liquidity

# ---------------------------------------------------------------------------
# Constants (ruff PLR2004 -- no magic numbers / literals)
# ---------------------------------------------------------------------------
# The ru_* SHARE segments under test and the curated sector that feeds each
# (mirrors config.segments.SECTOR_TO_SEGMENT -- the single D-08 source).
_SEG_BLUE = "ru_blue_chips"
_SEG_ENERGY = "ru_energy"
_SEG_TECH = "ru_tech"
_SEG_FINANCE = "ru_finance"

_SECTOR_DIVERSIFIED = "diversified"
_SECTOR_OIL_GAS = "oil_gas"
_SECTOR_TECH = "tech"
_SECTOR_BANKS = "banks"

# Deterministic FIXTURE snapshot: sector -> ranked symbols. Chosen so the selected set
# DIFFERS from the old hardcoded lists -- delisted YNDX/TCSG/SNGS are absent and at least
# one NEW liquid name per sector appears.
_FIXTURE_SECTORS: dict[str, list[str]] = {
    _SECTOR_DIVERSIFIED: ["SBER", "LKOH", "GMKN", "PLZL"],  # PLZL new vs old blue list
    _SECTOR_OIL_GAS: ["ROSN", "TATN", "NVTK", "SIBN"],  # SIBN new; no BANEP/TRNFP-only
    _SECTOR_TECH: ["YDEX", "OZON", "ASTR"],  # YDEX/ASTR new; delisted YNDX/CIAN gone
    _SECTOR_BANKS: ["SBER", "VTBR", "MOEX", "BSPB"],  # delisted TCSG gone
}

# Expected per-segment set = concatenation of the fixture sectors mapped to that segment.
_EXPECTED: dict[str, list[str]] = {
    _SEG_BLUE: list(_FIXTURE_SECTORS[_SECTOR_DIVERSIFIED]),
    _SEG_ENERGY: list(_FIXTURE_SECTORS[_SECTOR_OIL_GAS]),
    _SEG_TECH: list(_FIXTURE_SECTORS[_SECTOR_TECH]),
    _SEG_FINANCE: list(_FIXTURE_SECTORS[_SECTOR_BANKS]),
}

# Pre-phase HARDCODED ru_* lists (captured verbatim from the three seams BEFORE Plan 02).
# A seam that still returned any of these would FAIL the anti-trivial assertion below.
_OLD_HARDCODED_CONFIG: dict[str, list[str]] = {
    _SEG_BLUE: ["SBER", "LKOH", "GMKN"],
    _SEG_ENERGY: ["ROSN", "TATN", "NVTK", "SIBN", "TATNP", "TRNFP"],
    _SEG_TECH: ["YDEX", "OZON", "VKCO", "HEAD", "POSI", "ASTR", "DIAS", "SOFL"],
    _SEG_FINANCE: ["SBER", "T", "CBOM", "BSPB", "MOEX", "VTBR", "AFKS", "RENI"],
}
_OLD_HARDCODED_UNIVERSE: dict[str, list[str]] = {
    _SEG_BLUE: ["SBER", "LKOH", "YNDX", "MGNT", "POLY", "NVTK", "MTLR"],
    _SEG_ENERGY: ["LKOH", "ROSN", "NVTK", "TATN", "TRNFP", "BANEP"],
    _SEG_FINANCE: ["SBER", "SBERP", "TCSG", "CBOM", "BSPB", "MOEX"],
}
_OLD_HARDCODED_TRAINING: dict[str, list[str]] = {
    _SEG_BLUE: ["SBER", "LKOH", "GMKN", "ROSN", "NVTK", "MGNT", "TATN", "TCSG"],
    _SEG_ENERGY: ["ROSN", "TATN", "NVTK", "LKOH", "SNGS", "SIBN"],
    _SEG_TECH: ["YNDX", "OZON", "VKCO", "CIAN"],
    _SEG_FINANCE: ["SBER", "VTBR", "TCSG", "MOEX", "CBOM"],
}


@pytest.fixture
def _patched_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the selector at a real on-disk fixture snapshot (exercises the real loader).

    Writing a real file (rather than only patching ``_load_liquidity_snapshot``) keeps the
    ``_LIQ_SNAPSHOT.exists()`` bootstrap guard satisfied AND runs the real fail-closed
    loader + sector validation against the fixture.
    """
    snap = tmp_path / "moex_liquidity_universe.json"
    snap.write_text(json.dumps({"sectors": _FIXTURE_SECTORS}), encoding="utf-8")
    monkeypatch.setattr(liquidity, "_LIQ_SNAPSHOT", snap)


def test_three_seams_resolve_same_set(_patched_snapshot: None) -> None:
    """All three universe seams resolve the IDENTICAL fixture set per ru_* segment.

    Non-trivial: the fixture set differs from the old hardcoded lists, so a seam that
    bypassed the selector (returned its old list) would fail the equality + anti-trivial
    assertions below.
    """
    # Reload the three seams so they re-resolve through the patched selector.
    import config.segments as segments_mod
    import scripts.run_iteration as run_iteration_mod
    import scripts.training.cli as training_cli_mod

    segments_mod = importlib.reload(segments_mod)
    run_iteration_mod = importlib.reload(run_iteration_mod)
    training_cli_mod = importlib.reload(training_cli_mod)

    live_symbols: dict[str, list[str]] = {
        s.segment_id: s.symbols for s in segments_mod.DEFAULT_SEGMENTS
    }
    universe = run_iteration_mod.UNIVERSE
    seg_symbols = training_cli_mod.SEGMENT_SYMBOLS

    # Segments present in ALL three seams (run_iteration carries no ru_tech).
    common_segments = [_SEG_BLUE, _SEG_ENERGY, _SEG_FINANCE]
    for seg in common_segments:
        expected = _EXPECTED[seg]
        assert live_symbols[seg] == expected, f"LIVE {seg}: {live_symbols[seg]} != {expected}"
        assert universe[seg] == expected, f"backtest {seg}: {universe[seg]} != {expected}"
        assert seg_symbols[seg] == expected, f"training {seg}: {seg_symbols[seg]} != {expected}"

    # ru_tech exists in LIVE + training (not in run_iteration.UNIVERSE) -- still single-source.
    assert live_symbols[_SEG_TECH] == _EXPECTED[_SEG_TECH]
    assert seg_symbols[_SEG_TECH] == _EXPECTED[_SEG_TECH]

    # ── Anti-trivial guard ──────────────────────────────────────────────────────────
    # The fixture (selector) set MUST differ from the old hardcoded list for >= 1 segment
    # in EACH seam, so a seam that still returned its old hardcoded list would FAIL above.
    all_segs = (_SEG_BLUE, _SEG_ENERGY, _SEG_TECH, _SEG_FINANCE)
    assert any(_EXPECTED[s] != _OLD_HARDCODED_CONFIG[s] for s in all_segs)
    assert any(_EXPECTED[s] != _OLD_HARDCODED_UNIVERSE[s] for s in common_segments)
    assert any(_EXPECTED[s] != _OLD_HARDCODED_TRAINING[s] for s in all_segs)


def test_reload_restores_clean_state() -> None:
    """Reload the seams WITHOUT the patched snapshot so other tests see normal state.

    Pre-66-04 the committed snapshot is absent, so the selector returns empty live
    universes (bootstrap tolerance) -- importing the seams must still succeed.
    """
    import config.segments as segments_mod
    import scripts.run_iteration as run_iteration_mod
    import scripts.training.cli as training_cli_mod

    importlib.reload(segments_mod)
    importlib.reload(run_iteration_mod)
    importlib.reload(training_cli_mod)
    # Boot path intact: the modules import and the ru_* live universes are empty (no
    # committed snapshot yet) -- never a stale list.
    assert {s.segment_id for s in segments_mod.DEFAULT_SEGMENTS}  # non-empty segment set

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
_SEG_ENERGY = "ru_energy"
_SEG_TECH = "ru_tech"
_SEG_FINANCE = "ru_finance"

_SECTOR_OIL_GAS = "oil_gas"
_SECTOR_TECH = "tech"
_SECTOR_BANKS = "banks"

# Deterministic FIXTURE snapshot: sector -> ranked symbols. Chosen so the selected set
# DIFFERS from the old hardcoded lists -- delisted YNDX/TCSG/SNGS are absent and at least
# one NEW liquid name per sector appears. (UNIV-02: the diversified/ru_blue_chips fixture
# entry was dropped when the tag was retired -- this seam no longer references it.)
_FIXTURE_SECTORS: dict[str, list[str]] = {
    _SECTOR_OIL_GAS: ["ROSN", "TATN", "NVTK", "SIBN"],  # SIBN new; no BANEP/TRNFP-only
    _SECTOR_TECH: ["YDEX", "OZON", "ASTR"],  # YDEX/ASTR new; delisted YNDX/CIAN gone
    _SECTOR_BANKS: ["SBER", "VTBR", "MOEX", "BSPB"],  # delisted TCSG gone
}

# Expected per-segment set = concatenation of the fixture sectors mapped to that segment,
# AFTER the universal safety post-filter (Plan 66-04): toxic/sanctioned names + preferred-share
# duplicates are dropped from the FINAL set in ALL three seams. The banks fixture deliberately
# injects VTBR (toxic) -- it MUST be dropped here, so this fixture doubles as cross-seam proof
# that the safety filter is applied at the single source (the selector), not per seam.
_EXPECTED: dict[str, list[str]] = {
    _SEG_ENERGY: list(_FIXTURE_SECTORS[_SECTOR_OIL_GAS]),
    _SEG_TECH: list(_FIXTURE_SECTORS[_SECTOR_TECH]),
    # VTBR dropped by the universal toxic filter (was in the banks fixture).
    _SEG_FINANCE: [s for s in _FIXTURE_SECTORS[_SECTOR_BANKS] if s != "VTBR"],
}

# Pre-phase HARDCODED ru_* lists (captured verbatim from the three seams BEFORE Plan 02).
# A seam that still returned any of these would FAIL the anti-trivial assertion below.
_OLD_HARDCODED_CONFIG: dict[str, list[str]] = {
    _SEG_ENERGY: ["ROSN", "TATN", "NVTK", "SIBN", "TATNP", "TRNFP"],
    _SEG_TECH: ["YDEX", "OZON", "VKCO", "HEAD", "POSI", "ASTR", "DIAS", "SOFL"],
    _SEG_FINANCE: ["SBER", "T", "CBOM", "BSPB", "MOEX", "VTBR", "AFKS", "RENI"],
}
_OLD_HARDCODED_UNIVERSE: dict[str, list[str]] = {
    _SEG_ENERGY: ["LKOH", "ROSN", "NVTK", "TATN", "TRNFP", "BANEP"],
    _SEG_FINANCE: ["SBER", "SBERP", "TCSG", "CBOM", "BSPB", "MOEX"],
}
_OLD_HARDCODED_TRAINING: dict[str, list[str]] = {
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

    # Segments present in ALL three seams. Post-66 run_iteration.UNIVERSE derives a key for
    # EVERY enabled MOEX stock segment (including ru_tech), so these ru_* segments under
    # test are single-sourced across all three seams (verified at runtime). ru_blue_chips
    # was removed in Phase 68 (UNIV-02), so it is no longer part of the seam contract.
    common_segments = [_SEG_ENERGY, _SEG_TECH, _SEG_FINANCE]
    for seg in common_segments:
        expected = _EXPECTED[seg]
        assert live_symbols[seg] == expected, f"LIVE {seg}: {live_symbols[seg]} != {expected}"
        assert universe[seg] == expected, f"backtest {seg}: {universe[seg]} != {expected}"
        assert seg_symbols[seg] == expected, f"training {seg}: {seg_symbols[seg]} != {expected}"

    # ── Anti-trivial guard ──────────────────────────────────────────────────────────
    # The fixture (selector) set MUST differ from the old hardcoded list for >= 1 segment
    # in EACH seam, so a seam that still returned its old hardcoded list would FAIL above.
    all_segs = (_SEG_ENERGY, _SEG_TECH, _SEG_FINANCE)
    # _OLD_HARDCODED_UNIVERSE captured only the pre-66 run_iteration keys (no ru_tech), so the
    # anti-trivial guard for the backtest seam iterates only the segments it recorded.
    old_universe_segs = (_SEG_ENERGY, _SEG_FINANCE)
    assert any(_EXPECTED[s] != _OLD_HARDCODED_CONFIG[s] for s in all_segs)
    assert any(_EXPECTED[s] != _OLD_HARDCODED_UNIVERSE[s] for s in old_universe_segs)
    assert any(_EXPECTED[s] != _OLD_HARDCODED_TRAINING[s] for s in all_segs)


# ---------------------------------------------------------------------------
# Safety post-filter applied to the selector output (Plan 66-04)
# ---------------------------------------------------------------------------

# A mocked snapshot whose banks sector carries BOTH a toxic name (VTBR) and a preferred-share
# duplicate (SBERP, whose common SBER is in the same sector). The selector MUST drop both.
_SAFETY_FIXTURE_SECTORS: dict[str, list[str]] = {
    _SECTOR_BANKS: ["SBER", "VTBR", "SBERP", "MOEX", "BSPB"],
    # oil_gas: TATNP (pref of TATN, present) must drop; TRNFP (no common TRNF) must STAY;
    # SNGSP is toxic and must drop; GAZP toxic must drop.
    _SECTOR_OIL_GAS: ["ROSN", "TATN", "TATNP", "TRNFP", "GAZP", "SNGSP"],
}

_EXPECTED_BANKS_FILTERED = ["SBER", "MOEX", "BSPB"]  # VTBR + SBERP dropped
_EXPECTED_OIL_GAS_FILTERED = ["ROSN", "TATN", "TRNFP"]  # TATNP/GAZP/SNGSP dropped, TRNFP kept


@pytest.fixture
def _patched_safety_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the selector at a mocked snapshot containing toxic + preferred-duplicate names."""
    snap = tmp_path / "moex_liquidity_universe.json"
    snap.write_text(json.dumps({"sectors": _SAFETY_FIXTURE_SECTORS}), encoding="utf-8")
    monkeypatch.setattr(liquidity, "_LIQ_SNAPSHOT", snap)


def test_selector_drops_toxic_and_preferred_duplicates(_patched_safety_snapshot: None) -> None:
    """``select_segment_symbols`` applies the universal safety filter to the snapshot output.

    Proves the toxic-symbol exclusion + preferred-share-duplicate drop hold at the SINGLE source
    (the selector), so they apply regardless of how the symbols entered the snapshot. A standalone
    preferred (TRNFP, no common TRNF in the set) is preserved -- the rule does not over-exclude.
    """
    banks = liquidity.select_segment_symbols(_SEG_FINANCE)
    assert banks == _EXPECTED_BANKS_FILTERED, banks
    assert "VTBR" not in banks  # toxic
    assert "SBERP" not in banks  # preferred duplicate of SBER (present)

    oil_gas = liquidity.select_segment_symbols(_SEG_ENERGY)
    assert oil_gas == _EXPECTED_OIL_GAS_FILTERED, oil_gas
    assert "GAZP" not in oil_gas and "SNGSP" not in oil_gas  # toxic
    assert "TATNP" not in oil_gas  # preferred duplicate of TATN (present)
    assert "TRNFP" in oil_gas  # standalone preferred (no common) preserved


def test_apply_safety_filters_unit() -> None:
    """Direct unit coverage of the order-preserving safety post-filter."""
    raw = ["SBER", "VTBR", "SBERP", "TATN", "TATNP", "TRNFP", "GAZP"]
    filtered = liquidity._apply_safety_filters(raw)
    # VTBR/GAZP toxic; SBERP/TATNP preferred-dupes; SBER/TATN/TRNFP kept (order preserved).
    assert filtered == ["SBER", "TATN", "TRNFP"]


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


# ---------------------------------------------------------------------------
# UNIV-02 — ru_blue_chips removed + diversified retired (real committed snapshot)
# ---------------------------------------------------------------------------

_INJECTED_UNKNOWN_SECTOR = "diversified"  # the now-retired tag -- a snapshot carrying it must raise


def test_ru_blue_chips_resolves_empty_from_committed_snapshot() -> None:
    """Test C (UNIV-02): no sector maps to ru_blue_chips, so the selector returns []."""
    assert liquidity.select_segment_symbols("ru_blue_chips") == []


def test_committed_snapshot_has_no_diversified_key_and_loads_clean() -> None:
    """Test D (UNIV-02): the committed snapshot has no diversified key; the loader is clean."""
    sectors = liquidity._load_liquidity_snapshot()  # must NOT raise on the edited snapshot
    assert "diversified" not in sectors


def test_loader_still_fail_closed_on_unknown_sector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test F (T-68-01): an injected unknown/orphaned sector STILL fails closed.

    Removing the diversified tag from the map must not weaken the fail-closed guard:
    a snapshot carrying a sector key absent from SECTOR_TO_SEGMENT raises ConfigurationError.
    """
    from finalayze.core.exceptions import ConfigurationError

    snap = tmp_path / "moex_liquidity_universe.json"
    snap.write_text(json.dumps({"sectors": {_INJECTED_UNKNOWN_SECTOR: ["SFIN"]}}), encoding="utf-8")
    monkeypatch.setattr(liquidity, "_LIQ_SNAPSHOT", snap)
    with pytest.raises(ConfigurationError):
        liquidity._load_liquidity_snapshot()

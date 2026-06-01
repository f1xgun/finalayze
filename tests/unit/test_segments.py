"""Tests for config.segments — segment definitions and SegmentConfig."""

from __future__ import annotations

from config.segments import DEFAULT_SEGMENTS, SECTOR_TO_SEGMENT, SegmentConfig

from finalayze.markets.liquidity import select_segment_symbols

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEGMENT_MAP = {s.segment_id: s for s in DEFAULT_SEGMENTS}


def _get(segment_id: str) -> SegmentConfig:
    return _SEGMENT_MAP[segment_id]


# ---------------------------------------------------------------------------
# SegmentConfig defaults
# ---------------------------------------------------------------------------


class TestSegmentConfigDefaults:
    def test_instrument_type_defaults_to_stock(self) -> None:
        cfg = SegmentConfig(
            segment_id="test",
            market="us",
            broker="alpaca",
            currency="USD",
        )
        assert cfg.instrument_type == "stock"


# ---------------------------------------------------------------------------
# Bond segments: ru_ofz_pd
# ---------------------------------------------------------------------------


class TestRuOfzPd:
    def test_instrument_type_is_bond(self) -> None:
        seg = _get("ru_ofz_pd")
        assert seg.instrument_type == "bond"

    def test_has_7_symbols(self) -> None:
        seg = _get("ru_ofz_pd")
        assert len(seg.symbols) == 7

    def test_26238_excluded(self) -> None:
        seg = _get("ru_ofz_pd")
        assert "SU26238RMFS4" not in seg.symbols

    def test_active_strategies(self) -> None:
        seg = _get("ru_ofz_pd")
        assert seg.active_strategies == ["bond_duration_rotation"]

    def test_market_and_broker(self) -> None:
        seg = _get("ru_ofz_pd")
        assert seg.market == "moex"
        assert seg.broker == "tinkoff"
        assert seg.currency == "RUB"


# ---------------------------------------------------------------------------
# Bond segments: ru_ofz_pk
# ---------------------------------------------------------------------------


class TestRuOfzPk:
    def test_instrument_type_is_bond(self) -> None:
        seg = _get("ru_ofz_pk")
        assert seg.instrument_type == "bond"

    def test_has_4_symbols(self) -> None:
        seg = _get("ru_ofz_pk")
        assert len(seg.symbols) == 4

    def test_active_strategies(self) -> None:
        seg = _get("ru_ofz_pk")
        assert seg.active_strategies == ["bond_carry"]

    def test_max_allocation(self) -> None:
        seg = _get("ru_ofz_pk")
        assert seg.max_allocation_pct == 0.50


# ---------------------------------------------------------------------------
# Existing stock segments still have instrument_type="stock"
# ---------------------------------------------------------------------------

_STOCK_SEGMENT_IDS = [
    "us_tech",
    "us_healthcare",
    "us_finance",
    "us_broad",
    "ru_energy",
    "ru_tech",
    "ru_finance",
]

# D-05 control: ru_energy resolves the oil_gas sector and is UNAFFECTED by removing
# ru_blue_chips / retiring the diversified tag. Captured verbatim from the committed
# snapshot's oil_gas sector AFTER the universal safety filter (no toxic, no preferred
# duplicate -- TRNFP stays because its common TRNF is not present). Re-asserted byte-
# identical pre/post the Phase-68 edits to prove the changes are universe-local.
_RU_ENERGY_CONTROL = ["LKOH", "ROSN", "NVTK", "TATN", "TRNFP", "SIBN", "RNFT"]


class TestExistingStockSegments:
    def test_all_stock_segments_have_stock_instrument_type(self) -> None:
        for sid in _STOCK_SEGMENT_IDS:
            seg = _get(sid)
            assert seg.instrument_type == "stock", f"{sid} should be stock"


# ---------------------------------------------------------------------------
# ru_finance segment
# ---------------------------------------------------------------------------


class TestRuFinance:
    def test_sberp_not_in_ru_finance(self) -> None:
        seg = _get("ru_finance")
        assert "SBERP" not in seg.symbols


# ---------------------------------------------------------------------------
# S1.2 — US segments frozen, RU segments enabled
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# UNIV-02 — ru_blue_chips removed + diversified retired
# ---------------------------------------------------------------------------


class TestRuBlueChipsRemoved:
    """ru_blue_chips is gone from the segment set and the sector map (D-02)."""

    def test_not_in_default_segments(self) -> None:
        assert "ru_blue_chips" not in {s.segment_id for s in DEFAULT_SEGMENTS}

    def test_diversified_tag_retired_from_map(self) -> None:
        assert "diversified" not in SECTOR_TO_SEGMENT
        assert "ru_blue_chips" not in SECTOR_TO_SEGMENT.values()


# ---------------------------------------------------------------------------
# UNIV-04 / D-05 — ru_energy byte-identical control pin
# ---------------------------------------------------------------------------


class TestRuEnergyControlPin:
    """ru_energy selected set is unchanged by the Phase-68 edits (the control)."""

    def test_ru_energy_selected_set_byte_identical(self) -> None:
        assert select_segment_symbols("ru_energy") == _RU_ENERGY_CONTROL


class TestSegmentEnabledFlag:
    def test_enabled_field_defaults_to_true(self) -> None:
        cfg = SegmentConfig(
            segment_id="test",
            market="moex",
            broker="tinkoff",
            currency="RUB",
        )
        assert cfg.enabled is True

    def test_all_us_segments_disabled(self) -> None:
        us = [s for s in DEFAULT_SEGMENTS if s.market == "us"]
        assert us, "expected at least one US segment to be present (kept for history)"
        for seg in us:
            assert seg.enabled is False, f"US segment {seg.segment_id} must be disabled"

    def test_all_moex_segments_enabled(self) -> None:
        moex = [s for s in DEFAULT_SEGMENTS if s.market == "moex"]
        for seg in moex:
            assert seg.enabled is True, f"MOEX segment {seg.segment_id} must be enabled"

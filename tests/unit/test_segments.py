"""Tests for config.segments — segment definitions and SegmentConfig."""

from __future__ import annotations

from config.segments import DEFAULT_SEGMENTS, SegmentConfig

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
    "ru_blue_chips",
    "ru_energy",
    "ru_tech",
    "ru_finance",
]


class TestExistingStockSegments:
    def test_all_stock_segments_have_stock_instrument_type(self) -> None:
        for sid in _STOCK_SEGMENT_IDS:
            seg = _get(sid)
            assert seg.instrument_type == "stock", f"{sid} should be stock"

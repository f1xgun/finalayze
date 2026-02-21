"""Unit tests for configuration modules."""

from __future__ import annotations

from config.modes import WorkMode
from config.segments import DEFAULT_SEGMENTS, SegmentConfig

EXPECTED_MODE_COUNT = 4
EXPECTED_SEGMENT_COUNT = 8
EXPECTED_US_SEGMENT_COUNT = 4
EXPECTED_MOEX_SEGMENT_COUNT = 4


class TestWorkMode:
    def test_debug_mode(self) -> None:
        assert WorkMode.DEBUG.value == "debug"

    def test_sandbox_mode(self) -> None:
        assert WorkMode.SANDBOX.value == "sandbox"

    def test_test_mode(self) -> None:
        assert WorkMode.TEST.value == "test"

    def test_real_mode(self) -> None:
        assert WorkMode.REAL.value == "real"

    def test_mode_count(self) -> None:
        assert len(WorkMode) == EXPECTED_MODE_COUNT


class TestSegmentConfig:
    def test_default_segments_count(self) -> None:
        assert len(DEFAULT_SEGMENTS) == EXPECTED_SEGMENT_COUNT

    def test_us_segments(self) -> None:
        us = [s for s in DEFAULT_SEGMENTS if s.market == "us"]
        assert len(us) == EXPECTED_US_SEGMENT_COUNT

    def test_moex_segments(self) -> None:
        ru = [s for s in DEFAULT_SEGMENTS if s.market == "moex"]
        assert len(ru) == EXPECTED_MOEX_SEGMENT_COUNT

    def test_segment_is_frozen(self) -> None:
        seg = DEFAULT_SEGMENTS[0]
        assert isinstance(seg, SegmentConfig)

    def test_us_tech_has_strategies(self) -> None:
        us_tech = next(s for s in DEFAULT_SEGMENTS if s.segment_id == "us_tech")
        assert len(us_tech.active_strategies) > 0

    def test_all_segments_have_currency(self) -> None:
        for seg in DEFAULT_SEGMENTS:
            assert seg.currency in ("USD", "RUB")

    def test_all_segments_have_broker(self) -> None:
        for seg in DEFAULT_SEGMENTS:
            assert seg.broker in ("alpaca", "tinkoff")

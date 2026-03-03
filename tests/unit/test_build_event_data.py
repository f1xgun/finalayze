"""Unit tests for scripts/build_event_data.py."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add scripts/ to sys.path so we can import build_event_data directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import build_event_data  # noqa: E402

# ── Constants (no magic numbers per ruff PLR2004) ────────────────────────────

EXPECTED_CBR_DECISIONS_COUNT = 16
REQUIRED_CBR_KEYS = {"date", "rate_decision", "expected_rate", "surprise_bps"}
EXPECTED_DATE_FORMAT = "%Y-%m-%d"

# Total unique US symbols across all us_* segments (deduplicated)
EXPECTED_US_SYMBOL_COUNT = 84

RU_SEGMENT_PREFIXES = ("ru_",)


# ── 1. CBR_DECISIONS data integrity ──────────────────────────────────────────


class TestCbrDecisionsData:
    def test_has_correct_entry_count(self) -> None:
        assert len(build_event_data.CBR_DECISIONS) == EXPECTED_CBR_DECISIONS_COUNT

    def test_all_entries_have_required_keys(self) -> None:
        for entry in build_event_data.CBR_DECISIONS:
            assert entry.keys() >= REQUIRED_CBR_KEYS, f"Entry missing keys: {entry}"

    def test_dates_are_valid_iso_format(self) -> None:
        for entry in build_event_data.CBR_DECISIONS:
            date_str = str(entry["date"])
            # Raises ValueError if format is wrong.
            datetime.strptime(date_str, EXPECTED_DATE_FORMAT)  # noqa: DTZ007

    def test_rate_decision_is_numeric(self) -> None:
        for entry in build_event_data.CBR_DECISIONS:
            assert isinstance(entry["rate_decision"], (int, float)), (
                f"rate_decision is not numeric in {entry}"
            )

    def test_expected_rate_is_numeric(self) -> None:
        for entry in build_event_data.CBR_DECISIONS:
            assert isinstance(entry["expected_rate"], (int, float)), (
                f"expected_rate is not numeric in {entry}"
            )

    def test_surprise_bps_is_integer(self) -> None:
        for entry in build_event_data.CBR_DECISIONS:
            assert isinstance(entry["surprise_bps"], int), f"surprise_bps is not int in {entry}"

    def test_dates_are_chronologically_ordered(self) -> None:
        dates = [str(e["date"]) for e in build_event_data.CBR_DECISIONS]
        assert dates == sorted(dates), "CBR_DECISIONS dates are not sorted ascending"


# ── 2. write_cbr_decisions ───────────────────────────────────────────────────


class TestWriteCbrDecisions:
    def test_writes_json_file_to_correct_path(self, tmp_path: Path) -> None:
        build_event_data.write_cbr_decisions("2023-01-01", "2024-12-31", tmp_path)
        expected_path = tmp_path / "cbr" / "decisions.json"
        assert expected_path.exists()

    def test_output_is_valid_json(self, tmp_path: Path) -> None:
        build_event_data.write_cbr_decisions("2023-01-01", "2024-12-31", tmp_path)
        content = (tmp_path / "cbr" / "decisions.json").read_text()
        parsed = json.loads(content)
        assert isinstance(parsed, list)

    def test_output_records_have_required_keys(self, tmp_path: Path) -> None:
        build_event_data.write_cbr_decisions("2023-01-01", "2024-12-31", tmp_path)
        records = json.loads((tmp_path / "cbr" / "decisions.json").read_text())
        for record in records:
            assert record.keys() >= REQUIRED_CBR_KEYS

    def test_filtering_by_start_date(self, tmp_path: Path) -> None:
        # Only 2024 decisions should appear.
        build_event_data.write_cbr_decisions("2024-01-01", "2024-12-31", tmp_path)
        records = json.loads((tmp_path / "cbr" / "decisions.json").read_text())
        for record in records:
            assert str(record["date"]) >= "2024-01-01"

    def test_filtering_by_end_date(self, tmp_path: Path) -> None:
        # Only 2023 decisions should appear.
        build_event_data.write_cbr_decisions("2023-01-01", "2023-12-31", tmp_path)
        records = json.loads((tmp_path / "cbr" / "decisions.json").read_text())
        for record in records:
            assert str(record["date"]) <= "2023-12-31"

    def test_narrow_range_returns_single_entry(self, tmp_path: Path) -> None:
        # Only the 2023-07-21 surprise hike should appear.
        build_event_data.write_cbr_decisions("2023-07-21", "2023-07-21", tmp_path)
        records = json.loads((tmp_path / "cbr" / "decisions.json").read_text())
        assert len(records) == 1
        assert records[0]["date"] == "2023-07-21"

    def test_out_of_range_returns_empty_list(self, tmp_path: Path) -> None:
        build_event_data.write_cbr_decisions("2020-01-01", "2020-12-31", tmp_path)
        records = json.loads((tmp_path / "cbr" / "decisions.json").read_text())
        assert records == []

    def test_full_range_returns_all_entries(self, tmp_path: Path) -> None:
        build_event_data.write_cbr_decisions("2023-01-01", "2024-12-31", tmp_path)
        records = json.loads((tmp_path / "cbr" / "decisions.json").read_text())
        assert len(records) == EXPECTED_CBR_DECISIONS_COUNT


# ── 3. _all_us_symbols ───────────────────────────────────────────────────────


class TestAllUsSymbols:
    def test_returns_non_empty_list(self) -> None:
        symbols = build_event_data._all_us_symbols()
        assert len(symbols) > 0

    def test_all_symbols_are_unique(self) -> None:
        symbols = build_event_data._all_us_symbols()
        assert len(symbols) == len(set(symbols))

    def test_no_ru_segment_symbols_included(self) -> None:
        # ru_* segments use proxies (ETFs), none of which overlap with us_* symbols,
        # but we also verify by checking no symbol comes from a ru_* segment.
        us_symbols = set(build_event_data._all_us_symbols())
        for segment, symbols in build_event_data.UNIVERSE.items():
            if segment.startswith(RU_SEGMENT_PREFIXES):
                for sym in symbols:
                    # A symbol present in a ru_* segment must NOT appear in result
                    # unless it also appears in a us_* segment (it does not in this universe).
                    ru_only = sym not in {
                        s
                        for seg, syms in build_event_data.UNIVERSE.items()
                        if seg.startswith("us_")
                        for s in syms
                    }
                    if ru_only:
                        assert sym not in us_symbols, (
                            f"ru-only symbol {sym!r} found in _all_us_symbols()"
                        )

    def test_contains_only_us_segment_symbols(self) -> None:
        us_symbols = set(build_event_data._all_us_symbols())
        expected = {
            s
            for seg, syms in build_event_data.UNIVERSE.items()
            if seg.startswith("us_")
            for s in syms
        }
        assert us_symbols == expected

    def test_expected_symbol_count(self) -> None:
        symbols = build_event_data._all_us_symbols()
        assert len(symbols) == EXPECTED_US_SYMBOL_COUNT


# ── 4. _all_moex_symbols_in_universe ────────────────────────────────────────


class TestAllMoexSymbolsInUniverse:
    def test_returns_real_moex_symbols(self) -> None:
        # The ru_* segments now contain real MOEX tickers (SBER, GAZP, etc.)
        # that ARE in MOEX_FIGIS, so the result must be non-empty.
        result = build_event_data._all_moex_symbols_in_universe()
        assert len(result) > 0

    def test_returns_list_type(self) -> None:
        result = build_event_data._all_moex_symbols_in_universe()
        assert isinstance(result, list)

    def test_no_us_segment_symbols_ever_appear(self) -> None:
        result = build_event_data._all_moex_symbols_in_universe()
        us_symbols = {
            s
            for seg, syms in build_event_data.UNIVERSE.items()
            if seg.startswith("us_")
            for s in syms
        }
        for sym in result:
            assert sym not in us_symbols

    def test_only_moex_figi_members_can_appear(self) -> None:
        result = build_event_data._all_moex_symbols_in_universe()
        for sym in result:
            assert sym in build_event_data.MOEX_FIGIS


# ── 5. _write_json ───────────────────────────────────────────────────────────


class TestWriteJson:
    def test_creates_file_at_given_path(self, tmp_path: Path) -> None:
        target = tmp_path / "output.json"
        build_event_data._write_json(target, [])
        assert target.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "dir" / "data.json"
        build_event_data._write_json(target, [])
        assert target.exists()

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        target = tmp_path / "data.json"
        records: list[dict[str, object]] = [{"key": "value", "num": 42}]
        build_event_data._write_json(target, records)
        parsed = json.loads(target.read_text())
        assert parsed == records

    def test_writes_empty_list(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.json"
        build_event_data._write_json(target, [])
        parsed = json.loads(target.read_text())
        assert parsed == []

    def test_output_is_indented_json(self, tmp_path: Path) -> None:
        target = tmp_path / "indented.json"
        records: list[dict[str, object]] = [{"a": 1}]
        build_event_data._write_json(target, records)
        raw = target.read_text()
        # Indented JSON must contain newlines.
        assert "\n" in raw

    def test_multiple_records_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "multi.json"
        records: list[dict[str, object]] = [
            {"date": "2023-01-01", "value": 1.5},
            {"date": "2023-06-15", "value": 2.0},
        ]
        build_event_data._write_json(target, records)
        parsed = json.loads(target.read_text())
        assert len(parsed) == len(records)
        assert parsed[0]["date"] == "2023-01-01"
        assert parsed[1]["date"] == "2023-06-15"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "overwrite.json"
        build_event_data._write_json(target, [{"first": True}])
        build_event_data._write_json(target, [{"second": True}])
        parsed = json.loads(target.read_text())
        assert len(parsed) == 1
        assert parsed[0] == {"second": True}

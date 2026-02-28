"""Unit tests for train/test splitter (#134)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from finalayze.ml.training.splitter import LabelledRow, temporal_train_test_split

# Constants — no magic numbers (ruff PLR2004)
N_ROWS = 100
TEST_FRACTION = 0.2
EXPECTED_TEST_SIZE = int(N_ROWS * TEST_FRACTION)
EXPECTED_TRAIN_SIZE = N_ROWS - EXPECTED_TEST_SIZE
MIN_ROWS = 2


def _make_rows(
    n: int = N_ROWS,
    n_symbols: int = 1,
    shuffled: bool = False,
) -> list[LabelledRow]:
    """Create *n* synthetic labelled rows across *n_symbols* symbols.

    When *shuffled* is True the rows are returned in random order to verify
    that the splitter sorts them before splitting.
    """
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[LabelledRow] = []
    for i in range(n):
        symbol = f"SYM_{i % n_symbols}"
        rows.append(
            LabelledRow(
                timestamp=base + timedelta(days=i),
                symbol=symbol,
                features={"feat_0": float(i)},
                label=i % 2,
            )
        )
    if shuffled:
        import random  # noqa: PLC0415

        random.Random(42).shuffle(rows)  # noqa: S311
    return rows


class TestTemporalTrainTestSplit:
    def test_split_produces_correct_sizes(self) -> None:
        rows = _make_rows(N_ROWS)
        train, test = temporal_train_test_split(rows, test_fraction=TEST_FRACTION)
        assert len(train) == EXPECTED_TRAIN_SIZE
        assert len(test) == EXPECTED_TEST_SIZE

    def test_train_comes_before_test_in_time(self) -> None:
        """All training timestamps must be strictly earlier than all test timestamps."""
        rows = _make_rows(N_ROWS)
        train, test = temporal_train_test_split(rows, test_fraction=TEST_FRACTION)
        latest_train = max(r["timestamp"] for r in train)
        earliest_test = min(r["timestamp"] for r in test)
        assert latest_train < earliest_test

    def test_multi_symbol_temporal_ordering_preserved(self) -> None:
        """Multi-symbol datasets must still be split on global time order (#134)."""
        N_SYMBOLS = 5
        rows = _make_rows(N_ROWS, n_symbols=N_SYMBOLS, shuffled=True)
        train, test = temporal_train_test_split(rows, test_fraction=TEST_FRACTION)
        if train and test:
            latest_train = max(r["timestamp"] for r in train)
            earliest_test = min(r["timestamp"] for r in test)
            assert latest_train < earliest_test

    def test_shuffled_input_produces_same_split_as_sorted(self) -> None:
        """Shuffling the input rows must not affect the temporal split result (#134)."""
        rows_sorted = _make_rows(N_ROWS, shuffled=False)
        rows_shuffled = _make_rows(N_ROWS, shuffled=True)

        train1, test1 = temporal_train_test_split(rows_sorted, test_fraction=TEST_FRACTION)
        train2, test2 = temporal_train_test_split(rows_shuffled, test_fraction=TEST_FRACTION)

        ts1_train = [r["timestamp"] for r in train1]
        ts2_train = [r["timestamp"] for r in train2]
        assert ts1_train == ts2_train

        ts1_test = [r["timestamp"] for r in test1]
        ts2_test = [r["timestamp"] for r in test2]
        assert ts1_test == ts2_test

    def test_train_and_test_cover_all_rows(self) -> None:
        rows = _make_rows(N_ROWS)
        train, test = temporal_train_test_split(rows, test_fraction=TEST_FRACTION)
        assert len(train) + len(test) == N_ROWS

    def test_empty_rows_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            temporal_train_test_split([], test_fraction=TEST_FRACTION)

    def test_invalid_test_fraction_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="test_fraction"):
            temporal_train_test_split(_make_rows(MIN_ROWS), test_fraction=0.0)

    def test_invalid_test_fraction_one_raises(self) -> None:
        with pytest.raises(ValueError, match="test_fraction"):
            temporal_train_test_split(_make_rows(MIN_ROWS), test_fraction=1.0)

    def test_output_rows_are_sorted_ascending(self) -> None:
        rows = _make_rows(N_ROWS, shuffled=True)
        train, test = temporal_train_test_split(rows, test_fraction=TEST_FRACTION)
        train_ts = [r["timestamp"] for r in train]
        test_ts = [r["timestamp"] for r in test]
        assert train_ts == sorted(train_ts)
        assert test_ts == sorted(test_ts)

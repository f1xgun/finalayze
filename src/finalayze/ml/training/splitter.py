"""Train/test split utilities for time-series ML datasets (Layer 3).

When a dataset contains multiple symbols, rows MUST be sorted by timestamp
before splitting so that the train set always precedes the test set in time.
Shuffling before splitting would introduce look-ahead bias (#134).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from datetime import datetime


class LabelledRow(TypedDict):
    """A single labelled observation with a timestamp and symbol identifier."""

    timestamp: datetime
    symbol: str
    features: dict[str, float]
    label: int


def temporal_train_test_split(
    rows: list[LabelledRow],
    test_fraction: float = 0.2,
) -> tuple[list[LabelledRow], list[LabelledRow]]:
    """Split a (potentially multi-symbol) dataset into train and test sets.

    The split is performed on rows sorted ascending by ``timestamp`` so that
    the test set always contains the most-recent observations.  Shuffling is
    intentionally avoided because shuffling time-series data leaks future
    information into the training set (#134).

    Args:
        rows: Labelled observations, possibly from multiple symbols.
        test_fraction: Fraction of observations to reserve for the test set.
            Must be in (0, 1).

    Returns:
        ``(train_rows, test_rows)`` — each list sorted ascending by timestamp.

    Raises:
        ValueError: When ``test_fraction`` is not in (0, 1), or ``rows`` is empty.
    """
    if not rows:
        msg = "Cannot split an empty dataset"
        raise ValueError(msg)
    if not 0.0 < test_fraction < 1.0:
        msg = f"test_fraction must be in (0, 1), got {test_fraction}"
        raise ValueError(msg)

    # Sort by timestamp first — this is the only safe ordering for time-series.
    sorted_rows = sorted(rows, key=lambda r: r["timestamp"])

    split_idx = max(1, int(len(sorted_rows) * (1.0 - test_fraction)))
    return sorted_rows[:split_idx], sorted_rows[split_idx:]

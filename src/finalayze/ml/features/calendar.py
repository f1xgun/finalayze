"""Calendar feature computation (Layer 3)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

# Calendar cyclical encoding constants
_MONTHS_PER_YEAR = 12


def compute_calendar_features(
    last_timestamp: datetime,
) -> dict[str, float]:
    """Compute cyclical calendar encoding from the last candle's timestamp.

    No look-ahead bias: uses only the timestamp of the most recent candle.
    Encodes month as sin/cos pair for cyclical continuity.
    Day-of-week encoding removed (negligible effect post-2000, Sullivan et al. 2001).
    """
    month = last_timestamp.month  # 1-12

    two_pi = 2.0 * math.pi
    return {
        "month_sin": math.sin(two_pi * month / _MONTHS_PER_YEAR),
        "month_cos": math.cos(two_pi * month / _MONTHS_PER_YEAR),
    }

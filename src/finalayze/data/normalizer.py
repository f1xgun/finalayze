"""Market data normalizer — validates and tags candles (Layer 2)."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.exceptions import DataFetchError

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

logger = logging.getLogger(__name__)

_ZERO = Decimal(0)


class DataNormalizer:
    """Validates OHLCV candles and tags them with market_id and source."""

    def __init__(self, market_id: str, source: str) -> None:
        self._market_id = market_id
        self._source = source

    def normalize(self, candle: Candle) -> Candle:
        """Validate and tag a single candle. Raises DataFetchError if invalid."""
        self._validate(candle)
        return candle.model_copy(update={"market_id": self._market_id, "source": self._source})

    def normalize_batch(self, candles: list[Candle]) -> list[Candle]:
        """Normalize a batch, skipping invalid candles with a warning."""
        result: list[Candle] = []
        for candle in candles:
            try:
                result.append(self.normalize(candle))
            except DataFetchError as exc:
                logger.warning(
                    "Skipping invalid candle %s@%s: %s",
                    candle.symbol,
                    candle.timestamp,
                    exc,
                )
        return result

    def _validate(self, candle: Candle) -> None:
        """Raise DataFetchError if the candle fails OHLCV integrity checks."""
        if (
            candle.open <= _ZERO
            or candle.high <= _ZERO
            or candle.low <= _ZERO
            or candle.close <= _ZERO
        ):
            msg = (
                f"Candle has non-positive price: open={candle.open}, high={candle.high},"
                f" low={candle.low}, close={candle.close}"
            )
            raise DataFetchError(msg)
        if candle.low > candle.high:
            msg = f"Candle low {candle.low} > high {candle.high}"
            raise DataFetchError(msg)
        if not (candle.low <= candle.close <= candle.high):
            msg = f"Candle close {candle.close} outside [low={candle.low}, high={candle.high}]"
            raise DataFetchError(msg)
        if not (candle.low <= candle.open <= candle.high):
            msg = f"Candle open {candle.open} outside [low={candle.low}, high={candle.high}]"
            raise DataFetchError(msg)

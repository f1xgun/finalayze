"""Abstract base for trading strategies (Layer 4)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from finalayze.core.schemas import Candle, Signal  # noqa: TC001


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def supported_segments(self) -> list[str]: ...

    @abstractmethod
    def generate_signal(
        self, symbol: str, candles: list[Candle], segment_id: str, sentiment_score: float = 0.0
    ) -> Signal | None: ...

    @abstractmethod
    def get_parameters(self, segment_id: str) -> dict[str, object]: ...

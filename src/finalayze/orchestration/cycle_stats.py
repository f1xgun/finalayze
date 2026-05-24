"""Typed counters returned by SignalExecutor.process_instrument.

Replaces the legacy ``dict[str, Any]`` return shape. Immutability and the
``__add__`` operator give TradingLoop a clean accumulator for per-cycle stats
without scattering ``stats.get("...", 0)`` lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class CycleStats:
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    errors_caught: int = 0
    dropped_no_bars: int = 0
    dropped_below_threshold: int = 0
    dropped_pre_trade: int = 0

    def __add__(self, other: CycleStats) -> CycleStats:
        if not isinstance(other, CycleStats):
            return NotImplemented
        return CycleStats(
            **{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)}
        )

    @classmethod
    def no_bars(cls) -> CycleStats:
        return cls(dropped_no_bars=1)

    @classmethod
    def error_caught(cls) -> CycleStats:
        return cls(errors_caught=1)

    @classmethod
    def signal_generated(cls) -> CycleStats:
        return cls(signals_generated=1)

    @classmethod
    def signal_dropped_threshold(cls) -> CycleStats:
        return cls(dropped_below_threshold=1)

    @classmethod
    def pre_trade_rejected(cls) -> CycleStats:
        # A signal was generated and then rejected at the pre-trade gate;
        # preserve the legacy contract that signals_generated counts the signal.
        return cls(signals_generated=1, dropped_pre_trade=1)

    @classmethod
    def order_submitted(cls, *, filled: bool) -> CycleStats:
        return cls(
            signals_generated=1,
            orders_submitted=1,
            orders_filled=1 if filled else 0,
        )

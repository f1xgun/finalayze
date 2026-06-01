"""Engine-level CARDINAL D-05 no-look-ahead proof over ``run_portfolio``.

Mirrors ``tests/unit/test_pead_sue_proxy.py::TestA4LookAhead`` (the append-future-no-op +
in-window-changes shape), but proves the property at the BACKTEST-ENGINE level: the as-of
eligible-universe gate threaded into ``BacktestEngine.run_portfolio`` recomputes the eligible
set at each QUARTERLY rebalance bar from ONLY the candles dated ``<= ts``.

Two cardinal proofs (D-05 / LIQ-04):

1. ``test_future_candle_does_not_change_entries`` -- appending a FUTURE high-turnover candle to a
   borderline (sub-60-bar) symbol does NOT change which symbols are entered (the ``<= ts`` cutoff
   filtered the future bar -- no look-ahead). The borderline name stays excluded.
2. ``test_in_window_candle_changes_eligibility`` -- moving that SAME candle into the ``<= T`` window
   (so the borderline name now has the full 60-bar window) flips it into the eligible set, so it IS
   entered (proves the gate is LIVE, not a global no-op).

Plus ``test_cap_enforced_across_symbols`` -- the shared-broker D-09 concurrent-position cap (also
extended in ``test_engine_segment_cap.py`` per the plan), proven here over the as-of gate path.

The mechanism is the Plan-01 ``eligible_universe_as_of`` exclusion rule: a name with < 60 visible
bars (as of the rebalance) scores ``None`` and is excluded; a name with >= 60 visible bars and the
highest sector turnover wins the Top-N slot. The borderline symbol sits at exactly 59 visible bars
so a single in-window-vs-future 60th bar flips eligibility -- a faithful append-future / in-window
proof at the engine level.

All fixtures are deterministic: no live token, no DB, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from finalayze.backtest.config import BacktestConfig
from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection, TradeResult
from finalayze.markets.liquidity import eligible_universe_as_of
from finalayze.strategies.base import BaseStrategy

# ── Named constants (no magic numbers -- ruff PLR2004) ──────────────────────────
_WINDOW = 60  # D-02 trailing liquidity window (bars)
_TOP_N = 2  # keep both names when both are eligible (sector is shared)
_SECTOR = "oil_gas"  # one shared sector so within-sector eligibility is the lever
_SEGMENT = "ru_energy"  # the segment oil_gas maps to (config.SECTOR_TO_SEGMENT)
_BASE_PRICE = 100
_HIGH_VOLUME = 10_000_000  # the borderline symbol's turnover on its 60th bar
_LIQUID_VOLUME = 5_000_000  # the always-eligible liquid name's steady turnover
_START = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)

# Timeline / borderline-symbol layout (all 2024, only two quarter boundaries occur: 2024-Q1 at
# day 0 and 2024-Q2 at day 91 = 2024-04-01; the timeline ends at day 120 = 2024-04-30, still Q2,
# so NO third rebalance fires and the Q2 eligible set is carried forward to the run's end).
#
# BORDER carries exactly 59 daily bars dated <= the Q2 rebalance (days 31..89, all 2024-Q1), so as
# of the day-91 Q2 rebalance it has < _WINDOW visible bars -> scores None -> EXCLUDED. It also
# trades days 92..120 so that IF the Q2 eligible set admits it, it CAN actually be entered (a
# symbol can only be entered on bars where it has a candle). The pivotal 60th bar is placed EITHER
# after the rebalance (day 100, still Q2 -> > 91 so the as-of cutoff filters it -> still 59 visible
# -> no-op) OR in-window (day 90 -> <= 91 -> 60 visible -> eligible). The day-100 "future" bar stays
# inside Q2 so it does NOT trigger an extra rebalance that could re-admit BORDER.
_BORDER_PRE_DAYS = list(range(31, 90))  # 59 bars, all <= the Q2 rebalance
_BORDER_POST_DAYS = list(range(92, 121))  # tradeable bars AFTER the Q2 rebalance (skip 90, 91)
_BORDER_FUTURE_DAY = 100  # > the Q2 rebalance (day 91), same quarter -> filtered, no new rebalance
_BORDER_IN_WINDOW_DAY = 90  # <= the Q2 rebalance -> counted -> 60 visible bars
_LIQUID_BARS = 121  # days 0..120; spans Q1 + the start of Q2 so a Q2 rebalance occurs


def _candle(symbol: str, day: int, *, close: float, volume: int) -> Candle:
    price = Decimal(str(close))
    return Candle(
        symbol=symbol,
        market_id="moex",
        timeframe="1d",
        timestamp=_START + timedelta(days=day),
        open=price,
        high=price + Decimal(2),
        low=price - Decimal(2),
        close=price,
        volume=volume,
    )


def _series(symbol: str, *, n: int, close: float, volume: int, start_day: int = 0) -> list[Candle]:
    return [_candle(symbol, start_day + i, close=close, volume=volume) for i in range(n)]


class _AlwaysBuyStrategy(BaseStrategy):
    """Emits a BUY for any symbol with no open position, on every bar with >= 1 bar of history.

    Universe-agnostic: the as-of eligible-universe gate in ``run_portfolio`` is the ONLY thing
    that decides which symbols can actually be entered, so any difference in ENTERED symbols is
    attributable to the gate, not the strategy.
    """

    @property
    def name(self) -> str:
        return "always_buy"

    def supported_segments(self) -> list[str]:
        return [_SEGMENT]

    def generate_signal(  # type: ignore[override]
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        **kwargs: object,
    ) -> Signal | None:
        if kwargs.get("has_open_position"):
            return None
        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id="moex",
            segment_id=segment_id,
            direction=SignalDirection.BUY,
            confidence=0.9,
            strategy_payload={"momentum": 1.0},
            reasoning="test buy",
        )

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


def _entered_symbols(trades: list[TradeResult]) -> set[str]:
    """The set of symbols that opened a position (i.e. were ENTERED) during the run."""
    return {t.symbol for t in trades}


def _build_engine() -> BacktestEngine:
    return BacktestEngine(
        strategy=_AlwaysBuyStrategy(),
        config=BacktestConfig(
            initial_cash=Decimal(1_000_000),
            max_positions=10,
            force_close_at_end=True,
        ),
    )


def _eligible_at_factory(candles_by_symbol: dict[str, list[Candle]]):
    """Build the ``eligible_at(ts)`` callback backed by the as-of liquidity primitive.

    ``eligible_universe_as_of`` slices each symbol to ``timestamp <= ts`` internally -- the engine
    consults this callback at quarterly rebalance bars; it must NOT see future bars for entries.
    """
    sector_map = dict.fromkeys(candles_by_symbol, _SECTOR)

    def eligible_at(ts: datetime) -> set[str]:
        return eligible_universe_as_of(candles_by_symbol, ts, sector_map, _TOP_N)

    return eligible_at


def _base_universe() -> dict[str, list[Candle]]:
    """Liquid always-winner (days 0..120) + a borderline name (59 bars <= Q2, then trades)."""
    liquid = _series("LIQUID", n=_LIQUID_BARS, close=_BASE_PRICE, volume=_LIQUID_VOLUME)
    border = [
        _candle("BORDER", d, close=_BASE_PRICE, volume=_LIQUID_VOLUME)
        for d in (*_BORDER_PRE_DAYS, *_BORDER_POST_DAYS)
    ]
    return {"LIQUID": liquid, "BORDER": border}


class TestEngineAsOfNoLookAhead:
    """CARDINAL D-05 proof at the run_portfolio level."""

    def test_future_candle_does_not_change_entries(self) -> None:
        """Appending a FUTURE candle must NOT change which symbols are entered (as-of cutoff)."""
        candles_by_symbol = _base_universe()

        engine = _build_engine()
        trades, _snaps = engine.run_portfolio(
            list(candles_by_symbol),
            _SEGMENT,
            candles_by_symbol,
            eligible_at=_eligible_at_factory(candles_by_symbol),
        )
        baseline = _entered_symbols(trades)

        # BORDER has only 59 visible bars as of the Q2 rebalance -> excluded -> never entered.
        assert "BORDER" not in baseline
        assert "LIQUID" in baseline

        # Add BORDER's pivotal 60th bar AFTER the Q2 rebalance (day 100 > day 91, same quarter so
        # no new rebalance fires). The as-of cutoff (<= 91) must filter it -> BORDER stays at 59
        # visible bars as of the Q2 rebalance -> still excluded for the rest of the run.
        future = dict(candles_by_symbol)
        future["BORDER"] = [
            *candles_by_symbol["BORDER"],
            _candle("BORDER", _BORDER_FUTURE_DAY, close=_BASE_PRICE, volume=_HIGH_VOLUME),
        ]
        engine2 = _build_engine()
        trades2, _snaps2 = engine2.run_portfolio(
            list(future),
            _SEGMENT,
            future,
            eligible_at=_eligible_at_factory(future),
        )
        after_future = _entered_symbols(trades2)

        # Future candle is a NO-OP for entries (D-05 engine proof).
        assert "BORDER" not in after_future
        assert after_future == baseline

    def test_in_window_candle_changes_eligibility(self) -> None:
        """Moving the SAME candle into the <= T window flips BORDER eligible (gate is live)."""
        candles_by_symbol = _base_universe()

        # Move BORDER's pivotal 60th bar INTO the window (day 90 <= the Q2 rebalance at day 91).
        # BORDER now has 60 visible bars as of the Q2 rebalance -> scores a turnover -> eligible.
        in_window = dict(candles_by_symbol)
        in_window["BORDER"] = [
            *candles_by_symbol["BORDER"],
            _candle("BORDER", _BORDER_IN_WINDOW_DAY, close=_BASE_PRICE, volume=_HIGH_VOLUME),
        ]

        engine = _build_engine()
        trades, _snaps = engine.run_portfolio(
            list(in_window),
            _SEGMENT,
            in_window,
            eligible_at=_eligible_at_factory(in_window),
        )
        entered = _entered_symbols(trades)

        # With the 60th bar IN-WINDOW, BORDER now has >= _WINDOW visible bars at the Q2 rebalance
        # -> eligible -> IS entered. The very same candle that was a no-op in the future is live
        # in-window: the gate is real, not a global no-op.
        assert "BORDER" in entered

    def test_cap_enforced_across_symbols(self) -> None:
        """Per-segment concurrent-position cap holds ACROSS symbols in the shared-broker run."""
        cap = 2
        n_symbols = 5
        candles_by_symbol = {
            f"SYM{i}": _series(
                f"SYM{i}", n=_LIQUID_BARS, close=_BASE_PRICE + i, volume=_LIQUID_VOLUME
            )
            for i in range(n_symbols)
        }
        engine = BacktestEngine(
            strategy=_AlwaysBuyStrategy(),
            config=BacktestConfig(
                initial_cash=Decimal(1_000_000),
                max_concurrent_positions=cap,
                force_close_at_end=True,
            ),
        )
        sector_map = dict.fromkeys(candles_by_symbol, _SECTOR)

        def eligible_at(ts: datetime) -> set[str]:
            # Top-N large enough that the eligible set never limits below the cap.
            return eligible_universe_as_of(candles_by_symbol, ts, sector_map, n_symbols)

        _trades, snaps = engine.run_portfolio(
            list(candles_by_symbol),
            _SEGMENT,
            candles_by_symbol,
            eligible_at=eligible_at,
        )
        max_open = max(
            (sum(1 for q in s.positions.values() if q > 0) for s in snaps),
            default=0,
        )
        assert max_open <= cap

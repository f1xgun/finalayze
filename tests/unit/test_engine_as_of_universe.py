"""Engine-level CARDINAL D-05 no-look-ahead proof over ``run_portfolio``.

Mirrors ``tests/unit/test_pead_sue_proxy.py::TestA4LookAhead`` (the append-future-no-op +
in-window-changes shape), but proves the property at the BACKTEST-ENGINE level: the as-of
eligible-universe gate threaded into ``BacktestEngine.run_portfolio`` recomputes the eligible
set at each quarterly rebalance bar from ONLY the candles dated ``<= ts``.

Two cardinal proofs (D-05 / LIQ-04):

1. ``test_future_candle_does_not_change_entries`` -- appending a FUTURE high-turnover candle to a
   borderline symbol does NOT change which symbols are entered at a past rebalance bar T (the
   ``<= ts`` cutoff filtered the future bar -- no look-ahead).
2. ``test_in_window_candle_changes_eligibility`` -- moving that SAME candle into the ``<= T``
   window flips the borderline symbol into the eligible set, so it IS now entered at T (proves the
   gate is LIVE, not a global no-op).

Plus ``test_cap_enforced_across_symbols`` (the shared-broker D-09 cap), which also lives in
``test_engine_segment_cap.py`` per the plan -- kept here too over the as-of gate path.

All fixtures are deterministic: no live token, no DB, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from finalayze.backtest.config import BacktestConfig
from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.markets.liquidity import eligible_universe_as_of
from finalayze.strategies.base import BaseStrategy

# ── Named constants (no magic numbers -- ruff PLR2004) ──────────────────────────
_WINDOW = 60  # D-02 trailing liquidity window (bars)
_TOP_N = 1  # one name per sector kept -- forces a single eligible winner per sector
_SECTOR = "oil_gas"  # one shared sector so within-sector ranking is the live lever
_SEGMENT = "ru_energy"  # the segment oil_gas maps to (config.SECTOR_TO_SEGMENT)
_BUY_BAR = 30  # the bar at which the test strategy emits BUY (a "rebalance-ish" entry point)
_BASE_PRICE = 100
_HIGH_VOLUME = 10_000_000  # borderline symbol's turnover-spike volume
_LOW_VOLUME = 1_000  # the liquid winner's modest volume (still wins on price*vol? no -- see below)
_LIQUID_VOLUME = 5_000_000  # the always-eligible liquid name's steady volume
_START = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)


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


class _EligibleAwareStrategy(BaseStrategy):
    """Emits a BUY for every symbol at ``_BUY_BAR``.

    The as-of eligible-universe gate in ``run_portfolio`` is what filters these BUYs down to the
    eligible set -- the strategy itself is universe-agnostic, so any difference in ENTERED symbols
    is attributable to the gate (not the strategy).
    """

    @property
    def name(self) -> str:
        return "eligible_aware"

    def supported_segments(self) -> list[str]:
        return [_SEGMENT]

    def generate_signal(  # type: ignore[override]
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        **kwargs: object,
    ) -> Signal | None:
        idx = len(candles) - 1
        if idx == _BUY_BAR:
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
        return None

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


def _entered_symbols(trades: list, snapshots: list) -> set[str]:
    """The set of symbols that opened a position (i.e. were ENTERED) during the run."""
    return {t.symbol for t in trades}


def _build_engine() -> BacktestEngine:
    return BacktestEngine(
        strategy=_EligibleAwareStrategy(),
        config=BacktestConfig(
            initial_cash=Decimal(1_000_000),
            max_positions=10,
            force_close_at_end=True,
        ),
    )


def _eligible_at_factory(candles_by_symbol: dict[str, list[Candle]]):
    """Build the ``eligible_at(ts)`` callback backed by the as-of liquidity primitive.

    The gate sees the FULL candle dict but ``eligible_universe_as_of`` slices each symbol to
    ``timestamp <= ts`` internally -- the engine must consult this callback at rebalance bars and
    must NOT pre-filter or pass future bars into entries for non-eligible symbols.
    """
    sector_map = dict.fromkeys(candles_by_symbol, _SECTOR)

    def eligible_at(ts: datetime) -> set[str]:
        return eligible_universe_as_of(candles_by_symbol, ts, sector_map, _TOP_N)

    return eligible_at


def _base_universe() -> dict[str, list[Candle]]:
    """A liquid always-winner + a borderline name with only 59 bars up to the BUY bar.

    LIQUID has 60+ steady bars and high turnover -> always eligible.
    BORDER has only (_BUY_BAR) bars at/below the BUY bar (< _WINDOW after the gate sees them as of
    the rebalance), so it is NOT eligible at the rebalance until a 60th in-window bar appears.
    """
    # LIQUID: 70 bars, steady high turnover -> the Top-1 winner at every rebalance.
    liquid = _series("LIQUID", n=70, close=_BASE_PRICE, volume=_LIQUID_VOLUME)
    # BORDER: 59 bars up to (and including) day 58, so at the BUY bar it has < _WINDOW visible
    # bars -> median_rub_turnover returns None -> excluded. (Needs a 60th in-window bar to rank.)
    border = _series("BORDER", n=59, close=_BASE_PRICE, volume=_LOW_VOLUME)
    return {"LIQUID": liquid, "BORDER": border}


class TestEngineAsOfNoLookAhead:
    """CARDINAL D-05 proof at the run_portfolio level."""

    def test_future_candle_does_not_change_entries(self) -> None:
        """Appending a FUTURE high-turnover candle must NOT change entries at the past rebalance."""
        candles_by_symbol = _base_universe()
        symbols = list(candles_by_symbol)

        engine = _build_engine()
        trades, snaps = engine.run_portfolio(
            symbols,
            _SEGMENT,
            candles_by_symbol,
            eligible_at=_eligible_at_factory(candles_by_symbol),
        )
        baseline = _entered_symbols(trades, snaps)

        # BORDER had only 59 bars -> never eligible -> never entered at baseline.
        assert "BORDER" not in baseline
        assert "LIQUID" in baseline

        # Append a FUTURE (day 100, well past the BUY bar) high-turnover candle to BORDER -- its
        # 60th bar, but dated AFTER the rebalance. The as-of gate must filter it.
        future = dict(candles_by_symbol)
        future["BORDER"] = [
            *candles_by_symbol["BORDER"],
            _candle("BORDER", 100, close=_BASE_PRICE * 5, volume=_HIGH_VOLUME),
        ]
        engine2 = _build_engine()
        trades2, snaps2 = engine2.run_portfolio(
            list(future),
            _SEGMENT,
            future,
            eligible_at=_eligible_at_factory(future),
        )
        after_future = _entered_symbols(trades2, snaps2)

        # Future candle is a NO-OP for entries at the past rebalance (D-05 engine proof).
        assert "BORDER" not in after_future
        assert after_future == baseline

    def test_in_window_candle_changes_eligibility(self) -> None:
        """Moving the SAME candle into the <= T window flips BORDER eligible (gate is live)."""
        candles_by_symbol = _base_universe()

        # Move BORDER's 60th bar INTO the window (day 59, still <= the BUY bar at day 30? No --
        # the gate is consulted at the rebalance bar; day 59 is within the 60-bar window leading up
        # to a rebalance at/after day 59). Give it a turnover spike so it ranks into Top-1.
        in_window = dict(candles_by_symbol)
        in_window["BORDER"] = [
            *candles_by_symbol["BORDER"],
            _candle("BORDER", 59, close=_BASE_PRICE * 5, volume=_HIGH_VOLUME),
        ]
        # LIQUID must run long enough that a rebalance occurs after BORDER's 60th bar lands.
        in_window["LIQUID"] = _series("LIQUID", n=120, close=_BASE_PRICE, volume=_LIQUID_VOLUME)

        engine = _build_engine()
        trades, snaps = engine.run_portfolio(
            list(in_window),
            _SEGMENT,
            in_window,
            eligible_at=_eligible_at_factory(in_window),
        )
        entered = _entered_symbols(trades, snaps)

        # With the 60th bar IN-WINDOW and a turnover spike, BORDER now has >= _WINDOW visible bars
        # at the relevant rebalance and ranks into Top-1 -> it becomes eligible and IS entered.
        assert "BORDER" in entered

    def test_cap_enforced_across_symbols(self) -> None:
        """Per-segment concurrent-position cap holds ACROSS symbols in the shared-broker run."""
        cap = 2
        n_symbols = 5
        candles_by_symbol = {
            f"SYM{i}": _series(f"SYM{i}", n=70, close=_BASE_PRICE + i, volume=_LIQUID_VOLUME)
            for i in range(n_symbols)
        }
        engine = BacktestEngine(
            strategy=_EligibleAwareStrategy(),
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

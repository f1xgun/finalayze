"""Backtest journal bridge -- records decisions during engine execution.

Translates engine-level events (BUY / SELL / SKIP) into structured
:class:`~finalayze.backtest.decision_journal.DecisionRecord` entries,
extracting per-strategy contributions from a
:class:`~finalayze.backtest.journaling_combiner.JournalingStrategyCombiner`
when present.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.backtest.decision_journal import (
    CandleSnapshot,
    DecisionJournal,
    FinalAction,
    StrategySignalRecord,
)

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
    from finalayze.core.schemas import Candle, Signal
    from finalayze.execution.simulated_broker import SimulatedBroker
    from finalayze.strategies.base import BaseStrategy


class BacktestJournal:
    """Bridge between the backtest engine and :class:`DecisionJournal`.

    The engine delegates all journal-related bookkeeping here so that
    ``engine.py`` does not need to know about candle snapshots, strategy
    signal decomposition, or model feature extraction.
    """

    def __init__(
        self,
        decision_journal: DecisionJournal | None,
        strategy: BaseStrategy,
    ) -> None:
        self._journal = decision_journal
        self._strategy = strategy

    @property
    def has_journal(self) -> bool:
        """Return True when an underlying DecisionJournal is attached."""
        return self._journal is not None

    # ------------------------------------------------------------------
    # Public API used by the engine
    # ------------------------------------------------------------------

    def record_decision(
        self,
        *,
        action: FinalAction,
        timestamp: datetime,
        symbol: str,
        segment_id: str,
        broker: SimulatedBroker,
        history: list[Candle] | None = None,
        signal: Signal | None = None,
        skip_reason: str | None = None,
        pre_trade_passed: bool | None = None,
        pre_trade_violations: list[str] | None = None,
        position_value: Decimal | None = None,
        quantity: Decimal | None = None,
        fill_price: Decimal | None = None,
        stop_loss_price: Decimal | None = None,
        cb_level: str = "normal",
    ) -> None:
        """Record a decision in the journal (no-op when journal is ``None``)."""
        if self._journal is None:
            return

        portfolio = broker.get_portfolio()

        # Build recent candle snapshots (last 5)
        recent: list[CandleSnapshot] = [
            CandleSnapshot(
                timestamp=c.timestamp,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
            )
            for c in (history[-5:] if history else [])
        ]

        # Extract per-strategy signals if using JournalingStrategyCombiner
        strategy_signals, net_score, dominant = self._extract_strategy_signals()

        # Capture enriched features and model probas from combiner
        strategy_features, model_probas = self._extract_features()

        self._journal.record(
            self._journal.make_record(
                timestamp=timestamp,
                symbol=symbol,
                segment_id=segment_id,
                final_action=action,
                skip_reason=skip_reason,
                strategy_signals=strategy_signals,
                combined_direction=signal.direction.value if signal else None,
                combined_confidence=signal.confidence if signal else None,
                net_weighted_score=net_score,
                dominant_strategy=dominant,
                pre_trade_passed=pre_trade_passed,
                pre_trade_violations=pre_trade_violations or [],
                position_value=position_value,
                quantity=quantity,
                fill_price=fill_price,
                stop_loss_price=stop_loss_price,
                circuit_breaker_level=cb_level,
                portfolio_equity=portfolio.equity,
                portfolio_cash=portfolio.cash,
                open_position_count=len(portfolio.positions),
                recent_candles=recent,
                strategy_features=strategy_features,
                model_probas=model_probas,
            )
        )

    def record_skip(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        segment_id: str,
        broker: SimulatedBroker,
        history: list[Candle] | None = None,
        skip_reason: str,
        cb_level: str = "normal",
    ) -> None:
        """Convenience wrapper for journaling a SKIP decision."""
        if self._journal is None:
            return
        self.record_decision(
            action=FinalAction.SKIP,
            timestamp=timestamp,
            symbol=symbol,
            segment_id=segment_id,
            broker=broker,
            history=history,
            skip_reason=skip_reason,
            cb_level=cb_level,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_journaling_combiner(self) -> bool:
        """Check whether the strategy is a JournalingStrategyCombiner."""
        from finalayze.backtest.journaling_combiner import (  # noqa: PLC0415
            JournalingStrategyCombiner,
        )

        return isinstance(self._strategy, JournalingStrategyCombiner)

    def _as_journaling_combiner(self) -> JournalingStrategyCombiner | None:
        """Return the strategy cast to JournalingStrategyCombiner, or None."""
        from finalayze.backtest.journaling_combiner import (  # noqa: PLC0415
            JournalingStrategyCombiner,
        )

        if isinstance(self._strategy, JournalingStrategyCombiner):
            return self._strategy
        return None

    def _extract_strategy_signals(
        self,
    ) -> tuple[list[StrategySignalRecord], float | None, str | None]:
        """Extract per-strategy signal records from a JournalingStrategyCombiner.

        Returns:
            (strategy_signals, net_score, dominant_strategy_name)
        """
        from finalayze.core.schemas import SignalDirection  # noqa: PLC0415

        combiner = self._as_journaling_combiner()
        if combiner is None:
            return [], None, None

        strategy_signals: list[StrategySignalRecord] = []
        for name, sig in combiner.last_signals.items():
            weight = combiner.last_weights.get(name, Decimal("1.0"))
            if sig is not None:
                dir_score = Decimal(1) if sig.direction == SignalDirection.BUY else Decimal(-1)
                contribution = dir_score * Decimal(str(sig.confidence)) * weight
                strategy_signals.append(
                    StrategySignalRecord(
                        strategy_name=name,
                        direction=sig.direction.value,
                        confidence=sig.confidence,
                        weight=weight,
                        contribution=contribution,
                    )
                )
            else:
                strategy_signals.append(
                    StrategySignalRecord(
                        strategy_name=name,
                        direction=None,
                        confidence=None,
                        weight=weight,
                        contribution=Decimal(0),
                    )
                )

        net_score = combiner.last_net_score

        # Identify the strategy with the highest absolute contribution
        dominant: str | None = None
        if strategy_signals:
            firing = [s for s in strategy_signals if s.direction is not None]
            if firing:
                dominant = max(firing, key=lambda s: abs(s.contribution)).strategy_name

        return strategy_signals, net_score, dominant

    def _extract_features(
        self,
    ) -> tuple[dict[str, float] | None, dict[str, float] | None]:
        """Extract enriched features and model probas from a combiner.

        Returns:
            (strategy_features, model_probas)
        """
        combiner = self._as_journaling_combiner()
        if combiner is None:
            return None, None

        strategy_features: dict[str, float] | None = None
        feats = combiner.last_features
        if feats:
            strategy_features = feats
        model_probas = combiner.last_model_probas

        return strategy_features, model_probas

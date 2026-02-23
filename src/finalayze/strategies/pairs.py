"""Pairs trading strategy using cointegration-based spread z-scores (Layer 4)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import yaml
from statsmodels.tsa.stattools import coint

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"
_MIN_CANDLES = 60
_COINT_P_THRESHOLD = 0.05
_PAIR_LENGTH = 2


class PairsStrategy(BaseStrategy):
    """Statistical arbitrage via Engle-Granger cointegration spread z-score.

    Usage:
        strategy = PairsStrategy()
        strategy.set_peer_candles("MSFT", msft_candles)
        signal = strategy.generate_signal("AAPL", aapl_candles, "us_tech")
    """

    def __init__(self) -> None:
        self._peer_candles: dict[str, list[Candle]] = {}

    @property
    def name(self) -> str:
        return "pairs"

    def set_peer_candles(self, symbol: str, candles: list[Candle]) -> None:
        """Cache candles for a peer symbol so generate_signal can find them."""
        self._peer_candles[symbol] = candles

    def supported_segments(self) -> list[str]:
        """Return segment IDs where pairs strategy is enabled in YAML presets."""
        segments: list[str] = []
        for preset_path in sorted(_PRESETS_DIR.glob("*.yaml")):
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            strategies = data.get("strategies", {})
            pairs_cfg = strategies.get("pairs", {})
            if pairs_cfg.get("enabled", False):
                segments.append(data["segment_id"])
        return segments

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load pairs parameters from the YAML preset for the given segment."""
        try:
            preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            params: dict[str, object] = dict(data["strategies"]["pairs"]["params"])
            return params
        except (FileNotFoundError, KeyError):
            return {}

    def generate_signal(self, symbol: str, candles: list[Candle], segment_id: str) -> Signal | None:
        """Generate a pairs trading signal for symbol.

        Requires peer candles to be set via set_peer_candles() for all symbols
        configured as pairs with this symbol.

        Args:
            symbol: The primary symbol to generate a signal for.
            candles: Recent candles for symbol (must have >= 60).
            segment_id: Segment ID used to load YAML parameters.

        Returns:
            Signal if spread is beyond z_entry threshold, None otherwise.
        """
        if len(candles) < _MIN_CANDLES:
            return None

        params = self.get_parameters(segment_id)
        if not params:
            return None

        raw_pairs = cast("list[list[str]]", params.get("pairs", []))
        configured_pairs: list[list[str]] = [[str(s) for s in p] for p in raw_pairs]
        z_entry = float(cast("float", params.get("z_entry", 2.0)))
        z_exit = float(cast("float", params.get("z_exit", 0.5)))

        for pair in configured_pairs:
            if len(pair) != _PAIR_LENGTH:
                continue
            sym_a, sym_b = pair[0], pair[1]

            # Only process pairs involving this symbol
            if symbol not in (sym_a, sym_b):
                continue

            # Determine which symbol is the "other" one
            peer_sym = sym_b if symbol == sym_a else sym_a
            peer_candles = self._peer_candles.get(peer_sym)
            if peer_candles is None or len(peer_candles) < _MIN_CANDLES:
                continue

            # Use the same symbol_a / symbol_b ordering as configured
            if symbol == sym_a:
                candles_a, candles_b = candles, peer_candles
            else:
                candles_a, candles_b = peer_candles, candles

            signal = self._compute_signal(
                symbol=symbol,
                candles_a=candles_a,
                candles_b=candles_b,
                segment_id=segment_id,
                z_entry=z_entry,
                z_exit=z_exit,
            )
            if signal is not None:
                return signal

        return None

    def _compute_signal(
        self,
        symbol: str,
        candles_a: list[Candle],
        candles_b: list[Candle],
        segment_id: str,
        z_entry: float,
        z_exit: float,
    ) -> Signal | None:
        """Compute spread z-score and return signal or None."""
        n = min(len(candles_a), len(candles_b))
        sorted_a = sorted(candles_a, key=lambda c: c.timestamp)[-n:]
        sorted_b = sorted(candles_b, key=lambda c: c.timestamp)[-n:]

        log_a = np.log([float(c.close) for c in sorted_a])
        log_b = np.log([float(c.close) for c in sorted_b])

        # Cointegration gate
        _, p_value, _ = coint(log_a, log_b)
        if float(p_value) > _COINT_P_THRESHOLD:
            return None

        # OLS beta
        cov_matrix = np.cov(log_a, log_b)
        beta = float(cov_matrix[0, 1] / np.var(log_b))

        # Spread and z-score
        spread = log_a - beta * log_b
        spread_mean = float(spread.mean())
        spread_std = float(spread.std())

        if spread_std == 0.0:
            return None

        z = float((spread[-1] - spread_mean) / spread_std)

        # Entry/exit logic
        if abs(z) < z_exit:
            return None  # spread closed — no new entry

        if z < -z_entry:
            direction = SignalDirection.BUY
        elif z > z_entry:
            direction = SignalDirection.SELL
        else:
            return None  # between z_exit and z_entry — ambiguous zone

        confidence = min(1.0, abs(z) / z_entry)
        market_id = candles_a[0].market_id

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={"z_score": round(z, 4), "beta": round(beta, 4)},
            reasoning=f"pairs z={z:.2f} beta={beta:.3f}",
        )

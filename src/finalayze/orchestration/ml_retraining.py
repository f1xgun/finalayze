"""ML model retraining service (extracted from TradingLoop).

Orchestrates periodic retraining of ML ensemble models for all active segments.
For each segment: fetch candles, build training windows, train an ensemble,
validate accuracy, and hot-swap into the registry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

    from config.settings import Settings

    from finalayze.api.alerts import TelegramAlerter
    from finalayze.markets.instruments import InstrumentRegistry
    from finalayze.ml.registry import MLModelRegistry

_log = structlog.get_logger()


class MLRetrainingService:
    """Periodically retrain ML ensemble models for all active segments.

    Extracted from ``TradingLoop`` to reduce class size while preserving
    identical behavior.  Called by APScheduler in a dedicated executor.
    """

    def __init__(
        self,
        fetchers: dict[str, Any],
        registry: InstrumentRegistry,
        ml_registry: MLModelRegistry | None,
        settings: Settings,
        alerter: TelegramAlerter | None,
        collect_segments_fn: Callable[[], list[str]],
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._fetchers = fetchers
        self._registry = registry
        self._ml_registry = ml_registry
        self._settings = settings
        self._alerter = alerter
        self._collect_active_segments = collect_segments_fn
        self._now = now_fn or (lambda: datetime.now(UTC))

    # ── Public API ──────────────────────────────────────────────────────────

    def retrain_all(self) -> None:
        """Called by APScheduler. Iterates all active segments."""
        from finalayze.ml.loader import save_ensemble  # noqa: PLC0415
        from finalayze.ml.training import DEFAULT_WINDOW_SIZE, build_windows  # noqa: PLC0415

        if self._ml_registry is None:
            return

        min_samples = getattr(self._settings, "ml_min_train_samples", 252)
        model_dir = Path(getattr(self._settings, "ml_model_dir", "models/"))
        segments = self._collect_active_segments()

        for segment_id in segments:
            try:
                self._retrain_segment(
                    segment_id,
                    model_dir,
                    min_samples,
                    DEFAULT_WINDOW_SIZE,
                    build_windows,
                    save_ensemble,
                )
            except Exception:
                _log.exception("_retrain_cycle: failed for segment %s", segment_id)
                if self._alerter is not None:
                    self._alerter.on_error("MLRetrain", f"Retrain failed for {segment_id}")

    # ── Internal ────────────────────────────────────────────────────────────

    def _retrain_segment(
        self,
        segment_id: str,
        model_dir: Path,
        min_samples: int,
        window_size: int,
        build_windows_fn: object,
        save_ensemble_fn: object,
    ) -> None:
        """Retrain a single segment's ML ensemble with validation gating."""

        # Fetch candles for each instrument in this segment
        market_id = segment_id.split("_", maxsplit=1)[0]
        instruments = [
            instr
            for instr in self._registry.list_by_market(market_id)
            if getattr(instr, "segment_id", "") == segment_id
        ]

        all_features: list[dict[str, float]] = []
        all_labels: list[int] = []
        fetcher = self._fetchers.get(market_id)
        if fetcher is None:
            return

        for instrument in instruments:
            try:
                retrain_end = self._now()
                retrain_start = retrain_end - timedelta(days=500 * 2)
                candles = fetcher.fetch_candles(  # type: ignore[attr-defined]
                    symbol=instrument.symbol,
                    start=retrain_start,
                    end=retrain_end,
                )
            except Exception:
                _log.warning("_retrain: failed to fetch candles for %s", instrument.symbol)
                continue

            if len(candles) < window_size + 1:
                continue

            # Type-safe call to build_windows
            x_sym, y_sym, _ts = build_windows_fn(candles, window_size)  # type: ignore[operator]
            all_features.extend(x_sym)
            all_labels.extend(y_sym)

        if len(all_features) < min_samples:
            _log.info(
                "_retrain: only %d samples for %s (need %d) — skipping",
                len(all_features),
                segment_id,
                min_samples,
            )
            return

        # Temporal split: 70% train, gap of window_size, then validation
        n_train = int(len(all_features) * 0.7)
        gap_end = min(n_train + window_size, len(all_features))

        train_features = all_features[:n_train]
        train_labels = all_labels[:n_train]
        val_features = all_features[gap_end:]
        val_labels = all_labels[gap_end:]

        if not val_features:
            _log.info("_retrain: no validation data after gap for %s — skipping", segment_id)
            return

        # Train new ensemble
        assert self._ml_registry is not None
        ensemble = self._ml_registry.create_ensemble(segment_id)
        ensemble.fit(train_features, train_labels)

        # Validation gate: accuracy, Brier score, and log-loss (6C.7)
        from finalayze.ml.training import validate_ensemble  # noqa: PLC0415

        result = validate_ensemble(ensemble, val_features, val_labels)
        if not result.passed:
            _log.warning(
                "_retrain: validation failed for %s — acc=%.3f brier=%.3f logloss=%.3f",
                segment_id,
                result.accuracy,
                result.brier_score,
                result.log_loss_val,
            )
            return

        # Hot-swap into registry (thread-safe via lock)
        self._ml_registry.register(segment_id, ensemble)
        _log.info(
            "_retrain: registered new ensemble for %s (acc=%.3f brier=%.3f logloss=%.3f)",
            segment_id,
            result.accuracy,
            result.brier_score,
            result.log_loss_val,
        )

        # Persist to disk
        try:
            save_ensemble_fn(model_dir, segment_id, ensemble)  # type: ignore[operator]
        except Exception:
            _log.exception("_retrain: failed to save ensemble for %s", segment_id)

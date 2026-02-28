"""Tests for MLModelRegistry thread safety."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from finalayze.ml.registry import MLModelRegistry


class TestRegistryThreadSafety:
    def test_concurrent_get_register(self) -> None:
        """Concurrent get/register calls should not raise or corrupt state."""
        registry = MLModelRegistry()
        errors: list[Exception] = []
        n_iterations = 100

        def register_models() -> None:
            for i in range(n_iterations):
                try:
                    model = MagicMock()
                    model.name = f"model_{i}"
                    registry.register(f"seg_{i % 5}", model)
                except Exception as e:
                    errors.append(e)

        def read_models() -> None:
            for i in range(n_iterations):
                try:
                    registry.get(f"seg_{i % 5}")
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=register_models),
            threading.Thread(target=register_models),
            threading.Thread(target=read_models),
            threading.Thread(target=read_models),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        # At least one model should be registered
        assert any(registry.get(f"seg_{i}") is not None for i in range(5))

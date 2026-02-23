# Phase 3 Track B — ML & Strategies Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Extend the ML pipeline with an LSTM sequence model and add a pairs trading strategy powered by cointegration testing.

**Architecture:** `LSTMModel` lives in `src/finalayze/ml/models/lstm_model.py` at Layer 3 alongside the existing XGBoost and LightGBM models; it is integrated into `EnsembleModel` and the `MLModelRegistry` factory. `PairsStrategy` lives in `src/finalayze/strategies/pairs.py` at Layer 4, reads YAML presets for pair configuration, and uses `statsmodels.coint` for cointegration gating before computing spread z-scores.

**Tech Stack:** `torch>=2.5.0` (already present), `statsmodels>=0.14.0` (new dep), `numpy`, `pyyaml`, `argparse`, `yfinance`

**Worktree:** `.worktrees/phase3-ml-strategies` on branch `feature/phase3-ml-strategies`

---

## Project Conventions (read before writing any code)

- Every file starts with `"""Docstring."""\n\nfrom __future__ import annotations`
- Use `StrEnum` not `str, Enum` (ruff UP042)
- Exception names end in `Error` (ruff N818)
- No magic numbers — define named constants
- Run quality checks: `source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header`
- The project uses `uv run` for all Python commands
- Tests live in `tests/unit/` — mirror source structure

---

## Task 1 — Add statsmodels dependency

### Context

`statsmodels` is needed for the `coint()` function (Engle-Granger cointegration test) used in the pairs trading strategy (Task 5). Without it the `PairsStrategy` cannot gate signals on statistical cointegration. `torch` is already a declared dependency, so the LSTM work in Tasks 2–3 requires no new deps beyond statsmodels.

### Step 1.1 — Write the smoke test FIRST (RED)

Create `tests/unit/test_statsmodels_import.py`:

```python
"""Smoke test: statsmodels is importable and coint() is callable."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_coint_importable() -> None:
    from statsmodels.tsa.stattools import coint  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n = 100
    x = rng.standard_normal(n).cumsum()
    y = x + rng.standard_normal(n) * 0.1  # strongly cointegrated
    score, pvalue, _ = coint(x, y)
    assert isinstance(score, float)
    assert 0.0 <= pvalue <= 1.0
```

Run — expect `ModuleNotFoundError` (RED):

```bash
source ~/.zshrc && uv run pytest tests/unit/test_statsmodels_import.py -q --no-header --no-cov
```

Expected output:
```
E   ModuleNotFoundError: No module named 'statsmodels'
1 error in 0.xx seconds
```

### Step 1.2 — Add the dependency (GREEN)

Edit `/Users/f1xgun/finalayze/pyproject.toml`.

In the `[project]` `dependencies` list, add after the numpy line:

```toml
    "statsmodels>=0.14.0",
```

The relevant block after the edit:

```toml
dependencies = [
    ...
    "numpy>=1.26.0",
    "statsmodels>=0.14.0",
    ...
]
```

In the `[[tool.mypy.overrides]]` `module` list, add:

```toml
    "statsmodels.*",
```

The full overrides block after the edit:

```toml
[[tool.mypy.overrides]]
module = [
    "finnhub.*",
    "pandas",
    "pandas.*",
    "pandas_ta.*",
    "yfinance.*",
    "alpaca.*",
    "tinkoff.*",
    "t_tech.*",
    "openai.*",
    "streamlit.*",
    "celery.*",
    "apscheduler.*",
    "sklearn.*",
    "xgboost.*",
    "lightgbm.*",
    "torch.*",
    "openai.*",
    "t_tech.*",
    "statsmodels.*",
]
ignore_missing_imports = true
```

Then sync:

```bash
source ~/.zshrc && uv sync --extra dev
```

### Step 1.3 — Verify GREEN

```bash
source ~/.zshrc && uv run pytest tests/unit/test_statsmodels_import.py -q --no-header --no-cov
```

Expected output:
```
1 passed in 0.xx seconds
```

### Step 1.4 — Commit

```bash
git add pyproject.toml uv.lock tests/unit/test_statsmodels_import.py
git commit -m "$(cat <<'EOF'
feat(deps): add statsmodels for cointegration tests in pairs trading

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — LSTMModel

### Context

`LSTMModel` mirrors the existing `XGBoostModel` and `LightGBMModel` contracts (`BaseMLModel`) but uses a 2-layer LSTM internally. The key design decisions:

- **Untrained behaviour:** returns `0.5` (same as XGB/LGBM) so it is safe to include in ensembles before training.
- **Feature ordering:** features are sorted alphabetically (same convention as XGB/LGBM) to ensure dict-insertion-order independence.
- **Rolling buffer:** `predict_proba` maintains a `deque(maxlen=sequence_length)` so that single-sample inference does not require the caller to supply a sequence.
- **Save/load:** uses `torch.save` with a config dict so the architecture can be reconstructed from the file alone.

### Step 2.1 — Write tests FIRST (RED)

Create `tests/unit/test_lstm_model.py`:

```python
"""Unit tests for LSTMModel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from finalayze.core.exceptions import InsufficientDataError

# Constants — no magic numbers
N_TRAIN_SAMPLES = 60
SEQUENCE_LENGTH = 20
HIDDEN_SIZE = 32
NUM_LAYERS = 2
N_FEATURES = 6
HALF_PROB = 0.5


def _make_feature_dict(seed: int = 0) -> dict[str, float]:
    """Return a synthetic feature dict with N_FEATURES keys."""
    rng = np.random.default_rng(seed)
    return {f"feat_{i:02d}": float(rng.standard_normal()) for i in range(N_FEATURES)}


def _make_dataset(
    n: int = N_TRAIN_SAMPLES,
) -> tuple[list[dict[str, float]], list[int]]:
    rng = np.random.default_rng(1)
    X = [_make_feature_dict(i) for i in range(n)]
    y = [int(rng.integers(0, 2)) for _ in range(n)]
    return X, y


@pytest.mark.unit
class TestLSTMModelUntrained:
    def test_untrained_returns_half(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
        result = model.predict_proba(_make_feature_dict())
        assert result == pytest.approx(HALF_PROB)

    def test_untrained_is_false(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
        assert model._trained is False  # noqa: SLF001


@pytest.mark.unit
@pytest.mark.slow
class TestLSTMModelFit:
    def test_fit_trains_without_error(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="us_tech",
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        X, y = _make_dataset()
        model.fit(X, y)
        assert model._trained is True  # noqa: SLF001

    def test_fit_insufficient_data_raises(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
        X, y = _make_dataset(n=SEQUENCE_LENGTH - 1)  # one short
        with pytest.raises(InsufficientDataError):
            model.fit(X, y)

    def test_predict_after_fit_returns_float_in_range(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="us_tech",
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        X, y = _make_dataset()
        model.fit(X, y)
        result = model.predict_proba(_make_feature_dict())
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_feature_order_consistency(self) -> None:
        """Same features in different dict order produce the same prediction."""
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="us_tech",
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        X, y = _make_dataset()
        model.fit(X, y)

        features = _make_feature_dict(seed=99)
        features_reversed = dict(reversed(list(features.items())))

        # Flush buffer so both calls start from the same state
        model._feature_buffer.clear()  # noqa: SLF001
        r1 = model.predict_proba(features)
        model._feature_buffer.clear()  # noqa: SLF001
        r2 = model.predict_proba(features_reversed)
        assert r1 == pytest.approx(r2)


@pytest.mark.unit
@pytest.mark.slow
class TestLSTMModelSaveLoad:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="us_tech",
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        X, y = _make_dataset()
        model.fit(X, y)

        save_path = tmp_path / "lstm.pkl"
        model.save(save_path)
        assert save_path.exists()

        model2 = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
        model2.load(save_path)
        assert model2._trained is True  # noqa: SLF001

        features = _make_feature_dict()
        model._feature_buffer.clear()  # noqa: SLF001
        model2._feature_buffer.clear()  # noqa: SLF001
        r1 = model.predict_proba(features)
        r2 = model2.predict_proba(features)
        assert r1 == pytest.approx(r2, abs=1e-5)
```

Run — expect `ModuleNotFoundError` or `ImportError` (RED):

```bash
source ~/.zshrc && uv run pytest tests/unit/test_lstm_model.py -q --no-header --no-cov
```

Expected output:
```
E   ModuleNotFoundError: No module named 'finalayze.ml.models.lstm_model'
x errors in 0.xx seconds
```

### Step 2.2 — Implement `LSTMModel` (GREEN)

Create `/Users/f1xgun/finalayze/src/finalayze/ml/models/lstm_model.py`:

```python
"""LSTM per-segment model (Layer 3)."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from finalayze.core.exceptions import InsufficientDataError
from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5
_TRAIN_EPOCHS = 50
_LEARNING_RATE = 0.001


class _LSTMNet(nn.Module):
    """Internal PyTorch LSTM network."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self._lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self._linear = nn.Linear(hidden_size, 1)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape (batch, seq_len, input_size) → (batch, 1)."""
        lstm_out, _ = self._lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self._sigmoid(self._linear(last_hidden))


class LSTMModel(BaseMLModel):
    """2-layer LSTM classifier for directional prediction per segment."""

    segment_id: str

    def __init__(
        self,
        segment_id: str,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None:
        self.segment_id = segment_id
        self._sequence_length = sequence_length
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._n_features: int | None = None
        self._model: _LSTMNet | None = None
        self._trained: bool = False
        self._feature_buffer: deque[list[float]] = deque(maxlen=sequence_length)

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return BUY probability in [0.0, 1.0]. Returns 0.5 when untrained."""
        if not self._trained or self._model is None:
            return _UNTRAINED_PROB

        sorted_vals = [features[k] for k in sorted(features)]
        self._feature_buffer.append(sorted_vals)

        if len(self._feature_buffer) < self._sequence_length:
            # Pad with the current observation repeated until buffer is full
            while len(self._feature_buffer) < self._sequence_length:
                self._feature_buffer.appendleft(sorted_vals)

        seq = list(self._feature_buffer)
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            output: torch.Tensor = self._model(tensor)
        return float(output.squeeze())

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train the LSTM on feature dicts and binary labels.

        Args:
            X: List of feature dicts (all dicts must have identical keys).
            y: Binary labels (1=BUY, 0=SELL/HOLD), same length as X.

        Raises:
            InsufficientDataError: When len(X) < sequence_length.
        """
        if len(X) < self._sequence_length:
            msg = (
                f"Need at least {self._sequence_length} samples for LSTM training, "
                f"got {len(X)}"
            )
            raise InsufficientDataError(msg)

        feature_keys = sorted(X[0])
        n_features = len(feature_keys)
        self._n_features = n_features

        # Build tensor: shape (n_sequences, sequence_length, n_features)
        sequences: list[list[list[float]]] = []
        labels: list[float] = []
        for i in range(len(X) - self._sequence_length + 1):
            seq = [[row[k] for k in feature_keys] for row in X[i : i + self._sequence_length]]
            sequences.append(seq)
            labels.append(float(y[i + self._sequence_length - 1]))

        x_tensor = torch.tensor(sequences, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        self._model = _LSTMNet(n_features, self._hidden_size, self._num_layers)
        optimizer = torch.optim.Adam(  # type: ignore[attr-defined]
            self._model.parameters(), lr=_LEARNING_RATE
        )
        criterion = nn.BCELoss()

        self._model.train()
        for _ in range(_TRAIN_EPOCHS):
            optimizer.zero_grad()
            output = self._model(x_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        self._trained = True
        self._feature_buffer.clear()

    def save(self, path: Path) -> None:
        """Save model state dict and config to path."""
        if self._model is None:
            msg = "Cannot save an untrained LSTMModel"
            raise InsufficientDataError(msg)
        payload: dict[str, Any] = {
            "state_dict": self._model.state_dict(),
            "config": {
                "segment_id": self.segment_id,
                "sequence_length": self._sequence_length,
                "hidden_size": self._hidden_size,
                "num_layers": self._num_layers,
                "n_features": self._n_features,
            },
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        """Load model state dict and config from path."""
        payload: dict[str, Any] = torch.load(path, weights_only=False)  # type: ignore[assignment]
        cfg: dict[str, Any] = payload["config"]
        self._sequence_length = int(cfg["sequence_length"])
        self._hidden_size = int(cfg["hidden_size"])
        self._num_layers = int(cfg["num_layers"])
        self._n_features = int(cfg["n_features"])
        self._model = _LSTMNet(self._n_features, self._hidden_size, self._num_layers)
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
        self._trained = True
        self._feature_buffer = deque(maxlen=self._sequence_length)
```

### Step 2.3 — Verify GREEN

```bash
source ~/.zshrc && uv run pytest tests/unit/test_lstm_model.py -q --no-header --no-cov
```

Expected output:
```
9 passed in x.xx seconds
```

### Step 2.4 — Quality checks

```bash
source ~/.zshrc && uv run ruff check src/finalayze/ml/models/lstm_model.py && uv run ruff format --check src/finalayze/ml/models/lstm_model.py && uv run mypy src/finalayze/ml/models/lstm_model.py
```

Expected: no errors.

### Step 2.5 — Commit

```bash
git add src/finalayze/ml/models/lstm_model.py tests/unit/test_lstm_model.py
git commit -m "$(cat <<'EOF'
feat(ml): add LSTMModel with 2-layer LSTM for per-segment prediction

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — EnsembleModel update + MLModelRegistry update

### Context

The current `EnsembleModel` takes a flat `list[BaseMLModel]` and averages all of them indiscriminately. The update must:

1. Accept an optional `lstm_model: LSTMModel | None` so callers can explicitly pass or omit the LSTM.
2. Only include a model in the average if it is trained (`_trained` flag for LSTM, non-None `_model` for XGB/LGBM). This preserves the existing behaviour where untrained models contribute `0.5` — except now they are simply excluded, keeping the denominator accurate.
3. Fall back to `0.5` only when **no** trained model is available.

The `MLModelRegistry` gains a convenience factory `create_ensemble(segment_id)` that returns an `EnsembleModel` pre-populated with `XGBoostModel + LightGBMModel + LSTMModel` for a given segment.

### Step 3.1 — Write tests FIRST (RED)

Create `tests/unit/test_ensemble_with_lstm.py`:

```python
"""Unit tests for EnsembleModel with optional LSTMModel integration."""

from __future__ import annotations

import numpy as np
import pytest

# Constants
N_SAMPLES = 60
SEQUENCE_LENGTH = 20
HALF_PROB = 0.5
TOLERANCE = 1e-5


def _make_features(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {f"feat_{i:02d}": float(rng.standard_normal()) for i in range(6)}


def _make_dataset(n: int = N_SAMPLES) -> tuple[list[dict[str, float]], list[int]]:
    rng = np.random.default_rng(42)
    X = [_make_features(i) for i in range(n)]
    y = [int(rng.integers(0, 2)) for _ in range(n)]
    return X, y


@pytest.mark.unit
class TestEnsembleWithLSTM:
    def test_ensemble_no_lstm_behaves_as_before(self) -> None:
        """Passing lstm_model=None keeps existing XGB+LGBM averaging behaviour."""
        from finalayze.ml.models.ensemble import EnsembleModel
        from finalayze.ml.models.lightgbm_model import LightGBMModel
        from finalayze.ml.models.xgboost_model import XGBoostModel

        xgb = XGBoostModel(segment_id="us_tech")
        lgbm = LightGBMModel(segment_id="us_tech")
        ensemble = EnsembleModel(models=[xgb, lgbm], lstm_model=None)
        result = ensemble.predict_proba(_make_features())
        # Both untrained → average of 0.5 and 0.5 = 0.5
        assert result == pytest.approx(HALF_PROB)

    def test_ensemble_all_three_trained_averages_correctly(self) -> None:
        from finalayze.ml.models.ensemble import EnsembleModel
        from finalayze.ml.models.lightgbm_model import LightGBMModel
        from finalayze.ml.models.lstm_model import LSTMModel
        from finalayze.ml.models.xgboost_model import XGBoostModel

        X, y = _make_dataset()
        xgb = XGBoostModel(segment_id="us_tech")
        lgbm = LightGBMModel(segment_id="us_tech")
        lstm = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)

        xgb.fit(X, y)
        lgbm.fit(X, y)
        lstm.fit(X, y)

        ensemble = EnsembleModel(models=[xgb, lgbm], lstm_model=lstm)
        features = _make_features()
        result = ensemble.predict_proba(features)

        # Verify it is a proper average of the three
        p_xgb = xgb.predict_proba(features)
        p_lgbm = lgbm.predict_proba(features)
        lstm._feature_buffer.clear()  # noqa: SLF001
        p_lstm = lstm.predict_proba(features)
        expected = (p_xgb + p_lgbm + p_lstm) / 3.0
        assert result == pytest.approx(expected, abs=TOLERANCE)

    def test_ensemble_only_two_trained_uses_two(self) -> None:
        """When lstm is untrained, only XGB+LGBM contribute to average."""
        from finalayze.ml.models.ensemble import EnsembleModel
        from finalayze.ml.models.lightgbm_model import LightGBMModel
        from finalayze.ml.models.lstm_model import LSTMModel
        from finalayze.ml.models.xgboost_model import XGBoostModel

        X, y = _make_dataset()
        xgb = XGBoostModel(segment_id="us_tech")
        lgbm = LightGBMModel(segment_id="us_tech")
        lstm = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)  # NOT trained

        xgb.fit(X, y)
        lgbm.fit(X, y)

        ensemble = EnsembleModel(models=[xgb, lgbm], lstm_model=lstm)
        features = _make_features()
        result = ensemble.predict_proba(features)

        # Should be average of XGB and LGBM only
        p_xgb = xgb.predict_proba(features)
        p_lgbm = lgbm.predict_proba(features)
        expected = (p_xgb + p_lgbm) / 2.0
        assert result == pytest.approx(expected, abs=TOLERANCE)

    def test_ensemble_none_trained_returns_half(self) -> None:
        from finalayze.ml.models.ensemble import EnsembleModel

        ensemble = EnsembleModel(models=[], lstm_model=None)
        assert ensemble.predict_proba(_make_features()) == pytest.approx(HALF_PROB)


@pytest.mark.unit
class TestMLModelRegistryFactory:
    def test_create_ensemble_returns_ensemble_with_three_models(self) -> None:
        from finalayze.ml.models.ensemble import EnsembleModel
        from finalayze.ml.registry import MLModelRegistry

        registry = MLModelRegistry()
        ensemble = registry.create_ensemble("us_tech")
        assert isinstance(ensemble, EnsembleModel)
        # The ensemble should have exactly 2 base models + 1 lstm
        assert len(ensemble._models) == 2  # noqa: SLF001
        assert ensemble._lstm_model is not None  # noqa: SLF001

    def test_create_ensemble_registers_under_segment(self) -> None:
        from finalayze.ml.registry import MLModelRegistry

        registry = MLModelRegistry()
        ensemble = registry.create_ensemble("us_tech")
        registry.register("us_tech", ensemble)
        assert registry.get("us_tech") is ensemble
```

Run — expect `ImportError` / `AttributeError` (RED):

```bash
source ~/.zshrc && uv run pytest tests/unit/test_ensemble_with_lstm.py -q --no-header --no-cov
```

Expected:
```
E   TypeError: EnsembleModel.__init__() got an unexpected keyword argument 'lstm_model'
x errors in 0.xx seconds
```

### Step 3.2 — Update `EnsembleModel` (GREEN)

Replace `/Users/f1xgun/finalayze/src/finalayze/ml/models/ensemble.py` entirely:

```python
"""Ensemble model combining XGBoost + LightGBM + optional LSTM (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.ml.models.base import BaseMLModel
    from finalayze.ml.models.lstm_model import LSTMModel

_DEFAULT_PROB = 0.5


class EnsembleModel:
    """Averages probability predictions from multiple trained BaseMLModel instances.

    Only models that are trained contribute to the average.  Untrained models
    are skipped, so the denominator always reflects active models.  When no
    models are trained, returns 0.5 (neutral probability).
    """

    def __init__(
        self,
        models: list[BaseMLModel],
        lstm_model: LSTMModel | None = None,
    ) -> None:
        self._models = models
        self._lstm_model = lstm_model

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return mean BUY probability across all *trained* models.

        Falls back to 0.5 when no models are trained.
        """
        probs: list[float] = []

        for m in self._models:
            # XGBoostModel and LightGBMModel are trained when _model is not None
            if getattr(m, "_model", None) is not None:
                probs.append(m.predict_proba(features))

        if self._lstm_model is not None and getattr(self._lstm_model, "_trained", False):
            probs.append(self._lstm_model.predict_proba(features))

        if not probs:
            return _DEFAULT_PROB
        return sum(probs) / len(probs)

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train all constituent models (including LSTM if present)."""
        for model in self._models:
            model.fit(X, y)
        if self._lstm_model is not None:
            self._lstm_model.fit(X, y)
```

### Step 3.3 — Update `MLModelRegistry` (GREEN)

Replace `/Users/f1xgun/finalayze/src/finalayze/ml/registry.py` entirely:

```python
"""Per-segment ML model registry (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.ml.models.ensemble import EnsembleModel


class MLModelRegistry:
    """Maps segment IDs to trained EnsembleModel instances."""

    def __init__(self) -> None:
        self._models: dict[str, EnsembleModel] = {}

    def register(self, segment_id: str, model: EnsembleModel) -> None:
        """Register or replace a model for a segment."""
        self._models[segment_id] = model

    def get(self, segment_id: str) -> EnsembleModel | None:
        """Return the model for the segment, or None if not registered."""
        return self._models.get(segment_id)

    def create_ensemble(self, segment_id: str) -> EnsembleModel:
        """Create a new EnsembleModel with XGBoost + LightGBM + LSTM for a segment.

        The models are untrained; call ``ensemble.fit(X, y)`` or load saved
        weights via each model's ``.load()`` before prediction.
        """
        from finalayze.ml.models.ensemble import EnsembleModel
        from finalayze.ml.models.lightgbm_model import LightGBMModel
        from finalayze.ml.models.lstm_model import LSTMModel
        from finalayze.ml.models.xgboost_model import XGBoostModel

        xgb = XGBoostModel(segment_id=segment_id)
        lgbm = LightGBMModel(segment_id=segment_id)
        lstm = LSTMModel(segment_id=segment_id)
        return EnsembleModel(models=[xgb, lgbm], lstm_model=lstm)
```

### Step 3.4 — Verify GREEN (new tests + existing tests still pass)

```bash
source ~/.zshrc && uv run pytest tests/unit/test_ensemble_with_lstm.py tests/unit/test_ml_pipeline.py -q --no-header --no-cov
```

Expected output:
```
xx passed in x.xx seconds
```

Note: `test_ml_pipeline.py::TestEnsembleModel` still passes because `lstm_model` defaults to `None` and the existing `EnsembleModel(models=[xgb, lgb])` call is unchanged.

### Step 3.5 — Quality checks

```bash
source ~/.zshrc && uv run ruff check src/finalayze/ml/models/ensemble.py src/finalayze/ml/registry.py && uv run ruff format --check src/finalayze/ml/models/ensemble.py src/finalayze/ml/registry.py && uv run mypy src/finalayze/ml/models/ensemble.py src/finalayze/ml/registry.py
```

Expected: no errors.

### Step 3.6 — Commit

```bash
git add src/finalayze/ml/models/ensemble.py src/finalayze/ml/registry.py tests/unit/test_ensemble_with_lstm.py
git commit -m "$(cat <<'EOF'
feat(ml): integrate LSTMModel into EnsembleModel and registry

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — Model training script

### Context

`scripts/train_models.py` provides a repeatable, offline training workflow. It does **not** require a running database — it falls back to `yfinance` for candle data (same as `run_backtest.py`). The train/test split is always time-ordered (no shuffle) to avoid look-ahead bias.

Key design decisions:
- Uses `argparse` for CLI options.
- `scripts/` is exempted from `T20` (print) and `ARG` linting rules in `pyproject.toml`.
- `FileNotFoundError` from missing preset YAML is caught and the segment is skipped.
- Output files go to `models/<segment_id>/{xgb.pkl, lgbm.pkl, lstm.pkl}`.
- Accuracy is computed via `sklearn.metrics.accuracy_score`.
- The `--segment` flag is optional; when omitted, trains all segments found in `strategies/presets/*.yaml`.

### Step 4.1 — Write tests FIRST (RED)

Create `tests/unit/test_train_models_script.py`:

```python
"""Unit tests for the train_models.py training script."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from finalayze.core.schemas import Candle

# Constants
N_CANDLES = 120  # enough for 60-candle windows
WINDOW_SIZE = 60
EXPECTED_MODEL_FILES = {"xgb.pkl", "lgbm.pkl", "lstm.pkl"}


def _make_candles(n: int = N_CANDLES, symbol: str = "AAPL") -> list[Candle]:
    """Build synthetic candle list."""
    rng = np.random.default_rng(42)
    prices = 100.0 + rng.standard_normal(n).cumsum()
    base = datetime(2023, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            market_id="us",
            timeframe="1d",
            timestamp=base + timedelta(days=i),
            open=Decimal(str(round(float(prices[i]) * 0.999, 2))),
            high=Decimal(str(round(float(prices[i]) * 1.005, 2))),
            low=Decimal(str(round(float(prices[i]) * 0.995, 2))),
            close=Decimal(str(round(float(prices[i]), 2))),
            volume=int(1000 + rng.integers(0, 500)),
        )
        for i in range(n)
    ]


def _load_script_module() -> object:
    """Load scripts/train_models.py as a module without executing __main__."""
    project_root = Path(__file__).parent.parent.parent
    script_path = project_root / "scripts" / "train_models.py"
    spec = importlib.util.spec_from_file_location("train_models", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.mark.unit
@pytest.mark.slow
class TestTrainModelsScript:
    def test_script_creates_output_files(self, tmp_path: Path) -> None:
        """train_one_segment() produces xgb.pkl, lgbm.pkl, lstm.pkl."""
        mod = _load_script_module()
        candles = _make_candles()

        with patch.object(mod, "_fetch_candles", return_value=candles):  # type: ignore[union-attr]
            mod.train_one_segment(  # type: ignore[union-attr]
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
            )

        segment_dir = tmp_path / "us_tech"
        assert segment_dir.is_dir()
        created = {p.name for p in segment_dir.iterdir()}
        assert EXPECTED_MODEL_FILES.issubset(created)

    def test_script_handles_insufficient_candles_gracefully(self, tmp_path: Path) -> None:
        """train_one_segment() skips segments with too few candles without raising."""
        mod = _load_script_module()
        short_candles = _make_candles(n=30)  # too few for 60-candle windows

        with patch.object(mod, "_fetch_candles", return_value=short_candles):  # type: ignore[union-attr]
            # Should complete without raising
            mod.train_one_segment(  # type: ignore[union-attr]
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
            )

    def test_parse_args_defaults(self) -> None:
        """CLI defaults: segment=None, output_dir='models/'."""
        mod = _load_script_module()
        args = mod._parse_args([])  # type: ignore[union-attr]
        assert args.segment is None
        assert args.output_dir == "models/"

    def test_parse_args_with_segment(self) -> None:
        mod = _load_script_module()
        args = mod._parse_args(["--segment", "us_tech", "--output-dir", "/tmp/out"])  # type: ignore[union-attr]
        assert args.segment == "us_tech"
        assert args.output_dir == "/tmp/out"
```

Run — expect `ModuleNotFoundError` (RED):

```bash
source ~/.zshrc && uv run pytest tests/unit/test_train_models_script.py -q --no-header --no-cov
```

Expected:
```
E   ModuleNotFoundError: No module named 'train_models'
x errors in 0.xx seconds
```

### Step 4.2 — Implement `scripts/train_models.py` (GREEN)

Replace `/Users/f1xgun/finalayze/scripts/train_models.py` entirely (it currently does not exist; confirm with `ls /Users/f1xgun/finalayze/scripts/`):

```python
"""Train XGBoost + LightGBM + LSTM models per market segment.

Usage:
    uv run python scripts/train_models.py
    uv run python scripts/train_models.py --segment us_tech
    uv run python scripts/train_models.py --segment us_tech --output-dir models/
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure src/ is importable when run directly
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
from sklearn.metrics import accuracy_score

from finalayze.core.schemas import Candle
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.ml.features.technical import compute_features
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.lstm_model import LSTMModel
from finalayze.ml.models.xgboost_model import XGBoostModel

_WINDOW_SIZE = 60
_TRAIN_RATIO = 0.8
_LOOKBACK_DAYS = 730  # 2 years of history
_DEFAULT_OUTPUT_DIR = "models/"
_SEQUENCE_LENGTH = 20

# Map segment_id → representative symbols for yfinance fallback
_SEGMENT_SYMBOLS: dict[str, list[str]] = {
    "us_tech": ["AAPL", "MSFT", "GOOGL"],
    "us_healthcare": ["JNJ", "PFE", "UNH"],
    "us_finance": ["JPM", "BAC", "GS"],
    "us_broad": ["SPY", "QQQ", "IWM"],
    "ru_blue_chips": ["SBER.ME", "GAZP.ME", "LKOH.ME"],
    "ru_energy": ["NVTK.ME", "ROSN.ME"],
    "ru_tech": ["YNDX.ME", "OZON.ME"],
    "ru_finance": ["VTBR.ME", "MOEX.ME"],
}


def _fetch_candles(segment_id: str, symbols: list[str]) -> list[Candle]:
    """Fetch candles from yfinance for the given symbols."""
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=_LOOKBACK_DAYS)
    fetcher = YFinanceFetcher(market_id=segment_id.split("_")[0])
    candles: list[Candle] = []
    for symbol in symbols:
        try:
            fetched = fetcher.fetch_candles(symbol, start, end)
            candles.extend(fetched)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Could not fetch {symbol}: {exc}")
    return candles


def _build_dataset(
    candles: list[Candle],
) -> tuple[list[dict[str, float]], list[int]]:
    """Build (X, y) from windowed candles."""
    X: list[dict[str, float]] = []
    y: list[int] = []
    # Sort candles by timestamp
    sorted_candles = sorted(candles, key=lambda c: c.timestamp)
    for i in range(len(sorted_candles) - _WINDOW_SIZE):
        window = sorted_candles[i : i + _WINDOW_SIZE]
        try:
            features = compute_features(window)
        except Exception:  # noqa: BLE001
            continue
        next_close = float(sorted_candles[i + _WINDOW_SIZE].close)
        cur_close = float(sorted_candles[i + _WINDOW_SIZE - 1].close)
        label = 1 if next_close > cur_close else 0
        X.append(features)
        y.append(label)
    return X, y


def train_one_segment(
    segment_id: str,
    symbols: list[str],
    output_dir: Path,
) -> None:
    """Train and save models for a single segment."""
    print(f"\n[{segment_id}] Fetching candles for {symbols}...")
    candles = _fetch_candles(segment_id, symbols)

    if not candles:
        print(f"[{segment_id}] No candles — skipping.")
        return

    X, y = _build_dataset(candles)
    if len(X) < _WINDOW_SIZE:
        print(f"[{segment_id}] Only {len(X)} samples — need {_WINDOW_SIZE}+, skipping.")
        return

    split = int(len(X) * _TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_train) < _SEQUENCE_LENGTH:
        print(f"[{segment_id}] Train split too small for LSTM — skipping.")
        return

    segment_dir = output_dir / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, float] = {}

    # XGBoost
    xgb = XGBoostModel(segment_id=segment_id)
    xgb.fit(X_train, y_train)
    xgb.save(segment_dir / "xgb.pkl")  # type: ignore[attr-defined]
    if X_test:
        y_pred_xgb = [round(xgb.predict_proba(f)) for f in X_test]
        results["XGB"] = float(accuracy_score(y_test, y_pred_xgb))

    # LightGBM
    lgbm = LightGBMModel(segment_id=segment_id)
    lgbm.fit(X_train, y_train)
    lgbm.save(segment_dir / "lgbm.pkl")  # type: ignore[attr-defined]
    if X_test:
        y_pred_lgbm = [round(lgbm.predict_proba(f)) for f in X_test]
        results["LGBM"] = float(accuracy_score(y_test, y_pred_lgbm))

    # LSTM
    lstm = LSTMModel(segment_id=segment_id, sequence_length=_SEQUENCE_LENGTH)
    lstm.fit(X_train, y_train)
    lstm.save(segment_dir / "lstm.pkl")
    if X_test:
        y_pred_lstm = [round(lstm.predict_proba(f)) for f in X_test]
        results["LSTM"] = float(accuracy_score(y_test, y_pred_lstm))

    summary = " | ".join(f"{k}: {v:.2f}" for k, v in results.items())
    print(f"[{segment_id}] {summary}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost + LightGBM + LSTM models per segment"
    )
    parser.add_argument(
        "--segment",
        default=None,
        help="Segment ID to train (default: all segments)",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = _parse_args()
    output_dir = Path(args.output_dir)

    if args.segment:
        segments = {args.segment: _SEGMENT_SYMBOLS.get(args.segment, [])}
    else:
        segments = _SEGMENT_SYMBOLS

    for segment_id, symbols in segments.items():
        try:
            train_one_segment(
                segment_id=segment_id,
                symbols=symbols,
                output_dir=output_dir,
            )
        except FileNotFoundError as exc:
            print(f"[{segment_id}] FileNotFoundError — {exc}, skipping.")
        except Exception as exc:  # noqa: BLE001
            print(f"[{segment_id}] Unexpected error — {exc}, skipping.")


if __name__ == "__main__":
    main()
```

Note: `XGBoostModel` and `LightGBMModel` do not currently have a `.save()` method. Add minimal pickle-based save/load to each model as part of this task.

Add to `/Users/f1xgun/finalayze/src/finalayze/ml/models/xgboost_model.py` (after the `fit` method):

```python
    def save(self, path: Path) -> None:
        """Persist model to disk using pickle via joblib."""
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load_from(cls, path: Path) -> XGBoostModel:
        """Load a previously saved XGBoostModel."""
        import joblib

        return joblib.load(path)  # type: ignore[no-any-return]
```

Add analogous `save` / `load_from` to `/Users/f1xgun/finalayze/src/finalayze/ml/models/lightgbm_model.py`.

Both methods require `from pathlib import Path` at the top of the respective files (add to imports if not present).

### Step 4.3 — Verify GREEN

```bash
source ~/.zshrc && uv run pytest tests/unit/test_train_models_script.py -q --no-header --no-cov
```

Expected output:
```
4 passed in x.xx seconds
```

### Step 4.4 — Quality checks

```bash
source ~/.zshrc && uv run ruff check scripts/train_models.py src/finalayze/ml/models/xgboost_model.py src/finalayze/ml/models/lightgbm_model.py && uv run ruff format --check scripts/train_models.py src/finalayze/ml/models/xgboost_model.py src/finalayze/ml/models/lightgbm_model.py
```

### Step 4.5 — Commit

```bash
git add scripts/train_models.py src/finalayze/ml/models/xgboost_model.py src/finalayze/ml/models/lightgbm_model.py tests/unit/test_train_models_script.py
git commit -m "$(cat <<'EOF'
feat(scripts): add train_models.py for per-segment XGB+LGBM+LSTM training

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — PairsStrategy + YAML presets

### Context

`PairsStrategy` is a statistical arbitrage strategy that trades the mean-reversion of a spread between two cointegrated assets. Key design decisions:

- **Cointegration gate:** `statsmodels.coint(log_a, log_b)` with `p > 0.05` → skip (not cointegrated). This prevents trading spurious correlations.
- **Spread z-score:** `z = (spread[-1] - mean(spread)) / std(spread)` determines entry/exit.
- **Peer candles injection:** because `BaseStrategy.generate_signal` only provides one symbol's candles, `PairsStrategy` has a `set_peer_candles(symbol, candles)` method that must be called by the caller (e.g., `StrategyCombiner`) before `generate_signal`. The peer candle cache is held in `self._peer_candles: dict[str, list[Candle]]`.
- **Signal direction:** `z < -z_entry` → BUY (spread is abnormally negative, symbol_a is cheap relative to symbol_b), `z > z_entry` → SELL, `|z| < z_exit` → return None (spread closed, exit position externally).
- **Confidence:** `min(1.0, abs(z) / z_entry)`.
- **Minimum candles:** 60 for each symbol in the pair.

### Step 5.1 — Write tests FIRST (RED)

Create `tests/unit/test_pairs_strategy.py`:

```python
"""Unit tests for PairsStrategy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from finalayze.core.schemas import Candle, SignalDirection

# Constants — no magic numbers
MIN_CANDLES = 60
N_CANDLES = 80
Z_ENTRY = 2.0
Z_EXIT = 0.5
BASE_PRICE = 100.0
SPREAD_STD = 1.0


def _make_candles(
    n: int,
    symbol: str,
    prices: list[float],
    market_id: str = "us",
) -> list[Candle]:
    base = datetime(2023, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            market_id=market_id,
            timeframe="1d",
            timestamp=base + timedelta(days=i),
            open=Decimal(str(round(prices[i] * 0.999, 4))),
            high=Decimal(str(round(prices[i] * 1.005, 4))),
            low=Decimal(str(round(prices[i] * 0.995, 4))),
            close=Decimal(str(round(prices[i], 4))),
            volume=1000,
        )
        for i in range(n)
    ]


def _cointegrated_pair(
    n: int = N_CANDLES,
    z_score: float = 0.0,
    rng_seed: int = 42,
) -> tuple[list[Candle], list[Candle]]:
    """Build two cointegrated price series (AAPL, MSFT).

    z_score shifts the final spread observation by z_score * std to simulate
    a z-score of approximately z_score.
    """
    rng = np.random.default_rng(rng_seed)
    common = rng.standard_normal(n).cumsum() + BASE_PRICE
    noise_a = rng.standard_normal(n) * 0.05
    noise_b = rng.standard_normal(n) * 0.05

    prices_a = common + noise_a
    prices_b = common * 0.5 + noise_b  # beta ≈ 0.5

    if z_score != 0.0:
        import numpy as np2

        log_a = np2.log(prices_a)
        log_b = np2.log(prices_b)
        beta = float(np2.cov(log_a, log_b)[0, 1] / np2.var(log_b))
        spread = log_a - beta * log_b
        target_shift = z_score * float(spread.std())
        prices_a[-1] = float(np2.exp(log_a[-1] + target_shift))

    candles_a = _make_candles(n, "AAPL", prices_a.tolist())
    candles_b = _make_candles(n, "MSFT", prices_b.tolist())
    return candles_a, candles_b


def _non_cointegrated_pair(n: int = N_CANDLES) -> tuple[list[Candle], list[Candle]]:
    """Two independent random walks — not cointegrated."""
    rng = np.random.default_rng(7)
    prices_a = (BASE_PRICE + rng.standard_normal(n).cumsum()).tolist()
    rng2 = np.random.default_rng(99)
    prices_b = (BASE_PRICE * 2 + rng2.standard_normal(n).cumsum()).tolist()
    return (
        _make_candles(n, "AAPL", prices_a),
        _make_candles(n, "MSFT", prices_b),
    )


@pytest.fixture
def pairs_strategy() -> object:
    from finalayze.strategies.pairs import PairsStrategy

    return PairsStrategy()


@pytest.mark.unit
class TestPairsStrategyName:
    def test_name(self, pairs_strategy: object) -> None:
        assert pairs_strategy.name == "pairs"  # type: ignore[union-attr]


@pytest.mark.unit
class TestPairsStrategyInsufficientCandles:
    def test_returns_none_when_too_few_candles(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(n=30)
        strategy.set_peer_candles("MSFT", candles_b)
        result = strategy.generate_signal("AAPL", candles_a, "us_tech")
        assert result is None


@pytest.mark.unit
class TestPairsStrategyNonCointegrated:
    def test_non_cointegrated_returns_none(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _non_cointegrated_pair()
        strategy.set_peer_candles("MSFT", candles_b)
        result = strategy.generate_signal("AAPL", candles_a, "us_tech")
        # p-value > 0.05 for random walks → should return None
        # Note: may occasionally pass if the random walk happens to pass the test;
        # seed 7/99 are chosen to reliably fail cointegration
        assert result is None


@pytest.mark.unit
class TestPairsStrategySignals:
    def test_z_below_negative_entry_returns_buy(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=-3.0)
        strategy.set_peer_candles("MSFT", candles_b)
        signal = strategy.generate_signal("AAPL", candles_a, "us_tech")
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_z_above_positive_entry_returns_sell(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=3.0)
        strategy.set_peer_candles("MSFT", candles_b)
        signal = strategy.generate_signal("AAPL", candles_a, "us_tech")
        assert signal is not None
        assert signal.direction == SignalDirection.SELL

    def test_z_within_exit_band_returns_none(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=0.0)  # z near 0
        strategy.set_peer_candles("MSFT", candles_b)
        # For a cointegrated pair with z≈0, |z| < z_exit → None
        signal = strategy.generate_signal("AAPL", candles_a, "us_tech")
        # z=0 → |0| < 0.5 → return None
        assert signal is None


@pytest.mark.unit
class TestPairsStrategyConfidence:
    def test_confidence_bounded(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=-4.0)
        strategy.set_peer_candles("MSFT", candles_b)
        signal = strategy.generate_signal("AAPL", candles_a, "us_tech")
        if signal is not None:
            assert 0.0 <= signal.confidence <= 1.0

    def test_reasoning_contains_z_and_beta(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=-3.0)
        strategy.set_peer_candles("MSFT", candles_b)
        signal = strategy.generate_signal("AAPL", candles_a, "us_tech")
        if signal is not None:
            assert "z=" in signal.reasoning
            assert "beta=" in signal.reasoning


@pytest.mark.unit
class TestPairsStrategySupportedSegments:
    def test_supported_segments_returns_list(self, pairs_strategy: object) -> None:
        segments = pairs_strategy.supported_segments()  # type: ignore[union-attr]
        assert isinstance(segments, list)
        # us_tech and ru_blue_chips should be in list after YAML update in step 5.3
        assert "us_tech" in segments
        assert "ru_blue_chips" in segments

    def test_get_parameters_us_tech(self, pairs_strategy: object) -> None:
        params = pairs_strategy.get_parameters("us_tech")  # type: ignore[union-attr]
        assert "pairs" in params
        assert "z_entry" in params
        assert "z_exit" in params
```

Run — expect `ModuleNotFoundError` (RED):

```bash
source ~/.zshrc && uv run pytest tests/unit/test_pairs_strategy.py -q --no-header --no-cov
```

Expected:
```
E   ModuleNotFoundError: No module named 'finalayze.strategies.pairs'
x errors in 0.xx seconds
```

### Step 5.2 — Update YAML presets (before implementing, so strategy can discover segments)

Edit `/Users/f1xgun/finalayze/src/finalayze/strategies/presets/us_tech.yaml` — append:

```yaml
  pairs:
    enabled: true
    weight: 0.15
    params:
      pairs: [[AAPL, MSFT]]
      z_entry: 2.0
      z_exit: 0.5
      min_confidence: 0.6
```

The full file after edit:

```yaml
segment_id: us_tech
strategies:
  momentum:
    enabled: true
    weight: 0.4
    params:
      rsi_period: 14
      rsi_oversold: 30
      rsi_overbought: 70
      macd_fast: 12
      macd_slow: 26
      min_confidence: 0.6
  mean_reversion:
    enabled: true
    weight: 0.2
    params:
      bb_period: 20
      bb_std_dev: 2.0
      min_confidence: 0.65
  event_driven:
    enabled: true
    weight: 0.4
    params:
      min_sentiment: 0.7
      event_types: [earnings, fda, product_launch]
  pairs:
    enabled: true
    weight: 0.15
    params:
      pairs: [[AAPL, MSFT]]
      z_entry: 2.0
      z_exit: 0.5
      min_confidence: 0.6
```

Edit `/Users/f1xgun/finalayze/src/finalayze/strategies/presets/ru_blue_chips.yaml` — append:

```yaml
  pairs:
    enabled: true
    weight: 0.2
    params:
      pairs: [[SBER, VTBR], [GAZP, LKOH]]
      z_entry: 2.0
      z_exit: 0.5
      min_confidence: 0.6
```

The full file after edit:

```yaml
segment_id: ru_blue_chips
strategies:
  momentum:
    enabled: true
    weight: 0.3
    params:
      rsi_period: 14
      rsi_oversold: 25
      rsi_overbought: 75
      macd_fast: 12
      macd_slow: 26
      min_confidence: 0.65
  event_driven:
    enabled: true
    weight: 0.5
    params:
      min_sentiment: 0.6
      event_types: [geopolitical, sanctions, cbr_rate, commodity_price, earnings]
  mean_reversion:
    enabled: true
    weight: 0.2
    params:
      bb_period: 20
      bb_std_dev: 2.5
      min_confidence: 0.7
  pairs:
    enabled: true
    weight: 0.2
    params:
      pairs: [[SBER, VTBR], [GAZP, LKOH]]
      z_entry: 2.0
      z_exit: 0.5
      min_confidence: 0.6
```

### Step 5.3 — Implement `PairsStrategy` (GREEN)

Create `/Users/f1xgun/finalayze/src/finalayze/strategies/pairs.py`:

```python
"""Pairs trading strategy using cointegration-based spread z-scores (Layer 4)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from statsmodels.tsa.stattools import coint

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

if TYPE_CHECKING:
    pass

_PRESETS_DIR = Path(__file__).parent / "presets"
_MIN_CANDLES = 60
_COINT_P_THRESHOLD = 0.05


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

    def generate_signal(
        self, symbol: str, candles: list[Candle], segment_id: str
    ) -> Signal | None:
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

        configured_pairs: list[list[str]] = [
            list(p) for p in params.get("pairs", [])  # type: ignore[union-attr]
        ]
        z_entry = float(params.get("z_entry", 2.0))  # type: ignore[arg-type]
        z_exit = float(params.get("z_exit", 0.5))  # type: ignore[arg-type]

        for pair in configured_pairs:
            if len(pair) != 2:  # noqa: PLR2004
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
                sym_a=sym_a,
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
        sym_a: str,
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
```

### Step 5.4 — Verify GREEN

```bash
source ~/.zshrc && uv run pytest tests/unit/test_pairs_strategy.py -q --no-header --no-cov
```

Expected output:
```
xx passed in x.xx seconds
```

### Step 5.5 — Quality checks

```bash
source ~/.zshrc && uv run ruff check src/finalayze/strategies/pairs.py && uv run ruff format --check src/finalayze/strategies/pairs.py && uv run mypy src/finalayze/strategies/pairs.py
```

Expected: no errors.

### Step 5.6 — Commit

```bash
git add src/finalayze/strategies/pairs.py src/finalayze/strategies/presets/us_tech.yaml src/finalayze/strategies/presets/ru_blue_chips.yaml tests/unit/test_pairs_strategy.py
git commit -m "$(cat <<'EOF'
feat(strategies): add PairsStrategy with cointegration-based signal generation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

Run the full quality suite from the project root:

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header
```

Expected output (approximate):
```
All checks passed.
...
xx passed, xx warnings in x.xx seconds
```

All five tasks should be green, all pre-existing tests should still pass, and coverage should remain at or above the 50% threshold (Phase 1 target), trending toward 80% as more new code is covered by the new tests.

---

## File Manifest

| File | Action |
|------|--------|
| `/Users/f1xgun/finalayze/pyproject.toml` | Add `statsmodels>=0.14.0` dep + mypy override |
| `/Users/f1xgun/finalayze/src/finalayze/ml/models/lstm_model.py` | Create |
| `/Users/f1xgun/finalayze/src/finalayze/ml/models/ensemble.py` | Modify — add `lstm_model` param + trained-only averaging |
| `/Users/f1xgun/finalayze/src/finalayze/ml/models/xgboost_model.py` | Modify — add `save()` / `load_from()` |
| `/Users/f1xgun/finalayze/src/finalayze/ml/models/lightgbm_model.py` | Modify — add `save()` / `load_from()` |
| `/Users/f1xgun/finalayze/src/finalayze/ml/registry.py` | Modify — add `create_ensemble()` factory |
| `/Users/f1xgun/finalayze/scripts/train_models.py` | Create |
| `/Users/f1xgun/finalayze/src/finalayze/strategies/pairs.py` | Create |
| `/Users/f1xgun/finalayze/src/finalayze/strategies/presets/us_tech.yaml` | Modify — add pairs section |
| `/Users/f1xgun/finalayze/src/finalayze/strategies/presets/ru_blue_chips.yaml` | Modify — add pairs section |
| `/Users/f1xgun/finalayze/tests/unit/test_statsmodels_import.py` | Create |
| `/Users/f1xgun/finalayze/tests/unit/test_lstm_model.py` | Create |
| `/Users/f1xgun/finalayze/tests/unit/test_ensemble_with_lstm.py` | Create |
| `/Users/f1xgun/finalayze/tests/unit/test_train_models_script.py` | Create |
| `/Users/f1xgun/finalayze/tests/unit/test_pairs_strategy.py` | Create |

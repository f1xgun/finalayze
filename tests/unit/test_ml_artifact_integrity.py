"""ML artifact-integrity regression tests (audit 2026-06-28, HIGH).

calibrator.pkl and meta_learner.pkl were deserialized with bare ``joblib.load``,
bypassing the key-gated HMAC ``verify_model`` check the boosting models enforce.
joblib/pickle executes arbitrary code on load, so every .pkl must route through
``_verified_joblib_load``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from finalayze.core.exceptions import ModelIntegrityError
from finalayze.ml import loader as loader_mod
from finalayze.ml.models.ensemble import EnsembleModel


def test_verified_load_verifies_before_load_when_key_set(monkeypatch) -> None:
    sentinel = object()
    verify = MagicMock(name="verify_model")
    monkeypatch.setattr(loader_mod, "_get_hmac_key", lambda: "secret")
    monkeypatch.setattr("finalayze.ml.integrity.verify_model", verify)
    monkeypatch.setattr("joblib.load", lambda _p: sentinel)

    result = loader_mod._verified_joblib_load(Path("/x/calibrator.pkl"))

    assert result is sentinel
    verify.assert_called_once_with(Path("/x/calibrator.pkl"), b"secret")


def test_verified_load_skips_verify_without_key(monkeypatch) -> None:
    sentinel = object()
    verify = MagicMock(name="verify_model")
    monkeypatch.setattr(loader_mod, "_get_hmac_key", lambda: "")
    monkeypatch.setattr("finalayze.ml.integrity.verify_model", verify)
    monkeypatch.setattr("joblib.load", lambda _p: sentinel)

    result = loader_mod._verified_joblib_load(Path("/x/meta_learner.pkl"))

    assert result is sentinel
    verify.assert_not_called()  # back-compat: no key -> load unverified


def test_verified_load_raises_and_does_not_load_on_bad_digest(monkeypatch) -> None:
    loaded = MagicMock(name="joblib.load")

    def _bad(_path, _key) -> None:
        raise ModelIntegrityError("digest mismatch")

    monkeypatch.setattr(loader_mod, "_get_hmac_key", lambda: "secret")
    monkeypatch.setattr("finalayze.ml.integrity.verify_model", _bad)
    monkeypatch.setattr("joblib.load", loaded)

    with pytest.raises(ModelIntegrityError):
        loader_mod._verified_joblib_load(Path("/x/calibrator.pkl"))

    loaded.assert_not_called()  # never deserialize a tampered artifact


def test_load_meta_learner_routes_through_verified_load(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(loader_mod, "_verified_joblib_load", lambda _p: sentinel)

    ensemble = object.__new__(EnsembleModel)  # bypass heavy __init__
    ensemble.load_meta_learner(Path("/x/meta_learner.pkl"))

    assert ensemble._meta_learner is sentinel

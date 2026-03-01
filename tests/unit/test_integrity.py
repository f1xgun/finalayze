"""Tests for ML model HMAC integrity verification (6D.13)."""

from __future__ import annotations

from pathlib import Path

import pytest

from finalayze.core.exceptions import ModelIntegrityError
from finalayze.ml.integrity import compute_hmac, sign_model, verify_model

_TEST_KEY = b"test-hmac-key-for-unit-tests"


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    """Create a fake model file for testing."""
    p = tmp_path / "model.pkl"
    p.write_bytes(b"fake-model-bytes-1234567890")
    return p


class TestComputeHmac:
    def test_deterministic(self, model_file: Path) -> None:
        h1 = compute_hmac(model_file, _TEST_KEY)
        h2 = compute_hmac(model_file, _TEST_KEY)
        assert h1 == h2

    def test_different_key_different_digest(self, model_file: Path) -> None:
        h1 = compute_hmac(model_file, _TEST_KEY)
        h2 = compute_hmac(model_file, b"other-key")
        assert h1 != h2


class TestSignAndVerify:
    def test_round_trip(self, model_file: Path) -> None:
        """sign_model then verify_model should succeed."""
        sign_model(model_file, _TEST_KEY)
        # Should not raise
        verify_model(model_file, _TEST_KEY)

    def test_tampered_file_raises(self, model_file: Path) -> None:
        """Tampering with the model file after signing should fail verification."""
        sign_model(model_file, _TEST_KEY)
        # Tamper with file contents
        model_file.write_bytes(b"tampered-content")
        with pytest.raises(ModelIntegrityError, match="HMAC verification failed"):
            verify_model(model_file, _TEST_KEY)

    def test_missing_digest_raises(self, model_file: Path) -> None:
        """verify_model should raise when no digest file exists."""
        with pytest.raises(ModelIntegrityError, match="No HMAC digest file"):
            verify_model(model_file, _TEST_KEY)

    def test_wrong_key_raises(self, model_file: Path) -> None:
        """verify_model with wrong key should fail."""
        sign_model(model_file, _TEST_KEY)
        with pytest.raises(ModelIntegrityError, match="HMAC verification failed"):
            verify_model(model_file, b"wrong-key")

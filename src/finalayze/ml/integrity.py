"""HMAC integrity verification for serialized ML model files."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from finalayze.core.exceptions import ModelIntegrityError

_DIGEST_SUFFIX = ".sha256"


def compute_hmac(path: Path, key: bytes) -> str:
    """Compute HMAC-SHA256 of a file."""
    h = hmac.new(key, digestmod=hashlib.sha256)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sign_model(path: Path, key: bytes) -> None:
    """Write an HMAC digest file alongside the model file."""
    digest = compute_hmac(path, key)
    digest_path = Path(str(path) + _DIGEST_SUFFIX)
    digest_path.write_text(digest)


def verify_model(path: Path, key: bytes) -> None:
    """Verify model file against its HMAC digest. Raises ModelIntegrityError."""
    digest_path = Path(str(path) + _DIGEST_SUFFIX)
    if not digest_path.exists():
        msg = f"No HMAC digest file for {path}"
        raise ModelIntegrityError(msg)
    expected = digest_path.read_text().strip()
    actual = compute_hmac(path, key)
    if not hmac.compare_digest(expected, actual):
        msg = f"HMAC verification failed for {path}"
        raise ModelIntegrityError(msg)

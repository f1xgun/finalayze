"""API key authentication dependency (Layer 6).

Usage:
    router = APIRouter(dependencies=[Depends(api_key_auth)])

or the lower-level factory for cases where the key is known at module import time:
    router = APIRouter(dependencies=[Depends(require_api_key(settings.api_key))])
"""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

if TYPE_CHECKING:
    from collections.abc import Callable

_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(expected_key: str) -> Callable[..., Any]:
    """Return a FastAPI dependency that validates the X-API-Key header.

    The ``expected_key`` is captured at call time.  Prefer ``api_key_auth``
    when settings are injected via ``get_settings()`` at request time.
    """

    async def _verify(key: str | None = Security(_header_scheme)) -> None:
        if not expected_key:
            raise HTTPException(status_code=503, detail="API key not configured on server")
        if key is None:
            raise HTTPException(status_code=401, detail="X-API-Key header is required")
        if not hmac.compare_digest(key, expected_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    return _verify


async def api_key_auth(key: str | None = Security(_header_scheme)) -> None:
    """FastAPI dependency that validates X-API-Key using settings loaded at request time.

    This replaces the pattern ``Depends(require_api_key(_settings.api_key))``
    where ``_settings`` was evaluated at module import time (breaking env-var
    injection in tests and deployments).
    """
    # Import here to avoid circular imports at module level (config → finalayze)
    from config.settings import get_settings  # noqa: PLC0415

    expected = get_settings().api_key
    if not expected:
        raise HTTPException(status_code=503, detail="API key not configured on server")
    if key is None:
        raise HTTPException(status_code=401, detail="X-API-Key header is required")
    if not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")

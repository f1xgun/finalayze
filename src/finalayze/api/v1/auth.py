"""API key authentication dependency (Layer 6).

Usage:
    router = APIRouter(dependencies=[Depends(require_api_key(settings.api_key))])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

if TYPE_CHECKING:
    from collections.abc import Callable

_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(expected_key: str) -> Callable[..., Any]:
    """Return a FastAPI dependency that validates the X-API-Key header."""

    async def _verify(key: str | None = Security(_header_scheme)) -> None:
        if key is None:
            raise HTTPException(status_code=422, detail="X-API-Key header is required")
        if key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    return _verify

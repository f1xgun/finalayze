"""News endpoints (Layer 6)."""

from __future__ import annotations

from config.settings import Settings
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import require_api_key

_settings = Settings()
router = APIRouter(
    prefix="/news",
    tags=["news"],
    dependencies=[Depends(require_api_key(_settings.api_key))],
)


class ArticleEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    title: str
    source: str
    scope: str
    sentiment: float | None
    published_at: str


class NewsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    articles: list[ArticleEntry]


@router.get("", response_model=NewsResponse)
async def list_news(
    scope: str | None = None,  # noqa: ARG001
    limit: int = 20,  # noqa: ARG001
) -> NewsResponse:
    return NewsResponse(articles=[])

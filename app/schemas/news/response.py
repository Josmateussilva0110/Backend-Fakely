from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class NewsSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str | None
    url: str | None
    source: str | None
    published_at: date | None
    created_at: datetime


class NewsResponse(NewsSummaryResponse):
    text: str

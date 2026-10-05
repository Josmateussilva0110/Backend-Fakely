from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import get_settings
from app.models.news_model import SOURCE_MAX_LENGTH, Classification
from app.models.news_repository import ORDERABLE_COLUMNS

_settings = get_settings()


class NewsFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    page: int = Field(1, ge=1)
    page_size: int = Field(_settings.default_page_size, ge=1, le=_settings.max_page_size)
    classification: Classification | None = None
    source: str | None = Field(None, max_length=SOURCE_MAX_LENGTH)
    analyzed_from: datetime | None = None
    analyzed_to: datetime | None = None
    order_by: str = Field("-analyzed_at", description="Campo de ordenação; prefixo '-' para decrescente.")

    @field_validator("order_by")
    @classmethod
    def order_by_allowed(cls, value: str) -> str:
        if value.lstrip("-") not in ORDERABLE_COLUMNS:
            allowed = ", ".join(sorted(ORDERABLE_COLUMNS))
            raise ValueError(f"Ordenação inválida. Use um destes campos: {allowed}.")
        return value

    @model_validator(mode="after")
    def date_range_valid(self) -> "NewsFilters":
        if self.analyzed_from and self.analyzed_to and self.analyzed_from > self.analyzed_to:
            raise ValueError("analyzed_from deve ser anterior a analyzed_to.")
        return self

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

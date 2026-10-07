from datetime import datetime

from pydantic import Field, field_validator, model_validator

from app.models.news_model import SOURCE_MAX_LENGTH
from app.models.news_repository import ORDERABLE_COLUMNS
from app.schemas.common import PaginationParams, validate_order_by


class NewsFilters(PaginationParams):
    source: str | None = Field(None, max_length=SOURCE_MAX_LENGTH)
    created_from: datetime | None = None
    created_to: datetime | None = None
    order_by: str = Field("-created_at", description="Campo de ordenação; prefixo '-' para decrescente.")

    @field_validator("order_by")
    @classmethod
    def order_by_allowed(cls, value: str) -> str:
        return validate_order_by(value, ORDERABLE_COLUMNS)

    @model_validator(mode="after")
    def date_range_valid(self) -> "NewsFilters":
        if self.created_from and self.created_to and self.created_from > self.created_to:
            raise ValueError("created_from deve ser anterior a created_to.")
        return self

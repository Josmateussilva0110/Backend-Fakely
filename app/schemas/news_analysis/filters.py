from datetime import datetime

from pydantic import Field, field_validator, model_validator

from app.models.news_analysis_model import Classification, VerificationStatus
from app.models.news_analysis_repository import ORDERABLE_COLUMNS
from app.schemas.common import PaginationParams, validate_order_by


class NewsAnalysisFilters(PaginationParams):
    classification: Classification | None = None
    verification_status: VerificationStatus | None = None
    analyzed_from: datetime | None = None
    analyzed_to: datetime | None = None
    order_by: str = Field("-analyzed_at", description="Campo de ordenação; prefixo '-' para decrescente.")

    @field_validator("order_by")
    @classmethod
    def order_by_allowed(cls, value: str) -> str:
        return validate_order_by(value, ORDERABLE_COLUMNS)

    @model_validator(mode="after")
    def date_range_valid(self) -> "NewsAnalysisFilters":
        if self.analyzed_from and self.analyzed_to and self.analyzed_from > self.analyzed_to:
            raise ValueError("analyzed_from deve ser anterior a analyzed_to.")
        return self

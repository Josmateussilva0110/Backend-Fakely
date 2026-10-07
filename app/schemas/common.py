from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings

T = TypeVar("T")

_settings = get_settings()


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int


class PaginationParams(BaseModel):
    """Paginação comum às listagens; o teto de page_size vem de Settings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    page: int = Field(1, ge=1)
    page_size: int = Field(_settings.default_page_size, ge=1, le=_settings.max_page_size)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


def validate_order_by(value: str, allowed: dict) -> str:
    if value.lstrip("-") not in allowed:
        raise ValueError(f"Ordenação inválida. Use um destes campos: {', '.join(sorted(allowed))}.")
    return value


class ErrorItem(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    detail: str
    errors: list[ErrorItem] = []

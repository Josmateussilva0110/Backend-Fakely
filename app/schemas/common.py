from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int


class ErrorItem(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    detail: str
    errors: list[ErrorItem] = []

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, UrlConstraints, field_validator

from app.core.config import get_settings
from app.models.news_model import SOURCE_MAX_LENGTH, TITLE_MAX_LENGTH, URL_MAX_LENGTH

NewsUrl = Annotated[HttpUrl, UrlConstraints(max_length=URL_MAX_LENGTH)]


class NewsForm(BaseModel):
    """Campos e validações compartilhados pela criação da análise e pela edição da notícia."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str | None = Field(None, max_length=get_settings().max_text_length)
    title: str | None = Field(None, max_length=TITLE_MAX_LENGTH)
    url: NewsUrl | None = None
    source: str | None = Field(None, max_length=SOURCE_MAX_LENGTH)
    published_at: date | None = None

    # RN02: o texto, quando enviado, não pode estar vazio (os espaços já foram removidos)
    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("O texto da notícia não pode estar vazio.")
        return value

    @field_validator("title", "source")
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("published_at")
    @classmethod
    def published_at_not_in_future(cls, value: date | None) -> date | None:
        if value is not None and value > date.today():
            raise ValueError("A data de publicação não pode estar no futuro.")
        return value

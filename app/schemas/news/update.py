from pydantic import Field, model_validator

from app.core.config import get_settings
from app.schemas.news.form import NewsForm


class NewsUpdate(NewsForm):
    # PATCH: campos opcionais, mas as validações do form continuam valendo
    text: str | None = Field(None, max_length=get_settings().max_text_length)

    @model_validator(mode="after")
    def validate_changes(self) -> "NewsUpdate":
        if not self.model_fields_set:
            raise ValueError("Informe ao menos um campo para atualizar.")
        if "text" in self.model_fields_set and self.text is None:
            raise ValueError("O texto da notícia não pode ser removido.")
        return self

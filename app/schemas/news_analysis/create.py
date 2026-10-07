from pydantic import model_validator

from app.schemas.news.form import NewsForm


class NewsAnalysisCreate(NewsForm):
    """Entrada da verificação: texto, título e/ou link da notícia."""

    @model_validator(mode="after")
    def text_or_url_required(self) -> "NewsAnalysisCreate":
        if self.text is None and self.url is None:
            raise ValueError("Informe o texto ou a URL da notícia.")
        return self

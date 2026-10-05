import math
from dataclasses import asdict
from datetime import datetime, timezone

from app.core.exceptions import NotFoundError
from app.models.news_model import News
from app.models.news_repository import NewsRepository
from app.schemas.news.create import NewsCreate
from app.schemas.news.filters import NewsFilters
from app.schemas.news.update import NewsUpdate
from app.services.news_analysis_service import AnalysisResult, NewsAnalysisService


class NewsService:
    def __init__(self, repository: NewsRepository, analysis_service: NewsAnalysisService):
        self.repository = repository
        self.analysis_service = analysis_service

    def create(self, data: NewsCreate) -> News:
        # RN08: só chega aqui uma entrada já validada pelo schema
        news = News(
            text=data.text,
            title=data.title,
            url=str(data.url) if data.url else None,
            source=data.source,
            published_at=data.published_at,
        )
        self._apply_analysis(news)
        return self.repository.save(news)

    def list(self, filters: NewsFilters) -> tuple[list[News], int, int]:
        items, total = self.repository.find_page(
            offset=filters.offset,
            limit=filters.page_size,
            order_by=filters.order_by,
            classification=filters.classification,
            source=filters.source,
            analyzed_from=filters.analyzed_from,
            analyzed_to=filters.analyzed_to,
        )
        pages = math.ceil(total / filters.page_size) if total else 0
        return items, total, pages

    def get_by_id(self, news_id: int) -> News:
        news = self.repository.find_by_id(news_id)
        if news is None:
            raise NotFoundError("Notícia não encontrada.")
        return news

    def update(self, news_id: int, data: NewsUpdate) -> News:
        news = self.get_by_id(news_id)
        changes = data.model_dump(exclude_unset=True)
        if "url" in changes:
            changes["url"] = str(changes["url"]) if changes["url"] else None
        for field, value in changes.items():
            setattr(news, field, value)

        # Conteúdo alterado exige nova análise; o resultado nunca vem do cliente
        self._apply_analysis(news)
        news.analyzed_at = datetime.now(timezone.utc)
        return self.repository.save(news)

    def delete(self, news_id: int) -> None:
        self.repository.delete(self.get_by_id(news_id))

    def _apply_analysis(self, news: News) -> None:
        result: AnalysisResult = self.analysis_service.analyze(
            news.text, title=news.title, url=news.url, source=news.source
        )
        news.classification = result.classification
        news.confidence = result.confidence
        news.probability_fake = result.probability_fake
        news.method = result.method
        news.evidence = list(result.evidence)
        news.features = asdict(result.features)
        news.verification = asdict(result.verification) if result.verification else None

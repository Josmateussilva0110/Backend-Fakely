import math

from app.core.exceptions import NotFoundError
from app.models.news_model import News
from app.models.news_repository import NewsRepository
from app.schemas.news.filters import NewsFilters
from app.schemas.news.update import NewsUpdate


class NewsService:
    def __init__(self, repository: NewsRepository):
        self.repository = repository

    def list(self, filters: NewsFilters) -> tuple[list[News], int, int]:
        items, total = self.repository.find_page(
            offset=filters.offset,
            limit=filters.page_size,
            order_by=filters.order_by,
            source=filters.source,
            created_from=filters.created_from,
            created_to=filters.created_to,
        )
        pages = math.ceil(total / filters.page_size) if total else 0
        return items, total, pages

    def get_by_id(self, news_id: int) -> News:
        news = self.repository.find_by_id(news_id)
        if news is None:
            raise NotFoundError("Notícia não encontrada.")
        return news

    def update(self, news_id: int, data: NewsUpdate) -> News:
        # As análises já feitas são preservadas (auditoria); uma nova análise é criada sob demanda
        news = self.get_by_id(news_id)
        changes = data.model_dump(exclude_unset=True)
        if "url" in changes:
            changes["url"] = str(changes["url"]) if changes["url"] else None
        for field, value in changes.items():
            setattr(news, field, value)
        return self.repository.save(news)

    def delete(self, news_id: int) -> None:
        self.repository.delete(self.get_by_id(news_id))

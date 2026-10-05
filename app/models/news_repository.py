from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.news_model import Classification, News

# Lista branca de campos ordenáveis expostos na API
ORDERABLE_COLUMNS = {
    "analyzed_at": News.analyzed_at,
    "published_at": News.published_at,
    "confidence": News.confidence,
}


class NewsRepository:
    def __init__(self, db: Session):
        self.db = db

    def find_by_id(self, news_id: int) -> News | None:
        return self.db.get(News, news_id)

    def find_page(
        self,
        *,
        offset: int,
        limit: int,
        order_by: str,
        classification: Classification | None = None,
        source: str | None = None,
        analyzed_from: datetime | None = None,
        analyzed_to: datetime | None = None,
    ) -> tuple[list[News], int]:
        conditions = []
        if classification is not None:
            conditions.append(News.classification == classification)
        if source is not None:
            conditions.append(func.lower(News.source) == source.lower())
        if analyzed_from is not None:
            conditions.append(News.analyzed_at >= analyzed_from)
        if analyzed_to is not None:
            conditions.append(News.analyzed_at <= analyzed_to)

        total = self.db.scalar(select(func.count()).select_from(News).where(*conditions)) or 0

        column = ORDERABLE_COLUMNS[order_by.lstrip("-")]
        ordering = column.desc() if order_by.startswith("-") else column.asc()
        items = self.db.scalars(
            select(News).where(*conditions).order_by(ordering, News.id.desc()).offset(offset).limit(limit)
        ).all()
        return list(items), total

    def save(self, news: News) -> News:
        self.db.add(news)
        self.db.commit()
        self.db.refresh(news)
        return news

    def delete(self, news: News) -> None:
        self.db.delete(news)
        self.db.commit()

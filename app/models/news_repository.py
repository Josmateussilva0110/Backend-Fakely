from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.news_model import News

# Lista branca de campos ordenáveis expostos na API
ORDERABLE_COLUMNS = {
    "created_at": News.created_at,
    "published_at": News.published_at,
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
        source: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> tuple[list[News], int]:
        conditions = []
        if source is not None:
            conditions.append(func.lower(News.source) == source.lower())
        if created_from is not None:
            conditions.append(News.created_at >= created_from)
        if created_to is not None:
            conditions.append(News.created_at <= created_to)

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

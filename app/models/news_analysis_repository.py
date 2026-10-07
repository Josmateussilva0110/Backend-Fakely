from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.models.news_analysis_model import Classification, NewsAnalysis, VerificationStatus

# Lista branca de campos ordenáveis expostos na API
ORDERABLE_COLUMNS = {
    "analyzed_at": NewsAnalysis.analyzed_at,
}


class NewsAnalysisRepository:
    def __init__(self, db: Session):
        self.db = db

    def find_by_id(self, analysis_id: int) -> NewsAnalysis | None:
        return self.db.scalar(
            select(NewsAnalysis)
            .where(NewsAnalysis.id == analysis_id)
            .options(selectinload(NewsAnalysis.news), selectinload(NewsAnalysis.evidences))
        )

    def find_page(
        self,
        *,
        offset: int,
        limit: int,
        order_by: str,
        news_id: int | None = None,
        classification: Classification | None = None,
        verification_status: VerificationStatus | None = None,
        analyzed_from: datetime | None = None,
        analyzed_to: datetime | None = None,
    ) -> tuple[list[NewsAnalysis], int]:
        conditions = []
        if news_id is not None:
            conditions.append(NewsAnalysis.news_id == news_id)
        if classification is not None:
            conditions.append(NewsAnalysis.classification == classification)
        if verification_status is not None:
            conditions.append(NewsAnalysis.verification_status == verification_status)
        if analyzed_from is not None:
            conditions.append(NewsAnalysis.analyzed_at >= analyzed_from)
        if analyzed_to is not None:
            conditions.append(NewsAnalysis.analyzed_at <= analyzed_to)

        total = self.db.scalar(select(func.count()).select_from(NewsAnalysis).where(*conditions)) or 0

        column = ORDERABLE_COLUMNS[order_by.lstrip("-")]
        ordering = column.desc() if order_by.startswith("-") else column.asc()
        items = self.db.scalars(
            select(NewsAnalysis)
            .where(*conditions)
            .options(selectinload(NewsAnalysis.news))
            .order_by(ordering, NewsAnalysis.id.desc())
            .offset(offset)
            .limit(limit)
        ).all()
        return list(items), total

    def save(self, analysis: NewsAnalysis) -> NewsAnalysis:
        self.db.add(analysis)
        self.db.commit()
        return self.find_by_id(analysis.id)

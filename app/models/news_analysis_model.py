from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base
from app.models.evidence_model import Evidence
from app.models.news_model import News


class Classification(str, Enum):
    LIKELY_TRUE = "likely_true"
    LIKELY_FALSE = "likely_false"
    INCONCLUSIVE = "inconclusive"


class VerificationStatus(str, Enum):
    COMPLETE = "complete"
    # Parte das fontes não respondeu
    PARTIAL = "partial"
    # Nenhuma fonte respondeu
    FAILED = "failed"
    # Busca não executada (desativada ou texto insuficiente)
    SKIPPED = "skipped"


def _enum_column(enum: type[Enum], length: int) -> SAEnum:
    return SAEnum(enum, native_enum=False, length=length, values_callable=lambda e: [m.value for m in e])


class NewsAnalysis(Base):
    """Resultado de uma verificação: guardado à parte da notícia para auditoria e reavaliação."""

    __tablename__ = "news_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_id: Mapped[int] = mapped_column(ForeignKey("news.id", ondelete="CASCADE"), nullable=False, index=True)
    claim: Mapped[str | None] = mapped_column(Text)
    classification: Mapped[Classification] = mapped_column(
        _enum_column(Classification, 20), nullable=False, index=True
    )
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    limitations: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    style_indicators: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    sources_checked: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    sources_failed: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    verification_status: Mapped[VerificationStatus] = mapped_column(
        _enum_column(VerificationStatus, 20), nullable=False, index=True
    )
    source_assessment: Mapped[dict | None] = mapped_column(JSON)
    method: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    news: Mapped[News] = relationship(back_populates="analyses")
    evidences: Mapped[list[Evidence]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True, order_by=Evidence.position
    )

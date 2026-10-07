from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base

if TYPE_CHECKING:
    from app.models.news_analysis_model import NewsAnalysis

TITLE_MAX_LENGTH = 500
URL_MAX_LENGTH = 2048
SOURCE_MAX_LENGTH = 255


class News(Base):
    """Conteúdo enviado pelo usuário; cada verificação gera uma análise separada."""

    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String(TITLE_MAX_LENGTH))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(String(URL_MAX_LENGTH))
    source: Mapped[str | None] = mapped_column(String(SOURCE_MAX_LENGTH), index=True)
    published_at: Mapped[date | None] = mapped_column(Date, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    analyses: Mapped[list[NewsAnalysis]] = relationship(
        back_populates="news", cascade="all, delete-orphan", passive_deletes=True
    )

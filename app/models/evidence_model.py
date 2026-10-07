from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base

URL_MAX_LENGTH = 2048
NAME_MAX_LENGTH = 255
TITLE_MAX_LENGTH = 1000


class Evidence(Base):
    """Publicação consultada e sua relação com a afirmação analisada."""

    __tablename__ = "evidences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("news_analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    source_name: Mapped[str] = mapped_column(String(NAME_MAX_LENGTH), nullable=False)
    organization: Mapped[str] = mapped_column(String(NAME_MAX_LENGTH), nullable=False)
    category: Mapped[str] = mapped_column(String(20), nullable=False)
    retrieved_via: Mapped[str] = mapped_column(String(NAME_MAX_LENGTH), nullable=False)
    title: Mapped[str] = mapped_column(String(TITLE_MAX_LENGTH), nullable=False)
    url: Mapped[str] = mapped_column(String(URL_MAX_LENGTH), nullable=False)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    published_at: Mapped[date | None] = mapped_column(Date)
    accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    relevance: Mapped[float] = mapped_column(Float, nullable=False)
    stance: Mapped[str] = mapped_column(String(20), nullable=False)
    verdict: Mapped[str | None] = mapped_column(String(10))
    rating: Mapped[str | None] = mapped_column(String(NAME_MAX_LENGTH))
    independent: Mapped[bool] = mapped_column(Boolean, nullable=False)

from datetime import date, datetime, timezone
from enum import Enum

from sqlalchemy import JSON, Date, DateTime, Float, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base

TITLE_MAX_LENGTH = 500
URL_MAX_LENGTH = 2048
SOURCE_MAX_LENGTH = 255


class Classification(str, Enum):
    LIKELY_TRUE = "likely_true"
    LIKELY_FALSE = "likely_false"
    INCONCLUSIVE = "inconclusive"


class News(Base):
    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String(TITLE_MAX_LENGTH))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(String(URL_MAX_LENGTH))
    source: Mapped[str | None] = mapped_column(String(SOURCE_MAX_LENGTH), index=True)
    published_at: Mapped[date | None] = mapped_column(Date, index=True)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )
    classification: Mapped[Classification] = mapped_column(
        SAEnum(Classification, native_enum=False, length=20, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        index=True,
    )
    confidence: Mapped[float | None] = mapped_column(Float, index=True)
    probability_fake: Mapped[float | None] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(50), nullable=False)
    evidence: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    features: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    verification: Mapped[dict | None] = mapped_column(JSON)

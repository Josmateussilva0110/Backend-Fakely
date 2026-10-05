from datetime import date, datetime

from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field

from app.models.news_model import Classification

CLASSIFICATION_LABELS = {
    Classification.LIKELY_TRUE: "Possivelmente verdadeira",
    Classification.LIKELY_FALSE: "Possivelmente falsa",
    Classification.INCONCLUSIVE: "Inconclusiva",
}

DISCLAIMER = (
    "Resultado gerado automaticamente com base nos critérios do sistema; "
    "não é uma confirmação absoluta da veracidade da informação."
)


class NewsFeaturesResponse(BaseModel):
    word_count: int
    uppercase_ratio: float
    exclamation_count: int
    question_count: int
    repeated_punctuation_count: int
    sensational_terms: list[str]
    alarmist_terms: list[str]
    domain: str | None
    trusted_source: bool | None


class VerificationMatchResponse(BaseModel):
    source_name: str
    source_type: Literal["fact_checker", "news_outlet"]
    title: str
    url: str
    excerpt: str
    published_at: date | None
    similarity: float
    verdict: Literal["false", "true"] | None


class VerificationResponse(BaseModel):
    keywords: list[str]
    matches: list[VerificationMatchResponse]
    sources_checked: list[str]
    sources_failed: list[str]


class NewsSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str | None
    url: str | None
    source: str | None
    published_at: date | None
    analyzed_at: datetime
    classification: Classification
    confidence: float | None

    @computed_field
    @property
    def classification_label(self) -> str:
        return CLASSIFICATION_LABELS[self.classification]


class NewsResponse(NewsSummaryResponse):
    text: str
    probability_fake: float | None
    method: str
    evidence: list[str]
    features: NewsFeaturesResponse
    verification: VerificationResponse | None
    disclaimer: str = DISCLAIMER

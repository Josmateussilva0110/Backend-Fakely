from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, computed_field

from app.models.news_analysis_model import Classification, VerificationStatus
from app.services.evidence_service import EvidenceCategory, Stance
from app.services.source_registry_service import SourceCategory
from app.services.verification_config_service import Verdict

CLASSIFICATION_LABELS = {
    Classification.LIKELY_TRUE: "Possivelmente verdadeira",
    Classification.LIKELY_FALSE: "Possivelmente falsa",
    Classification.INCONCLUSIVE: "Inconclusiva",
}

DISCLAIMER = (
    "Resultado gerado automaticamente a partir das evidências encontradas; "
    "não é uma confirmação absoluta da veracidade da informação. Consulte as referências."
)


class NewsReferenceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str | None
    url: str | None
    source: str | None
    published_at: date | None


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    position: int
    source_name: str
    category: EvidenceCategory
    retrieved_via: str
    title: str
    url: str
    excerpt: str
    published_at: date | None
    accessed_at: datetime
    relevance: float
    stance: Stance
    verdict: Verdict | None
    independent: bool


class SourceAssessmentResponse(BaseModel):
    domain: str
    recognized: bool
    name: str | None
    category: SourceCategory | None


class NewsAnalysisSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    news: NewsReferenceResponse
    claim: str | None
    classification: Classification
    verification_status: VerificationStatus
    analyzed_at: datetime

    @computed_field
    @property
    def classification_label(self) -> str:
        return CLASSIFICATION_LABELS[self.classification]


class NewsAnalysisResponse(NewsAnalysisSummaryResponse):
    justification: str
    limitations: list[str]
    style_indicators: list[str]
    evidences: list[EvidenceResponse]
    keywords: list[str]
    sources_checked: list[str]
    sources_failed: list[str]
    source_assessment: SourceAssessmentResponse | None
    method: str
    model_version: str
    disclaimer: str = DISCLAIMER

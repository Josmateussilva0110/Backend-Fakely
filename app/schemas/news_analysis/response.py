from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from app.models.evidence_model import Evidence
from app.models.news_analysis_model import Classification, Confidence, NewsAnalysis, VerificationStatus
from app.services.evidence_service import EvidenceCategory
from app.services.source_registry_service import SourceCategory
from app.services.verification_config_service import Verdict

CLASSIFICATION_LABELS = {
    Classification.LIKELY_TRUE: "Possivelmente verdadeira",
    Classification.LIKELY_FALSE: "Possivelmente falsa",
    Classification.INCONCLUSIVE: "Inconclusiva",
}

CONFIDENCE_LABELS = {
    Confidence.HIGH: "Alta",
    Confidence.MEDIUM: "Média",
    Confidence.LOW: "Baixa",
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


class ResultSummaryResponse(BaseModel):
    classification: Classification
    label: str
    confidence: Confidence | None
    confidence_label: str | None


class ResultResponse(ResultSummaryResponse):
    justification: str
    # Fontes independentes que pesaram na decisão
    supporting_sources: int
    contradicting_sources: int


class EvidenceScoresResponse(BaseModel):
    """Notas usadas na avaliação (auditoria): palavras em comum e implicação estimada pelo modelo."""

    relevance: float
    entailment: float | None


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    position: int
    source_name: str
    category: EvidenceCategory
    title: str
    url: str
    excerpt: str
    published_at: date | None
    accessed_at: datetime
    verdict: Verdict | None
    # Falso quando é republicação ou outra publicação da mesma organização
    independent: bool
    scores: EvidenceScoresResponse

    @model_validator(mode="before")
    @classmethod
    def from_model(cls, data: Any) -> Any:
        if isinstance(data, Evidence):
            return {
                **{field: getattr(data, field) for field in cls.model_fields if field != "scores"},
                "scores": {"relevance": data.relevance, "entailment": data.entailment},
            }
        return data


class EvidenceGroupsResponse(BaseModel):
    supporting: list[EvidenceResponse]
    contradicting: list[EvidenceResponse]
    # Sobre o assunto, mas sem confirmar nem contradizer
    related: list[EvidenceResponse]


class SourceAssessmentResponse(BaseModel):
    domain: str
    recognized: bool
    name: str | None
    category: SourceCategory | None


class WarningsResponse(BaseModel):
    limitations: list[str]
    style_indicators: list[str]
    source_assessment: SourceAssessmentResponse | None


class VerificationResponse(BaseModel):
    status: VerificationStatus
    sources_checked: list[str]
    sources_failed: list[str]
    keywords: list[str]
    method: str
    model_version: str


class NewsAnalysisSummaryResponse(BaseModel):
    id: int
    analyzed_at: datetime
    news: NewsReferenceResponse
    claim: str | None
    result: ResultSummaryResponse

    @model_validator(mode="before")
    @classmethod
    def from_model(cls, data: Any) -> Any:
        if isinstance(data, NewsAnalysis):
            return {**summary_fields(data), "result": result_summary(data)}
        return data


class NewsAnalysisResponse(NewsAnalysisSummaryResponse):
    result: ResultResponse
    evidences: EvidenceGroupsResponse
    warnings: WarningsResponse
    verification: VerificationResponse
    disclaimer: str = DISCLAIMER

    @model_validator(mode="before")
    @classmethod
    def from_model(cls, data: Any) -> Any:
        if not isinstance(data, NewsAnalysis):
            return data
        by_stance = {stance: [e for e in data.evidences if e.stance == stance]
                     for stance in ("supports", "contradicts", "neutral")}
        return {
            **summary_fields(data),
            "result": {
                **result_summary(data),
                "justification": data.justification,
                "supporting_sources": sum(e.independent for e in by_stance["supports"]),
                "contradicting_sources": sum(e.independent for e in by_stance["contradicts"]),
            },
            "evidences": {
                "supporting": by_stance["supports"],
                "contradicting": by_stance["contradicts"],
                "related": by_stance["neutral"],
            },
            "warnings": {
                "limitations": data.limitations,
                "style_indicators": data.style_indicators,
                "source_assessment": data.source_assessment,
            },
            "verification": {
                "status": data.verification_status,
                "sources_checked": data.sources_checked,
                "sources_failed": data.sources_failed,
                "keywords": data.keywords,
                "method": data.method,
                "model_version": data.model_version,
            },
        }


def summary_fields(analysis: NewsAnalysis) -> dict:
    return {"id": analysis.id, "analyzed_at": analysis.analyzed_at, "news": analysis.news, "claim": analysis.claim}


def result_summary(analysis: NewsAnalysis) -> dict:
    confidence = analysis.confidence
    return {
        "classification": analysis.classification,
        "label": CLASSIFICATION_LABELS[analysis.classification],
        "confidence": confidence,
        "confidence_label": CONFIDENCE_LABELS[confidence] if confidence else None,
    }

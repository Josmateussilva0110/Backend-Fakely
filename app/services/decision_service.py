"""Regras de classificação baseadas em evidências; na dúvida, o resultado é inconclusivo."""

from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings
from app.models.news_analysis_model import Classification
from app.services.evidence_service import EvidenceItem

DecisionReason = Literal[
    "supported",
    "contradicted",
    "conflicting_evidence",
    "insufficient_evidence",
    "no_evidence",
    "verification_unavailable",
    "insufficient_text",
]


@dataclass(frozen=True)
class Decision:
    classification: Classification
    reason: DecisionReason
    supporting: tuple[EvidenceItem, ...] = ()
    contradicting: tuple[EvidenceItem, ...] = ()


class DecisionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def is_decisive(self, evidence: EvidenceItem) -> bool:
        """Relevante, de fonte cadastrada, independente e com posição definida."""
        return (
            evidence.independent
            and evidence.category != "unknown"
            and evidence.stance != "neutral"
            and evidence.relevance >= self.settings.decision_min_relevance
        )

    def decide(self, evidences: list[EvidenceItem], sources_checked: tuple[str, ...]) -> Decision:
        # Sem nenhuma fonte consultada não há como verificar; nunca se presume falsidade
        if not sources_checked:
            return Decision(Classification.INCONCLUSIVE, "verification_unavailable")

        decisive = [e for e in evidences if self.is_decisive(e)]
        supporting = tuple(e for e in decisive if e.stance == "supports")
        contradicting = tuple(e for e in decisive if e.stance == "contradicts")

        if supporting and contradicting:
            return Decision(Classification.INCONCLUSIVE, "conflicting_evidence", supporting, contradicting)
        if len(contradicting) >= self.settings.decision_min_contradicting_sources:
            return Decision(Classification.LIKELY_FALSE, "contradicted", contradicting=contradicting)
        if len(supporting) >= self.settings.decision_min_supporting_sources:
            return Decision(Classification.LIKELY_TRUE, "supported", supporting=supporting)
        # Ausência de resultados não significa que a afirmação seja falsa
        reason: DecisionReason = "insufficient_evidence" if evidences else "no_evidence"
        return Decision(Classification.INCONCLUSIVE, reason, supporting, contradicting)

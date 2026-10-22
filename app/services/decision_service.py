"""Regras de classificação baseadas em evidências; na dúvida, o resultado é inconclusivo."""

from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings
from app.models.news_analysis_model import Classification, Confidence, VerificationStatus
from app.services.evidence_service import EvidenceItem, verdict_matches_claim

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
            and self.is_strong(evidence)
        )

    def is_strong(self, evidence: EvidenceItem) -> bool:
        # Reportagem avaliada pelo modelo: a posição "sustenta" já exigiu a implicação mínima
        if evidence.verdict is None and evidence.entailment is not None:
            return evidence.entailment >= self.settings.nli_min_entailment
        if evidence.verdict is not None:
            return verdict_matches_claim(evidence.relevance, evidence.entailment, self.settings)
        return evidence.relevance >= self.settings.decision_min_relevance

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

    def confidence(self, decision: Decision, status: VerificationStatus) -> Confidence:
        """Regra visível, não probabilidade: quantas fontes independentes concordam e se todas responderam."""
        if decision.classification == Classification.INCONCLUSIVE:
            return Confidence.LOW
        is_true = decision.classification == Classification.LIKELY_TRUE
        agreeing = decision.supporting if is_true else decision.contradicting
        if len(agreeing) >= self.settings.confidence_high_min_sources and status == VerificationStatus.COMPLETE:
            return Confidence.HIGH
        return Confidence.MEDIUM

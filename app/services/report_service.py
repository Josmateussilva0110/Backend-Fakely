"""Gera a justificativa e as limitações do relatório, sempre a partir das evidências coletadas."""

from dataclasses import dataclass
from datetime import date, timedelta

from app.core.config import Settings
from app.services.decision_service import Decision
from app.services.evidence_service import EvidenceItem
from app.services.source_registry_service import CATEGORY_LABELS, SourceAssessment

MAX_CITED_EVIDENCES = 3


@dataclass(frozen=True)
class Report:
    justification: str
    limitations: tuple[str, ...]


class ReportService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build(
        self,
        decision: Decision,
        evidences: list[EvidenceItem],
        sources_checked: tuple[str, ...],
        sources_failed: tuple[str, ...],
        source_assessment: SourceAssessment | None,
        today: date,
    ) -> Report:
        return Report(
            justification=self.build_justification(decision, sources_checked),
            limitations=tuple(
                self.build_limitations(decision, evidences, sources_failed, source_assessment, today)
            ),
        )

    def build_justification(self, decision: Decision, sources_checked: tuple[str, ...]) -> str:
        supporting = _cite(decision.supporting)
        contradicting = _cite(decision.contradicting)
        match decision.reason:
            case "supported":
                return (
                    f"A afirmação é sustentada por {_count(decision.supporting)}: {supporting}. "
                    "Nenhuma fonte confiável consultada a contradiz."
                )
            case "contradicted":
                return (
                    f"A afirmação é contradita por {_count(decision.contradicting)}: {contradicting}. "
                    "Nenhuma fonte confiável consultada a sustenta."
                )
            case "conflicting_evidence":
                return (
                    f"As fontes divergem. Sustentam a afirmação: {supporting}. "
                    f"Contradizem: {contradicting}. Por isso o resultado é inconclusivo."
                )
            case "insufficient_evidence":
                return (
                    "Foram encontradas publicações relacionadas ao assunto, mas nenhuma confirma ou contradiz "
                    "a afirmação de forma direta e suficiente."
                )
            case "no_evidence":
                return (
                    f"Nenhuma publicação correspondente foi encontrada nas fontes consultadas "
                    f"({', '.join(sources_checked)}). Isso não indica que a afirmação seja falsa."
                )
            case "verification_unavailable":
                return "Não foi possível consultar as fontes externas, então a verificação não foi concluída."
            case _:
                return "O texto é curto demais para identificar uma afirmação verificável."

    def build_limitations(
        self,
        decision: Decision,
        evidences: list[EvidenceItem],
        sources_failed: tuple[str, ...],
        source_assessment: SourceAssessment | None,
        today: date,
    ) -> list[str]:
        limitations = []
        if sources_failed:
            limitations.append(
                f"Não foi possível consultar: {', '.join(sources_failed)}. "
                "O resultado pode mudar quando essas fontes estiverem disponíveis."
            )
        decisive = (*decision.supporting, *decision.contradicting)
        outdated_before = today - timedelta(days=self.settings.evidence_outdated_after_days)
        outdated = [e for e in decisive if e.published_at and e.published_at < outdated_before]
        if outdated:
            limitations.append(
                f"Evidências publicadas há mais de {self.settings.evidence_outdated_after_days} dias "
                f"podem estar desatualizadas: {_cite(outdated)}."
            )
        if any(e.published_at is None for e in decisive):
            limitations.append("A data de publicação de parte das evidências não foi identificada.")
        if evidences:
            limitations.append(
                "A relação entre as evidências e a afirmação foi estimada pelas palavras em comum, "
                "não pelo significado. Confira os links antes de tirar conclusões."
            )
        if source_assessment is not None:
            limitations.append(_describe_source(source_assessment))
        return limitations


def _describe_source(assessment: SourceAssessment) -> str:
    if assessment.recognized:
        return (
            f"A notícia foi atribuída a {assessment.name} ({CATEGORY_LABELS[assessment.category]}). "
            "Estar em um site conhecido, sozinho, não confirma a afirmação."
        )
    return (
        f"O domínio {assessment.domain} não consta no cadastro de fontes. "
        "Isso, sozinho, não torna a notícia falsa."
    )


def _cite(evidences) -> str:
    cited = []
    for evidence in list(evidences)[:MAX_CITED_EVIDENCES]:
        details = [f"\"{evidence.title}\""]
        if evidence.rating:
            details.append(f"classificada como \"{evidence.rating}\"")
        if evidence.published_at:
            details.append(evidence.published_at.strftime("%d/%m/%Y"))
        cited.append(f"{evidence.source_name} ({', '.join(details)})")
    return "; ".join(cited)


def _count(items) -> str:
    total = len(items)
    return "1 fonte independente" if total == 1 else f"{total} fontes independentes"

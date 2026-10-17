"""Transforma resultados de busca em evidências: relevância, posição, deduplicação e independência."""

import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Literal
from urllib.parse import urlparse

from app.core.config import Settings
from app.integrations.search_client import SearchResult
from app.ml.entailment_model import EntailmentScorer
from app.services.evidence_search_service import SearchOutcome
from app.services.source_registry_service import SourceRegistry
from app.services.text_processing_service import WORD_PATTERN, find_terms, normalize_text, tokenize
from app.services.verification_config_service import (
    Verdict,
    VerdictPatterns,
    VerificationConfig,
    VerificationSourceConfig,
)

Stance = Literal["supports", "contradicts", "neutral"]
EvidenceCategory = Literal["official", "scientific", "fact_checker", "news_outlet", "unknown"]

# Prefixo usado como "radical" simples (ex.: eficacia/eficaz, vacina/vacinas)
STEM_LENGTH = 5
MIN_TOKEN_LENGTH = 3
VERDICT_SCAN_LENGTH = 200
# Categorias que publicam apuração própria e podem sustentar uma afirmação
REPORTING_CATEGORIES = ("official", "scientific", "news_outlet")


@dataclass(frozen=True)
class EvidenceItem:
    position: int
    source_name: str
    organization: str
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
    # Falso quando é republicação ou outra publicação da mesma organização
    independent: bool
    # Probabilidade de a manchete implicar a afirmação (modelo NLI); None sem o modelo
    entailment: float | None = None


@dataclass(frozen=True)
class Candidate:
    provider: VerificationSourceConfig
    result: SearchResult
    relevance: float
    # A manchete cita os nomes próprios da afirmação (condição para o modelo sustentar)
    names_covered: bool


class EvidenceService:
    def __init__(
        self,
        settings: Settings,
        config: VerificationConfig,
        registry: SourceRegistry,
        stopwords: frozenset[str],
        entailment_model: EntailmentScorer | None = None,
    ):
        self.settings = settings
        self.verdict_patterns: VerdictPatterns = config.verdict_patterns
        self.denial_terms = config.denial_terms
        self.registry = registry
        self.stopwords = stopwords
        self.entailment_model = entailment_model

    def build_evidences(self, outcome: SearchOutcome, claim: str, title: str | None = None) -> list[EvidenceItem]:
        claim_text = normalize_text(f"{title or ''} {claim}")
        required = self.required_stems(f"{title or ''} {claim}", outcome.keywords)
        candidates = self._candidates(outcome, self.stems(claim_text), required)
        entailments = self.score_entailment(candidates, claim)
        evidences = [
            evidence
            for candidate, entailment in zip(candidates, entailments, strict=True)
            if (evidence := self._to_evidence(candidate, claim_text, entailment, outcome.accessed_at))
        ]
        unique = self.deduplicate(evidences)
        # Mais relevantes primeiro; com veredito explícito antes, em caso de empate
        unique.sort(key=lambda e: (e.relevance, e.verdict is not None), reverse=True)
        return self.mark_independence(unique)

    def required_stems(self, text: str, keywords: tuple[str, ...]) -> set[str]:
        """Nomes próprios da afirmação; sem nenhum (texto todo em minúsculas), as palavras-chave mais específicas.

        O modelo dá nota alta a manchetes com as mesmas entidades e outro protagonista
        ("Courtois renova com o Real Madrid" para "Neymar foi contratado pelo Real Madrid").
        """
        names = [word for word in WORD_PATTERN.findall(text) if word[0].isupper()]
        return self.stems(" ".join(names)) or self.stems(" ".join(keywords[: self.settings.verification_min_keywords]))

    def _candidates(self, outcome: SearchOutcome, claim_stems: set[str], required: set[str]) -> list[Candidate]:
        """Resultados com palavras suficientes em comum com a afirmação (filtro antes do modelo)."""
        candidates = []
        for source_results in outcome.results:
            for result in source_results.results:
                relevance, overlap = self.compute_relevance(result.title, claim_stems)
                if overlap >= self.settings.evidence_min_overlap:
                    names_covered = required <= self.stems(result.title)
                    candidates.append(Candidate(source_results.source, result, relevance, names_covered))
        return candidates

    def score_entailment(self, candidates: list[Candidate], claim: str) -> list[float | None]:
        """Implicação de cada resultado; None sem o modelo ou quando faltam os nomes da afirmação."""
        scores: list[float | None] = [None] * len(candidates)
        if self.entailment_model is None:
            return scores
        covered = [index for index, candidate in enumerate(candidates) if candidate.names_covered]
        premises = [self.premise(candidates[index].result) for index in covered]
        for index, score in zip(covered, self.entailment_model.entailment(premises, claim), strict=True):
            scores[index] = round(score, 3)
        return scores

    def premise(self, result: SearchResult) -> str:
        # O resumo completa a manchete: "Paquetá aprova empate do Flamengo com o Estudiantes"
        # só diz que foi "pela Libertadores" no resumo
        excerpt = result.excerpt[: self.settings.evidence_excerpt_max_length]
        return f"{result.title}. {excerpt}" if excerpt else result.title

    def is_relevant(self, relevance: float, entailment: float | None) -> bool:
        # O modelo recupera manchetes que dizem o mesmo com outras palavras ("anunciado" × "contratado")
        if entailment is not None and entailment >= self.settings.nli_min_entailment:
            return True
        return relevance >= self.settings.evidence_min_relevance

    def _to_evidence(
        self, candidate: Candidate, claim_text: str, entailment: float | None, accessed_at: datetime
    ) -> EvidenceItem | None:
        provider, result, relevance = candidate.provider, candidate.result, candidate.relevance
        if not self.is_relevant(relevance, entailment):
            return None

        registered = self.registry.find(result.url)
        category: EvidenceCategory = registered.category if registered else "unknown"
        verdict = self.detect_verdict(provider, result, category)
        return EvidenceItem(
            position=0,
            source_name=registered.name if registered else provider.name,
            organization=registered.key if registered else (urlparse(result.url).hostname or "").removeprefix("www."),
            category=category,
            retrieved_via=provider.name,
            title=result.title,
            url=result.url,
            excerpt=result.excerpt[: self.settings.evidence_excerpt_max_length],
            published_at=result.published_at,
            accessed_at=accessed_at,
            relevance=round(relevance, 3),
            stance=self.detect_stance(
                verdict, category, relevance, entailment, denial=self.is_denial(result.title, claim_text)
            ),
            verdict=verdict,
            independent=True,
            entailment=entailment,
        )

    def compute_relevance(self, result_title: str, claim_stems: set[str]) -> tuple[float, int]:
        """Semelhança entre o título do resultado e a afirmação, de 0 a 1.

        Usa o maior valor entre a fração do título coberta pela afirmação (boa para afirmações
        longas) e o coeficiente de Dice (não pune manchetes que trazem a afirmação inteira mais
        detalhes, como "Corinthians anuncia Clarence Seedorf como novo técnico").
        """
        title = normalize_text(result_title)
        # Remove o veredito ("é falso que...") para comparar só o assunto
        for pattern in (*self.verdict_patterns.false, *self.verdict_patterns.true):
            title = title.replace(pattern, " ")
        title_stems = self.stems(title)
        if not title_stems:
            return 0.0, 0
        overlap = len(title_stems & claim_stems)
        title_coverage = overlap / len(title_stems)
        dice = 2 * overlap / (len(title_stems) + len(claim_stems))
        return max(title_coverage, dice), overlap

    def detect_verdict(
        self, provider: VerificationSourceConfig, result: SearchResult, category: EvidenceCategory
    ) -> Verdict | None:
        if category != "fact_checker":
            return None
        content = normalize_text(f"{result.title} {result.excerpt[:VERDICT_SCAN_LENGTH]}")
        return self._match_verdict(content, self.verdict_patterns) or provider.default_verdict

    @staticmethod
    def _match_verdict(content: str, patterns: VerdictPatterns) -> Verdict | None:
        # "não é verdade" contém "é verdade": padrões de falso são verificados primeiro
        if any(re.search(rf"(?<!\w){re.escape(p)}", content) for p in patterns.false):
            return "false"
        if any(re.search(rf"(?<!\w){re.escape(p)}", content) for p in patterns.true):
            return "true"
        return None

    def is_denial(self, result_title: str, claim_text: str) -> bool:
        """Título com desmentido ("nega", "descarta") que a própria afirmação não contém."""
        found = find_terms(normalize_text(result_title), self.denial_terms)
        return any(term not in find_terms(claim_text, self.denial_terms) for term in found)

    def detect_stance(
        self,
        verdict: Verdict | None,
        category: EvidenceCategory,
        relevance: float,
        entailment: float | None = None,
        denial: bool = False,
    ) -> Stance:
        if verdict == "false":
            return "contradicts"
        if verdict == "true":
            return "supports"
        # Só reportagens de fontes com apuração própria sustentam; manchete que desmente algo, nunca
        if category not in REPORTING_CATEGORIES or denial:
            return "neutral"
        # Com o modelo, a manchete precisa implicar a afirmação; sem ele, vale a semelhança de palavras.
        # O modelo nunca gera "contradiz": contradição só vem do veredito das agências de checagem
        if entailment is not None:
            supported = entailment >= self.settings.nli_min_entailment
        else:
            supported = relevance >= self.settings.decision_min_relevance
        return "supports" if supported else "neutral"

    @staticmethod
    def deduplicate(evidences: list[EvidenceItem]) -> list[EvidenceItem]:
        """Mesma página vinda de provedores diferentes vira uma evidência só."""
        best: dict[str, EvidenceItem] = {}
        for evidence in evidences:
            key = canonical_url(evidence.url)
            current = best.get(key)
            if current is None or (evidence.verdict is not None, evidence.relevance) > (
                current.verdict is not None, current.relevance
            ):
                best[key] = evidence
        return list(best.values())

    def mark_independence(self, evidences: list[EvidenceItem]) -> list[EvidenceItem]:
        """Cada organização conta uma vez por posição (apoio/contradição); republicações não contam."""
        seen_organizations: set[tuple[str, Stance]] = set()
        seen_titles: set[frozenset[str]] = set()
        marked = []
        for position, evidence in enumerate(evidences, start=1):
            organization_key = (evidence.organization, evidence.stance)
            independent = organization_key not in seen_organizations
            seen_organizations.add(organization_key)
            # Mesmo título em outro veículo indica republicação; checagens distintas da mesma
            # alegação compartilham o texto da alegação e continuam independentes
            if evidence.category != "fact_checker":
                title_key = frozenset(self.stems(evidence.title))
                independent = independent and title_key not in seen_titles
                seen_titles.add(title_key)
            marked.append(replace(evidence, position=position, independent=independent))
        return marked

    def stems(self, text: str) -> set[str]:
        return {
            token[:STEM_LENGTH]
            for token in tokenize(normalize_text(text))
            if len(token) >= MIN_TOKEN_LENGTH and token not in self.stopwords
        }


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    return f"{host}{parsed.path.rstrip('/')}"


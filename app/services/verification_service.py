"""Pipeline: afirmação → busca → evidências → decisão → relatório (com cache)."""

from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.cache import ThreadSafeTTLCache, build_cache_key
from app.core.config import Settings
from app.models.news_analysis_model import Classification, VerificationStatus
from app.services.analysis_rules_service import AnalysisRules
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.decision_service import Decision, DecisionReason, DecisionService
from app.services.evidence_search_service import EvidenceSearchService, SearchOutcome
from app.services.evidence_service import EvidenceItem, EvidenceService
from app.services.report_service import ReportService
from app.services.source_registry_service import SourceAssessment, SourceRegistry
from app.services.text_processing_service import describe_style_signals, extract_features

METHOD_EVIDENCE_RULES = "evidence_rules"
VERIFICATION_DISABLED_NOTICE = "A busca de evidências está desativada neste ambiente."


@dataclass(frozen=True)
class VerificationReport:
    claim: str | None
    classification: Classification
    justification: str
    limitations: tuple[str, ...]
    style_indicators: tuple[str, ...]
    keywords: tuple[str, ...]
    evidences: tuple[EvidenceItem, ...]
    sources_checked: tuple[str, ...]
    sources_failed: tuple[str, ...]
    verification_status: VerificationStatus
    source_assessment: SourceAssessment | None
    method: str
    model_version: str


class VerificationService:
    def __init__(
        self,
        settings: Settings,
        rules: AnalysisRules,
        registry: SourceRegistry,
        search_service: EvidenceSearchService | None = None,
        evidence_service: EvidenceService | None = None,
    ):
        self.settings = settings
        self.rules = rules
        self.registry = registry
        self.search_service = search_service
        self.evidence_service = evidence_service
        self.claim_service = ClaimExtractionService(rules, settings.min_words, settings.claim_max_length)
        self.decision_service = DecisionService(settings)
        self.report_service = ReportService(settings)
        # Mesmo conteúdo gera a mesma análise; evita repetir buscas externas
        self.cache = ThreadSafeTTLCache(settings.analysis_cache_max_entries, settings.analysis_cache_ttl_seconds)

    @property
    def search_enabled(self) -> bool:
        return self.search_service is not None and self.evidence_service is not None

    @property
    def entailment_enabled(self) -> bool:
        return self.evidence_service is not None and self.evidence_service.entailment_model is not None

    def verify(
        self,
        text: str,
        title: str | None = None,
        url: str | None = None,
        source: str | None = None,
        refresh: bool = False,
    ) -> VerificationReport:
        cache_key = build_cache_key(text, title, url, source)
        # refresh: reanálise pedida para buscar publicações novas, ignorando o cache
        cached = None if refresh else self.cache.get(cache_key)
        if cached is not None:
            return cached

        report = self._run(text, title, url, source)
        # Não guarda resultado incompleto (fonte fora do ar), para tentar de novo depois
        if report.verification_status not in (VerificationStatus.PARTIAL, VerificationStatus.FAILED):
            self.cache.set(cache_key, report)
        return report

    def _run(self, text: str, title: str | None, url: str | None, source: str | None) -> VerificationReport:
        style_indicators = self.detect_style(text, title)
        assessment = self.registry.assess(url, source)

        def skipped(claim: str | None, reason: DecisionReason) -> VerificationReport:
            decision = Decision(Classification.INCONCLUSIVE, reason)
            return self._build(claim, decision, [], None, style_indicators, assessment, VerificationStatus.SKIPPED)

        claim = self.claim_service.extract(text, title)
        if claim is None:
            return skipped(None, "insufficient_text")
        if not self.search_enabled:
            return skipped(claim, "verification_unavailable")
        outcome = self.search_service.search(claim, title)
        # Sem palavras-chave suficientes não há busca possível
        if outcome is None:
            return skipped(claim, "insufficient_text")

        evidences = self.evidence_service.build_evidences(outcome, claim, title)
        decision = self.decision_service.decide(evidences, outcome.sources_checked)
        return self._build(claim, decision, evidences, outcome, style_indicators, assessment, search_status(outcome))

    def detect_style(self, text: str, title: str | None) -> list[str]:
        """Sinais de estilo: alertas exibidos ao usuário, sem peso na decisão."""
        features = extract_features(text, self.rules, title=title)
        return describe_style_signals(features, self.rules)

    def _build(
        self,
        claim: str | None,
        decision: Decision,
        evidences: list[EvidenceItem],
        outcome: SearchOutcome | None,
        style_indicators: list[str],
        assessment: SourceAssessment | None,
        status: VerificationStatus,
    ) -> VerificationReport:
        checked = outcome.sources_checked if outcome else ()
        failed = outcome.sources_failed if outcome else ()
        report = self.report_service.build(
            decision, evidences, checked, failed, assessment, datetime.now(UTC).date()
        )
        limitations = list(report.limitations)
        if not self.search_enabled:
            limitations.insert(0, VERIFICATION_DISABLED_NOTICE)
        return VerificationReport(
            claim=claim,
            classification=decision.classification,
            justification=report.justification,
            limitations=tuple(limitations),
            style_indicators=tuple(style_indicators),
            keywords=outcome.keywords if outcome else (),
            evidences=tuple(evidences),
            sources_checked=checked,
            sources_failed=failed,
            verification_status=status,
            source_assessment=assessment,
            method=METHOD_EVIDENCE_RULES,
            model_version=self.settings.analysis_version,
        )


def search_status(outcome: SearchOutcome) -> VerificationStatus:
    if not outcome.sources_checked:
        return VerificationStatus.FAILED
    if outcome.sources_failed:
        return VerificationStatus.PARTIAL
    return VerificationStatus.COMPLETE

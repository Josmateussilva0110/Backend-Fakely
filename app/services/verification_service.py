"""Pipeline: afirmação → busca → evidências → decisão → relatório (com cache)."""

from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.cache import ThreadSafeTTLCache, build_cache_key
from app.core.config import Settings
from app.models.news_analysis_model import Classification, VerificationStatus
from app.services.analysis_rules_service import AnalysisRules
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.decision_service import Decision, DecisionService
from app.services.evidence_search_service import EvidenceSearchService, SearchOutcome
from app.services.evidence_service import EvidenceItem, EvidenceService
from app.services.report_service import ReportService
from app.services.source_registry_service import SourceAssessment, SourceRegistry
from app.services.text_processing_service import describe_style_signals, extract_features, preprocess_text

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
        self.claim_service = ClaimExtractionService(
            rules, settings.min_words, max_candidates=1, max_length=settings.claim_max_length
        )
        self.decision_service = DecisionService(settings)
        self.report_service = ReportService(settings)
        # Mesmo conteúdo gera a mesma análise; evita repetir buscas externas
        self.cache = ThreadSafeTTLCache(settings.analysis_cache_max_entries, settings.analysis_cache_ttl_seconds)

    @property
    def search_enabled(self) -> bool:
        return self.search_service is not None and self.evidence_service is not None

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
        claim = self.claim_service.extract(text, title).main_claim

        outcome: SearchOutcome | None = None
        if claim and self.search_enabled:
            outcome = self.search_service.search(claim, title)

        if claim is None or (self.search_enabled and outcome is None):
            decision = Decision(Classification.INCONCLUSIVE, "insufficient_text")
            return self._build(claim, decision, [], outcome, style_indicators, assessment, VerificationStatus.SKIPPED)
        if outcome is None:
            decision = Decision(Classification.INCONCLUSIVE, "verification_unavailable")
            return self._build(claim, decision, [], None, style_indicators, assessment, VerificationStatus.SKIPPED)

        evidences = self.evidence_service.build_evidences(outcome, claim, title)
        decision = self.decision_service.decide(evidences, outcome.sources_checked)
        if not outcome.sources_checked:
            status = VerificationStatus.FAILED
        elif outcome.sources_failed:
            status = VerificationStatus.PARTIAL
        else:
            status = VerificationStatus.COMPLETE
        return self._build(claim, decision, evidences, outcome, style_indicators, assessment, status)

    def detect_style(self, text: str, title: str | None) -> list[str]:
        """Sinais de estilo: alertas exibidos ao usuário, sem peso na decisão."""
        features = extract_features(preprocess_text(text, self.rules.stopwords), self.rules, title=title)
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
            decision, evidences, checked, failed, assessment, datetime.now(timezone.utc).date()
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

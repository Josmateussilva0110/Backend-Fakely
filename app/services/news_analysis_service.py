import math
from dataclasses import asdict
from datetime import date

from app.core.exceptions import NotFoundError
from app.models.evidence_model import TITLE_MAX_LENGTH as EVIDENCE_TITLE_MAX_LENGTH
from app.models.evidence_model import Evidence
from app.models.news_analysis_model import NewsAnalysis
from app.models.news_analysis_repository import NewsAnalysisRepository
from app.models.news_model import TITLE_MAX_LENGTH, News
from app.models.news_repository import NewsRepository
from app.schemas.news_analysis.create import NewsAnalysisCreate
from app.schemas.news_analysis.filters import NewsAnalysisFilters
from app.services.content_extraction_service import ContentExtractionService
from app.services.evidence_service import EvidenceItem
from app.services.verification_service import VerificationReport, VerificationService


class NewsAnalysisService:
    def __init__(
        self,
        repository: NewsAnalysisRepository,
        news_repository: NewsRepository,
        verification_service: VerificationService,
        content_service: ContentExtractionService,
    ):
        self.repository = repository
        self.news_repository = news_repository
        self.verification_service = verification_service
        self.content_service = content_service

    def create(self, data: NewsAnalysisCreate) -> NewsAnalysis:
        # RN08: só chega aqui uma entrada já validada pelo schema
        news = self._build_news(data)
        return self._analyze(news)

    def reanalyze(self, news_id: int) -> NewsAnalysis:
        """Nova verificação de uma notícia existente; as análises anteriores são mantidas."""
        news = self.news_repository.find_by_id(news_id)
        if news is None:
            raise NotFoundError("Notícia não encontrada.")
        return self._analyze(news, refresh=True)

    def get_by_id(self, analysis_id: int) -> NewsAnalysis:
        analysis = self.repository.find_by_id(analysis_id)
        if analysis is None:
            raise NotFoundError("Análise não encontrada.")
        return analysis

    def list(self, filters: NewsAnalysisFilters, news_id: int | None = None) -> tuple[list[NewsAnalysis], int, int]:
        if news_id is not None and self.news_repository.find_by_id(news_id) is None:
            raise NotFoundError("Notícia não encontrada.")
        items, total = self.repository.find_page(
            offset=filters.offset,
            limit=filters.page_size,
            order_by=filters.order_by,
            news_id=news_id,
            classification=filters.classification,
            verification_status=filters.verification_status,
            analyzed_from=filters.analyzed_from,
            analyzed_to=filters.analyzed_to,
        )
        pages = math.ceil(total / filters.page_size) if total else 0
        return items, total, pages

    def _build_news(self, data: NewsAnalysisCreate) -> News:
        url = str(data.url) if data.url else None
        text, title, published_at = data.text, data.title, data.published_at
        # Só com a URL: o texto e os metadados vêm da página (com proteção contra SSRF)
        if text is None:
            content = self.content_service.extract_from_url(url)
            text = content.text
            title = title or (content.title[:TITLE_MAX_LENGTH] if content.title else None)
            if published_at is None and content.published_at and content.published_at <= date.today():
                published_at = content.published_at
        return News(text=text, title=title, url=url, source=data.source, published_at=published_at)

    def _analyze(self, news: News, refresh: bool = False) -> NewsAnalysis:
        report = self.verification_service.verify(
            news.text, title=news.title, url=news.url, source=news.source, refresh=refresh
        )
        analysis = to_analysis(report)
        analysis.news = news
        return self.repository.save(analysis)


def to_analysis(report: VerificationReport) -> NewsAnalysis:
    return NewsAnalysis(
        claim=report.claim,
        classification=report.classification,
        justification=report.justification,
        limitations=list(report.limitations),
        style_indicators=list(report.style_indicators),
        keywords=list(report.keywords),
        sources_checked=list(report.sources_checked),
        sources_failed=list(report.sources_failed),
        verification_status=report.verification_status,
        source_assessment=asdict(report.source_assessment) if report.source_assessment else None,
        method=report.method,
        model_version=report.model_version,
        evidences=[to_evidence(item) for item in report.evidences],
    )


def to_evidence(item: EvidenceItem) -> Evidence:
    return Evidence(
        position=item.position,
        source_name=item.source_name,
        organization=item.organization,
        category=item.category,
        retrieved_via=item.retrieved_via,
        title=item.title[:EVIDENCE_TITLE_MAX_LENGTH],
        url=item.url,
        excerpt=item.excerpt,
        published_at=item.published_at,
        accessed_at=item.accessed_at,
        relevance=item.relevance,
        stance=item.stance,
        verdict=item.verdict,
        rating=item.rating,
        independent=item.independent,
    )

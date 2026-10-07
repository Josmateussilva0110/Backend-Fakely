from datetime import date, datetime, timezone

from app.integrations.search_client import SearchError, SearchResult
from app.services.evidence_search_service import SearchOutcome, SourceResults
from app.services.verification_config_service import VerificationConfig


class FakeSearchClient:
    def __init__(self, results: list[SearchResult] | None = None, error: bool = False):
        self.results = results or []
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, limit: int) -> list[SearchResult]:
        self.queries.append(query)
        if self.error:
            raise SearchError("site indisponível")
        return self.results[:limit]


class FakeSearchService:
    """Substitui o EvidenceSearchService devolvendo um resultado pronto."""

    def __init__(self, outcome: SearchOutcome | None):
        self.outcome = outcome
        self.calls = 0

    def search(self, claim: str, title: str | None = None) -> SearchOutcome | None:
        self.calls += 1
        return self.outcome


def make_result(
    title: str, url: str, excerpt: str = "", published_at: date | None = date(2026, 8, 13), **extra
) -> SearchResult:
    return SearchResult(title=title, url=url, excerpt=excerpt, published_at=published_at, **extra)


def make_outcome(
    config: VerificationConfig, results: dict[str, list[SearchResult]], failed: tuple[str, ...] = ()
) -> SearchOutcome:
    sources = {source.key: source for source in config.sources}
    return SearchOutcome(
        keywords=("ivermectina", "covid"),
        results=tuple(SourceResults(sources[key], tuple(items)) for key, items in results.items()),
        sources_checked=tuple(sources[key].name for key in results),
        sources_failed=failed,
        accessed_at=datetime(2026, 10, 6, tzinfo=timezone.utc),
    )

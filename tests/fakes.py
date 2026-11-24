from datetime import UTC, date, datetime

from app.integrations.search_client import SearchError, SearchResult
from app.services.evidence_search_service import SearchOutcome, SourceResults
from app.services.verification_config_service import VerificationConfig


class FakeSearchClient:
    def __init__(self, results: list[SearchResult] | None = None, error: bool = False, supports_period: bool = True):
        self.results = results or []
        self.error = error
        self.supports_period = supports_period
        self.queries: list[str] = []
        self.periods: list[tuple[date, date] | None] = []

    def search(self, query: str, limit: int, period: tuple[date, date] | None = None) -> list[SearchResult]:
        self.queries.append(query)
        self.periods.append(period)
        if self.error:
            raise SearchError("site indisponível")
        return self.results[:limit]


class FakeSearchService:
    """Substitui o EvidenceSearchService devolvendo um resultado pronto."""

    # Sem classe gramatical, como o serviço real com o modelo desligado
    tagger = None

    def __init__(self, outcome: SearchOutcome | None):
        self.outcome = outcome
        self.calls = 0

    def search(
        self, claim: str, title: str | None = None, reference_date: date | None = None
    ) -> SearchOutcome | None:
        self.calls += 1
        return self.outcome


class FakeEntailmentModel:
    """Devolve a implicação definida para cada manchete (0 quando não definida).

    A chave pode ser a premissa ou o par (premissa, hipótese), quando a hipótese também importa.
    """

    def __init__(self, scores: dict):
        self.scores = scores
        self.hypotheses: list[str] = []

    def entailment(self, premises: list[str], hypothesis: str) -> list[float]:
        self.hypotheses.append(hypothesis)
        return [self.scores.get((premise, hypothesis), self.scores.get(premise, 0.0)) for premise in premises]


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
        accessed_at=datetime(2026, 10, 6, tzinfo=UTC),
    )


class FakeTagger:
    """Classe gramatical fixa por palavra; o resto é substantivo."""

    def __init__(self, classes: dict[str, str]):
        self.classes = classes
        self.calls = 0

    def tag(self, text: str) -> list[tuple[str, str]]:
        self.calls += 1
        return [(word, self.classes.get(word, "NOUN")) for word in text.replace(",", " ").replace(".", " ").split()]

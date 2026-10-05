from datetime import date

from app.integrations.search_client import SearchError, SearchResult


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


def make_result(title: str, url: str, excerpt: str = "") -> SearchResult:
    return SearchResult(title=title, url=url, excerpt=excerpt, published_at=date(2026, 8, 13))

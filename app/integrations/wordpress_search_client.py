"""Busca via API REST pública do WordPress (/wp-json/wp/v2/posts)."""

import json
from datetime import date, datetime

from app.integrations.search_client import HttpFetcher, SearchError, SearchResult, html_to_text, is_allowed_url


class WordpressSearchClient:
    def __init__(self, fetcher: HttpFetcher, search_url: str, allowed_domains: tuple[str, ...]):
        self.fetcher = fetcher
        self.search_url = search_url
        self.allowed_domains = allowed_domains

    def search(self, query: str, limit: int) -> list[SearchResult]:
        params = {
            "search": query,
            "per_page": limit,
            "orderby": "relevance",
            "_fields": "title,link,date,excerpt",
        }
        body = self.fetcher.get_text(self.search_url, params, self.allowed_domains)
        try:
            posts = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SearchError("Resposta do WordPress não é JSON válido") from exc
        if not isinstance(posts, list):
            raise SearchError("Formato inesperado na resposta do WordPress")

        results = []
        for post in posts[:limit]:
            result = self._parse_post(post)
            if result is not None:
                results.append(result)
        return results

    def _parse_post(self, post: dict) -> SearchResult | None:
        try:
            url = str(post["link"])
            title = html_to_text(post["title"]["rendered"])
            excerpt = html_to_text(post.get("excerpt", {}).get("rendered", ""))
        except (KeyError, TypeError):
            return None
        if not title or not is_allowed_url(url, self.allowed_domains):
            return None
        return SearchResult(title=title, url=url, excerpt=excerpt, published_at=_parse_date(post.get("date")))


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None

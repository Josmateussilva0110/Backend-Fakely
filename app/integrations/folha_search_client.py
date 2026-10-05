"""Scraping da página de busca da Folha de S.Paulo."""

import re
from datetime import date

from bs4 import BeautifulSoup

from app.integrations.search_client import HttpFetcher, SearchResult, is_allowed_url

_MONTHS = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}
# Formato usado pela Folha: "11.set.2026 às 11h28"
_DATE_PATTERN = re.compile(r"(\d{1,2})\.([a-z]{3})\.(\d{4})")


class FolhaSearchClient:
    def __init__(self, fetcher: HttpFetcher, search_url: str, allowed_domains: tuple[str, ...]):
        self.fetcher = fetcher
        self.search_url = search_url
        self.allowed_domains = allowed_domains

    def search(self, query: str, limit: int) -> list[SearchResult]:
        html = self.fetcher.get_text(self.search_url, {"q": query}, self.allowed_domains)
        return parse_folha_results(html, self.allowed_domains, limit)


def parse_folha_results(html: str, allowed_domains: tuple[str, ...], limit: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for item in soup.select("ol.c-search li.c-headline"):
        link = item.select_one(".c-headline__content a[href]")
        title = item.select_one(".c-headline__title")
        if link is None or title is None:
            continue
        url = link["href"].strip()
        if not is_allowed_url(url, allowed_domains):
            continue
        excerpt = item.select_one(".c-headline__standfirst")
        published = item.select_one("time.c-headline__dateline")
        results.append(SearchResult(
            title=" ".join(title.get_text(" ").split()),
            url=url,
            excerpt=" ".join(excerpt.get_text(" ").split()) if excerpt else "",
            published_at=_parse_date(published.get("datetime", "") if published else ""),
        ))
        if len(results) >= limit:
            break
    return results


def _parse_date(value: str) -> date | None:
    match = _DATE_PATTERN.search(value.lower())
    if not match:
        return None
    day, month, year = match.groups()
    if month not in _MONTHS:
        return None
    try:
        return date(int(year), _MONTHS[month], int(day))
    except ValueError:
        return None

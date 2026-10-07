"""Busca checagens publicadas via Google Fact Check Tools API (claims:search)."""

import json
from datetime import date, datetime

from pydantic import SecretStr

from app.integrations.search_client import HttpFetcher, SearchError, SearchResult
from app.integrations.url_safety import is_public_web_url


class GoogleFactCheckClient:
    def __init__(
        self,
        fetcher: HttpFetcher,
        search_url: str,
        allowed_domains: tuple[str, ...],
        api_key: SecretStr,
        language_code: str,
    ):
        self.fetcher = fetcher
        self.search_url = search_url
        self.allowed_domains = allowed_domains
        self.api_key = api_key
        self.language_code = language_code

    def search(self, query: str, limit: int) -> list[SearchResult]:
        params = {"query": query, "languageCode": self.language_code, "pageSize": limit}
        # Chave no cabeçalho, e não na URL, para não aparecer em logs de requisição
        headers = {"X-Goog-Api-Key": self.api_key.get_secret_value()}
        body = self.fetcher.get_text(self.search_url, params, self.allowed_domains, headers=headers)
        try:
            claims = json.loads(body).get("claims", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            raise SearchError("Resposta inválida do Google Fact Check") from exc
        if not isinstance(claims, list):
            raise SearchError("Formato inesperado na resposta do Google Fact Check")

        results = []
        for claim in claims:
            results.extend(self._parse_claim(claim))
            if len(results) >= limit:
                break
        return results[:limit]

    def _parse_claim(self, claim: dict) -> list[SearchResult]:
        try:
            claim_text = " ".join(str(claim["text"]).split())
            reviews = claim.get("claimReview", [])
        except (KeyError, TypeError, AttributeError):
            return []
        results = []
        for review in reviews if isinstance(reviews, list) else []:
            try:
                url = str(review["url"])
                rating = str(review.get("textualRating") or "").strip() or None
                publisher = review.get("publisher") or {}
            except (KeyError, TypeError, AttributeError):
                continue
            # Links apontam para os sites das agências; só aceita http(s) público
            if not claim_text or not is_public_web_url(url):
                continue
            results.append(SearchResult(
                # A alegação revisada é o que se compara com a afirmação do usuário
                title=claim_text,
                url=url,
                excerpt=" ".join(str(review.get("title") or "").split()),
                published_at=_parse_date(review.get("reviewDate")),
                rating=rating,
                publisher_name=str(publisher.get("name") or "").strip() or None,
            ))
        return results


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return None

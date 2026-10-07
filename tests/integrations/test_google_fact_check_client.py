import httpx
import pytest
from pydantic import SecretStr

from app.integrations.google_fact_check_client import GoogleFactCheckClient
from app.integrations.search_client import HttpFetcher, SearchError

SEARCH_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
DOMAINS = ("factchecktools.googleapis.com",)


def build_client(handler) -> GoogleFactCheckClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return GoogleFactCheckClient(HttpFetcher(http, 100_000), SEARCH_URL, DOMAINS, SecretStr("secret"), "pt-BR")


def test_parses_claim_reviews_and_sends_key_in_header():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Goog-Api-Key"] == "secret"
        assert "key" not in request.url.params
        assert request.url.params["languageCode"] == "pt-BR"
        return httpx.Response(200, json={"claims": [
            {"text": "Vacina X foi proibida no Brasil", "claimReview": [
                {"publisher": {"name": "Aos Fatos", "site": "aosfatos.org"},
                 "url": "https://www.aosfatos.org/noticias/vacina-x/", "title": "Vacina X não foi proibida",
                 "reviewDate": "2026-09-01T00:00:00Z", "textualRating": "Falso"},
                # Link interno é descartado
                {"publisher": {"name": "X"}, "url": "http://127.0.0.1/admin", "textualRating": "Falso"},
            ]},
            {"claimReview": []},
        ]})

    results = build_client(handler).search("vacina proibida", 5)
    assert len(results) == 1
    result = results[0]
    assert result.title == "Vacina X foi proibida no Brasil"
    assert result.rating == "Falso"
    assert result.publisher_name == "Aos Fatos"
    assert result.published_at.isoformat() == "2026-09-01"


def test_empty_response_returns_no_results():
    assert build_client(lambda request: httpx.Response(200, json={})).search("x y", 5) == []


@pytest.mark.parametrize("response", [httpx.Response(403), httpx.Response(200, text="<html>")])
def test_errors_raise_search_error(response):
    with pytest.raises(SearchError):
        build_client(lambda request: response).search("x y", 5)

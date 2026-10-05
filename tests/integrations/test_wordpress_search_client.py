import json

import httpx
import pytest

from app.integrations.search_client import HttpFetcher, SearchError
from app.integrations.wordpress_search_client import WordpressSearchClient

SEARCH_URL = "https://www.example-checker.org/wp-json/wp/v2/posts"
DOMAINS = ("example-checker.org",)


def build_client(handler, max_bytes: int = 100_000) -> WordpressSearchClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return WordpressSearchClient(HttpFetcher(http, max_bytes), SEARCH_URL, DOMAINS)


def test_parses_posts_and_cleans_html():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"] == "ivermectina covid"
        assert request.url.params["per_page"] == "5"
        return httpx.Response(200, json=[
            {"title": {"rendered": "&Eacute; falso que a <em>ivermectina</em> cure"},
             "link": "https://www.example-checker.org/post-1",
             "date": "2026-08-13T10:00:00",
             "excerpt": {"rendered": "<p>Boato &#8211; texto</p>"}},
            # Link de outro domínio é descartado
            {"title": {"rendered": "Externo"}, "link": "https://malicioso.com/x", "date": None,
             "excerpt": {"rendered": ""}},
            {"title": "formato inválido"},
        ])

    results = build_client(handler).search("ivermectina covid", 5)
    assert len(results) == 1
    assert results[0].title == "É falso que a ivermectina cure"
    assert results[0].excerpt == "Boato – texto"
    assert results[0].published_at.isoformat() == "2026-08-13"


def test_http_error_raises_search_error():
    client = build_client(lambda request: httpx.Response(503))
    with pytest.raises(SearchError):
        client.search("x y", 5)


def test_oversized_response_is_rejected():
    body = json.dumps([{"title": {"rendered": "x" * 5000}}])
    client = build_client(lambda request: httpx.Response(200, text=body), max_bytes=1000)
    with pytest.raises(SearchError):
        client.search("x y", 5)


def test_redirect_to_other_domain_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.example-checker.org":
            return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest"})
        return httpx.Response(200, json=[])

    http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    client = WordpressSearchClient(HttpFetcher(http, 100_000), SEARCH_URL, DOMAINS)
    with pytest.raises(SearchError):
        client.search("x y", 5)


def test_invalid_json_raises_search_error():
    client = build_client(lambda request: httpx.Response(200, text="<html>erro</html>"))
    with pytest.raises(SearchError):
        client.search("x y", 5)

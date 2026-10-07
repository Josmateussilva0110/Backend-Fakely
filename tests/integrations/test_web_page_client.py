import httpx
import pytest

from app.integrations.url_safety import UnsafeUrlError, ensure_public_url, is_public_web_url
from app.integrations.web_page_client import BlockedUrlError, WebPageClient, WebPageError

PUBLIC_IP = "93.184.216.34"
HTML_HEADERS = {"content-type": "text/html; charset=utf-8"}


def fake_resolver(mapping: dict[str, str]):
    def resolve(host: str) -> list[str]:
        if host in mapping:
            return [mapping[host]]
        # IP literal resolve para ele mesmo
        return [host]

    return resolve


def build_client(handler, mapping=None, max_bytes: int = 100_000) -> WebPageClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = fake_resolver(mapping or {"noticias.com": PUBLIC_IP})
    return WebPageClient(http, max_bytes=max_bytes, max_redirects=2, timeout_seconds=1, resolver=resolver)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost/", "http://10.0.0.5/", "http://169.254.169.254/latest",
    "http://[::1]/", "http://[::ffff:127.0.0.1]/", "ftp://noticias.com/", "http://noticias.com:8080/",
    "http://user:pass@noticias.com/", "http://intranet.local/",
])
def test_blocks_unsafe_urls(url):
    with pytest.raises(UnsafeUrlError):
        ensure_public_url(url, fake_resolver({"noticias.com": PUBLIC_IP}))


def test_blocks_domain_resolving_to_private_ip():
    with pytest.raises(UnsafeUrlError):
        ensure_public_url("https://interno.empresa.com/", fake_resolver({"interno.empresa.com": "192.168.0.10"}))


def test_is_public_web_url():
    assert is_public_web_url("https://www.aosfatos.org/x")
    assert not is_public_web_url("http://127.0.0.1/x")
    assert not is_public_web_url("javascript:alert(1)")


def test_fetches_html_page():
    client = build_client(lambda request: httpx.Response(200, headers=HTML_HEADERS, text="<p>ok</p>"))
    assert client.fetch("https://noticias.com/a").html == "<p>ok</p>"


def test_redirect_to_private_address_is_blocked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data"})

    with pytest.raises(BlockedUrlError):
        build_client(handler).fetch("https://noticias.com/a")


def test_follows_safe_redirect_and_limits_count():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(301, headers={"Location": "/b"})
        if request.url.path == "/b":
            return httpx.Response(200, headers=HTML_HEADERS, text="final")
        return httpx.Response(302, headers={"Location": "/loop"})

    client = build_client(handler)
    assert client.fetch("https://noticias.com/a").html == "final"
    with pytest.raises(WebPageError):
        client.fetch("https://noticias.com/loop")


@pytest.mark.parametrize("response", [
    httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"),
    httpx.Response(404, headers=HTML_HEADERS, text="não encontrado"),
    httpx.Response(200, headers=HTML_HEADERS, text="x" * 5000),
])
def test_rejects_non_html_errors_and_oversized(response):
    with pytest.raises(WebPageError):
        build_client(lambda request: response, max_bytes=1000).fetch("https://noticias.com/a")

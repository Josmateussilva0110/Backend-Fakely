"""Contrato e utilitários HTTP comuns aos clientes de busca dos sites externos."""

from dataclasses import dataclass
from datetime import date
from typing import Protocol
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


class SearchError(Exception):
    pass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    excerpt: str
    published_at: date | None
    # Preenchidos por APIs de checagem: classificação textual e quem publicou a checagem
    rating: str | None = None
    publisher_name: str | None = None


class SearchClient(Protocol):
    def search(self, query: str, limit: int) -> list[SearchResult]: ...


def html_to_text(html: str) -> str:
    return " ".join(BeautifulSoup(html, "html.parser").get_text(" ").split())


def is_allowed_url(url: str, allowed_domains: tuple[str, ...]) -> bool:
    # Só repassa ao cliente links http(s) do próprio site consultado
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme in ("http", "https") and any(
        host == domain or host.endswith("." + domain) for domain in allowed_domains
    )


class HttpFetcher:
    """GET com limite de tamanho da resposta e validação do domínio final."""

    def __init__(self, client: httpx.Client, max_response_bytes: int):
        self.client = client
        self.max_response_bytes = max_response_bytes

    def get_text(
        self, url: str, params: dict, allowed_domains: tuple[str, ...], headers: dict | None = None
    ) -> str:
        try:
            with self.client.stream("GET", url, params=params, headers=headers) as response:
                if response.status_code != 200:
                    raise SearchError(f"HTTP {response.status_code} em {url}")
                # Redirecionamento para outro domínio é rejeitado
                if not is_allowed_url(str(response.url), allowed_domains):
                    raise SearchError(f"Redirecionamento inesperado para {response.url.host}")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise SearchError(f"Resposta maior que o limite em {url}")
                    chunks.append(chunk)
                return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
        except httpx.HTTPError as exc:
            raise SearchError(f"Falha de rede em {url}: {exc.__class__.__name__}") from exc

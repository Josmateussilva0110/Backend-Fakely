"""Download seguro da página enviada pelo usuário (validação de URL a cada redirecionamento)."""

from dataclasses import dataclass

import httpx

from app.integrations.url_safety import Resolver, UnsafeUrlError, ensure_public_url, resolve_host

HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


class WebPageError(Exception):
    pass


class BlockedUrlError(WebPageError):
    pass


@dataclass(frozen=True)
class WebPage:
    url: str
    html: str


class WebPageClient:
    def __init__(
        self,
        client: httpx.Client,
        max_bytes: int,
        max_redirects: int,
        timeout_seconds: float,
        resolver: Resolver = resolve_host,
    ):
        self.client = client
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.timeout_seconds = timeout_seconds
        self.resolver = resolver

    def fetch(self, url: str) -> WebPage:
        current = url
        try:
            # Redirecionamentos seguidos manualmente para validar cada destino
            for _ in range(self.max_redirects + 1):
                self._ensure_allowed(current)
                with self.client.stream(
                    "GET", current, follow_redirects=False, timeout=self.timeout_seconds,
                    headers={"Accept": ", ".join(HTML_CONTENT_TYPES)},
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise WebPageError("Redirecionamento sem destino")
                        current = str(response.url.join(location))
                        continue
                    return WebPage(url=str(response.url), html=self._read_html(response))
        except httpx.HTTPError as exc:
            raise WebPageError(f"Falha de rede: {exc.__class__.__name__}") from exc
        raise WebPageError("Redirecionamentos demais")

    def _ensure_allowed(self, url: str) -> None:
        try:
            ensure_public_url(url, self.resolver)
        except UnsafeUrlError as exc:
            raise BlockedUrlError(str(exc)) from exc

    def _read_html(self, response: httpx.Response) -> str:
        if response.status_code != 200:
            raise WebPageError(f"HTTP {response.status_code}")
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type not in HTML_CONTENT_TYPES:
            raise WebPageError("O conteúdo não é uma página HTML")
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > self.max_bytes:
                raise WebPageError("Página maior que o limite permitido")
            chunks.append(chunk)
        return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")

"""Extrai título, texto e data de publicação da página informada pelo usuário."""

from dataclasses import dataclass
from datetime import date, datetime

from bs4 import BeautifulSoup

from app.core.exceptions import UnprocessableError
from app.integrations.web_page_client import BlockedUrlError, WebPageClient, WebPageError

# Elementos que não fazem parte do conteúdo da notícia
NOISE_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form", "figure")
MIN_PARAGRAPH_LENGTH = 40
TITLE_META = (("property", "og:title"), ("name", "twitter:title"))
DATE_META = (
    ("property", "article:published_time"),
    ("name", "article:published_time"),
    ("itemprop", "datePublished"),
    ("name", "date"),
)


@dataclass(frozen=True)
class ArticleContent:
    title: str | None
    text: str
    published_at: date | None


class ContentExtractionService:
    def __init__(self, web_page_client: WebPageClient | None, max_text_length: int):
        self.web_page_client = web_page_client
        self.max_text_length = max_text_length

    def extract_from_url(self, url: str) -> ArticleContent:
        if self.web_page_client is None:
            raise UnprocessableError("A leitura de URLs está desativada. Envie o texto da notícia.")
        try:
            page = self.web_page_client.fetch(url)
        except BlockedUrlError as exc:
            raise UnprocessableError("URL não permitida.") from exc
        except WebPageError as exc:
            raise UnprocessableError(
                "Não foi possível obter o conteúdo da URL informada. Envie o texto da notícia."
            ) from exc

        content = parse_article_html(page.html, self.max_text_length)
        if not content.text:
            raise UnprocessableError("Não foi possível identificar o texto da notícia na página.")
        return content


def parse_article_html(html: str, max_text_length: int) -> ArticleContent:
    soup = BeautifulSoup(html, "html.parser")
    title = _meta_content(soup, TITLE_META) or (soup.title.get_text(" ", strip=True) if soup.title else None)
    published_at = _parse_date(_meta_content(soup, DATE_META))
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # Prioriza o corpo do artigo; sem <article>, usa os parágrafos da página
    container = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = [" ".join(p.get_text(" ").split()) for p in container.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p) >= MIN_PARAGRAPH_LENGTH)
    return ArticleContent(
        title=" ".join(title.split()) if title else None,
        text=text[:max_text_length],
        published_at=published_at,
    )


def _meta_content(soup: BeautifulSoup, candidates: tuple[tuple[str, str], ...]) -> str | None:
    for attribute, value in candidates:
        tag = soup.find("meta", attrs={attribute: value})
        if tag and tag.get("content", "").strip():
            return tag["content"].strip()
    return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date()
    except ValueError:
        return None

from datetime import date

import pytest

from app.core.exceptions import UnprocessableError
from app.integrations.web_page_client import BlockedUrlError, WebPage, WebPageError
from app.services.content_extraction_service import ContentExtractionService, parse_article_html

PARAGRAPH = "O Ministério da Saúde divulgou nesta segunda-feira o boletim semanal de vacinação."
HTML = f"""
<html><head>
  <title>Título da aba</title>
  <meta property="og:title" content="Boletim de vacinação é divulgado">
  <meta property="article:published_time" content="2026-10-05T10:00:00Z">
</head><body>
  <nav><p>Menu com um texto comprido o suficiente para parecer parágrafo.</p></nav>
  <article><p>{PARAGRAPH}</p><p>curto</p><script>alert(1)</script></article>
</body></html>
"""


class FakeWebPageClient:
    def __init__(self, html: str = HTML, error: Exception | None = None):
        self.html = html
        self.error = error

    def fetch(self, url: str) -> WebPage:
        if self.error:
            raise self.error
        return WebPage(url=url, html=self.html)


def test_parse_article_extracts_title_text_and_date():
    content = parse_article_html(HTML, max_text_length=10_000)
    assert content.title == "Boletim de vacinação é divulgado"
    assert content.text == PARAGRAPH
    assert content.published_at == date(2026, 10, 5)


def test_extract_from_url():
    service = ContentExtractionService(FakeWebPageClient(), 10_000)
    assert service.extract_from_url("https://g1.globo.com/x").text == PARAGRAPH


@pytest.mark.parametrize("client, message", [
    (FakeWebPageClient(error=BlockedUrlError("privado")), "URL não permitida."),
    (FakeWebPageClient(error=WebPageError("404")), "Não foi possível obter o conteúdo"),
    (FakeWebPageClient(html="<html><body><p>oi</p></body></html>"), "Não foi possível identificar"),
    (None, "desativada"),
])
def test_extract_from_url_errors(client, message):
    with pytest.raises(UnprocessableError) as error:
        ContentExtractionService(client, 10_000).extract_from_url("https://site.com/x")
    assert message in error.value.message

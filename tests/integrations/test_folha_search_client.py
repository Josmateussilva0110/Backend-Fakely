from datetime import date

from app.integrations.folha_search_client import parse_folha_results

DOMAINS = ("folha.uol.com.br",)

HTML = """
<ol class="u-list-unstyled c-search">
  <li class="c-headline c-headline--newslist">
    <div class="c-headline__content">
      <a href="https://www1.folha.uol.com.br/equilibrioesaude/2026/09/estudo-vacina.shtml">
        <h2 class="c-headline__title">  Estudo confirma   eficácia da vacina  </h2>
        <p class="c-headline__standfirst">Resumo da <b>matéria</b> ...</p>
        <time class="c-headline__dateline" datetime="11.set.2026 às 11h28">11.set.2026</time>
      </a>
    </div>
  </li>
  <li class="c-headline c-headline--newslist">
    <div class="c-headline__content">
      <a href="javascript:alert(1)"><h2 class="c-headline__title">Link inválido</h2></a>
    </div>
  </li>
  <li class="c-headline c-headline--newslist">
    <div class="c-headline__content">
      <a href="https://www1.folha.uol.com.br/mundo/2026/01/outra.shtml">
        <h2 class="c-headline__title">Outra notícia</h2>
      </a>
    </div>
  </li>
</ol>
"""


def test_parses_results_and_ignores_invalid_links():
    results = parse_folha_results(HTML, DOMAINS, limit=5)
    assert [r.title for r in results] == ["Estudo confirma eficácia da vacina", "Outra notícia"]
    assert results[0].excerpt == "Resumo da matéria ..."
    assert results[0].published_at == date(2026, 9, 11)
    assert results[1].published_at is None


def test_respects_limit():
    assert len(parse_folha_results(HTML, DOMAINS, limit=1)) == 1


def test_empty_page_returns_no_results():
    assert parse_folha_results("<html><body>Nenhum resultado</body></html>", DOMAINS, 5) == []

import pytest

from app.services.source_verification_service import SourceVerificationService, _Source
from app.services.verification_config_service import load_verification_config
from tests.fakes import FakeSearchClient, make_result

USER_TITLE = "Ivermectina tem 70% de eficácia contra a Covid, diz estudo"
USER_TEXT = "Um estudo divulgado nas redes afirma que a ivermectina tem 70% de eficácia contra a Covid-19."


@pytest.fixture
def config(settings):
    return load_verification_config(settings.verification_sources_path)


def build_service(settings, config, rules, clients: dict) -> SourceVerificationService:
    sources = [_Source(source, clients[source.key]) for source in config.sources if source.key in clients]
    return SourceVerificationService(settings, config, rules.stopwords, sources)


def test_config_has_the_three_sources(config):
    assert [s.key for s in config.sources] == ["lupa", "boatos", "folha"]


def test_extract_keywords_prefers_title_and_specific_words(settings, config, rules):
    service = build_service(settings, config, rules, {})
    keywords = service.extract_keywords(USER_TEXT, USER_TITLE)
    assert len(keywords) == settings.verification_max_keywords
    assert "ivermectina" in keywords
    assert "eficacia" in keywords


def test_extract_keywords_prioritizes_proper_nouns(settings, config, rules):
    service = build_service(settings, config, rules, {})
    text = "Fauci admitiu no Senado americano que Bolsonaro estava certo sobre o tratamento precoce."
    keywords = service.extract_keywords(text)
    assert keywords[:2] == ["bolsonaro", "senado"]
    assert "fauci" in keywords


def test_fact_check_false_is_matched(settings, config, rules):
    lupa = FakeSearchClient([
        make_result("É falso que a ivermectina tenha 70% de eficácia contra a Covid",
                    "https://www.agencialupa.org/checagem/x"),
        make_result("Paracetamol não causa autismo", "https://www.agencialupa.org/checagem/y"),
    ])
    service = build_service(settings, config, rules, {"lupa": lupa})

    result = service.verify(USER_TEXT, USER_TITLE)
    assert result.sources_checked == ("Agência Lupa",)
    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.verdict == "false"
    assert match.source_type == "fact_checker"
    assert match.similarity >= settings.verification_min_similarity


def test_boatos_uses_default_verdict(settings, config, rules):
    boatos = FakeSearchClient([make_result(
        "Ivermectina tem eficácia de 70% contra Covid", "https://www.boatos.org/saude/x.html",
    )])
    result = build_service(settings, config, rules, {"boatos": boatos}).verify(USER_TEXT, USER_TITLE)
    assert result.matches[0].verdict == "false"


def test_news_outlet_has_no_verdict(settings, config, rules):
    folha = FakeSearchClient([make_result(
        "Estudo diz que ivermectina não tem eficácia contra Covid", "https://www1.folha.uol.com.br/x.shtml",
    )])
    result = build_service(settings, config, rules, {"folha": folha}).verify(USER_TEXT, USER_TITLE)
    assert result.matches[0].verdict is None


def test_detect_verdict_checks_false_before_true(settings, config, rules):
    service = build_service(settings, config, rules, {})
    lupa = config.sources[0]
    assert service.detect_verdict(lupa, make_result("Não é verdade que X", "u")) == "false"
    assert service.detect_verdict(lupa, make_result("É verdade que X", "u")) == "true"
    assert service.detect_verdict(lupa, make_result("Entenda o caso X", "u")) is None


def test_failed_source_is_reported_without_breaking(settings, config, rules):
    clients = {"lupa": FakeSearchClient(error=True), "folha": FakeSearchClient([])}
    result = build_service(settings, config, rules, clients).verify(USER_TEXT, USER_TITLE)
    assert result.sources_failed == ("Agência Lupa",)
    assert result.sources_checked == ("Folha de S.Paulo",)
    assert result.has_failures


def test_retries_with_fewer_keywords_when_empty(settings, config, rules):
    lupa = FakeSearchClient([])
    build_service(settings, config, rules, {"lupa": lupa}).verify(USER_TEXT, USER_TITLE)
    assert len(lupa.queries) == 2
    assert len(lupa.queries[1].split()) == settings.verification_min_keywords


def test_text_without_keywords_skips_verification(settings, config, rules):
    lupa = FakeSearchClient([])
    assert build_service(settings, config, rules, {"lupa": lupa}).verify("de o a e", None) is None
    assert lupa.queries == []

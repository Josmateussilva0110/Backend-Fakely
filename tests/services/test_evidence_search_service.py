from pydantic import SecretStr

from app.services.evidence_search_service import EvidenceSearchService, SearchProvider
from tests.fakes import FakeSearchClient, make_result

USER_TITLE = "Ivermectina tem 70% de eficácia contra a Covid, diz estudo"
USER_CLAIM = "Um estudo divulgado nas redes afirma que a ivermectina tem 70% de eficácia contra a Covid-19."


def build_service(settings, config, rules, clients: dict) -> EvidenceSearchService:
    providers = [SearchProvider(source, clients[source.key]) for source in config.sources if source.key in clients]
    return EvidenceSearchService(settings, rules.stopwords, providers)


def test_config_lists_search_providers(verification_config):
    keys = [s.key for s in verification_config.sources]
    assert keys == ["lupa", "folha", "cnn_brasil", "google_fact_check"]


def test_google_fact_check_is_only_enabled_with_api_key(settings, verification_config, rules):
    import httpx

    with httpx.Client() as http:
        without_key = EvidenceSearchService.build(settings, verification_config, rules.stopwords, http)
        assert "Google Fact Check Tools" not in without_key.provider_names
        with_key = settings.model_copy(update={"google_fact_check_api_key": SecretStr("k")})
        assert "Google Fact Check Tools" in EvidenceSearchService.build(
            with_key, verification_config, rules.stopwords, http
        ).provider_names


def test_extract_keywords_prefers_specific_words(settings, verification_config, rules):
    service = build_service(settings, verification_config, rules, {})
    keywords = service.extract_keywords(USER_CLAIM, USER_TITLE)
    assert len(keywords) == settings.verification_max_keywords
    assert "ivermectina" in keywords
    assert "eficacia" in keywords


def test_extract_keywords_prioritizes_proper_nouns(settings, verification_config, rules):
    service = build_service(settings, verification_config, rules, {})
    text = "Fauci admitiu no Senado americano que Bolsonaro estava certo sobre o tratamento precoce."
    keywords = service.extract_keywords(text)
    assert keywords[:2] == ["bolsonaro", "senado"]
    assert "fauci" in keywords


def test_collects_results_per_source(settings, verification_config, rules):
    lupa = FakeSearchClient([make_result("É falso que a ivermectina tenha 70% de eficácia", "https://agencialupa.org/x")])
    outcome = build_service(settings, verification_config, rules, {"lupa": lupa}).search(USER_CLAIM, USER_TITLE)
    assert outcome.sources_checked == ("Agência Lupa",)
    assert outcome.results[0].source.key == "lupa"
    assert len(outcome.results[0].results) == 1
    assert outcome.accessed_at.tzinfo is not None


def test_failed_source_is_reported_without_breaking(settings, verification_config, rules):
    clients = {"lupa": FakeSearchClient(error=True), "folha": FakeSearchClient([])}
    outcome = build_service(settings, verification_config, rules, clients).search(USER_CLAIM, USER_TITLE)
    assert outcome.sources_failed == ("Agência Lupa",)
    assert outcome.sources_checked == ("Folha de S.Paulo",)


def test_retries_with_fewer_keywords_when_empty(settings, verification_config, rules):
    lupa = FakeSearchClient([])
    build_service(settings, verification_config, rules, {"lupa": lupa}).search(USER_CLAIM, USER_TITLE)
    assert len(lupa.queries) == 2
    assert len(lupa.queries[1].split()) == settings.verification_min_keywords


def test_text_without_keywords_skips_search(settings, verification_config, rules):
    lupa = FakeSearchClient([])
    assert build_service(settings, verification_config, rules, {"lupa": lupa}).search("de o a e", None) is None
    assert lupa.queries == []

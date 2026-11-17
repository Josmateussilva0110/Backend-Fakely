from app.services.evidence_search_service import EvidenceSearchService, SearchProvider
from tests.fakes import FakeSearchClient, FakeTagger, make_result

USER_TITLE = "Ivermectina tem 70% de eficácia contra a Covid, diz estudo"
USER_CLAIM = "Um estudo divulgado nas redes afirma que a ivermectina tem 70% de eficácia contra a Covid-19."


def build_service(settings, config, rules, clients: dict) -> EvidenceSearchService:
    providers = [SearchProvider(source, clients[source.key]) for source in config.sources if source.key in clients]
    return EvidenceSearchService(settings, rules.stopwords, providers)


def test_config_lists_search_providers(verification_config):
    keys = [s.key for s in verification_config.sources]
    assert keys == ["lupa", "folha", "cnn_brasil", "veja", "exame", "boatos"]


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


def test_extract_keywords_keeps_acronyms(settings, verification_config, rules):
    service = EvidenceSearchService(settings, rules.stopwords, [], acronyms=rules.acronyms)
    keywords = service.extract_keywords("o stf condenou o bolsonaro e o governo vai taxar o pix")
    assert keywords[:2] == ["stf", "pix"]


def test_extract_keywords_detects_unknown_uppercase_acronym(settings, verification_config, rules):
    service = EvidenceSearchService(settings, rules.stopwords, [], acronyms=rules.acronyms)
    keywords = service.extract_keywords("As chuvas foram provocadas pelo HAARP, segundo o papa Leão XIV")
    assert keywords[0] == "haarp"
    assert "xiv" not in keywords


def test_extract_keywords_ignores_case_in_shouting_text(settings, verification_config, rules):
    service = EvidenceSearchService(settings, rules.stopwords, [], acronyms=rules.acronyms)
    keywords = service.extract_keywords("URGENTE: STF CONDENA BOLSONARO A 27 ANOS DE PRISÃO")
    assert keywords == ["stf", "bolsonaro", "condena", "prisao"]


def test_extract_keywords_prefers_nouns_over_adjectives_and_verbs(settings, rules):
    tagger = FakeTagger({"Donald": "PROPN", "anunciou": "VERB", "importados": "ADJ", "elevadas": "ADJ", "sobre": "ADP"})
    service = EvidenceSearchService(settings, rules.stopwords, [], acronyms=rules.acronyms, tagger=tagger)
    keywords = service.extract_keywords("Donald anunciou tarifas elevadas sobre produtos importados")
    assert keywords == ["produtos", "tarifas", "donald", "anunciou"]


def test_extract_keywords_skips_tagger_in_shouting_text(settings, rules):
    tagger = FakeTagger({})
    service = EvidenceSearchService(settings, rules.stopwords, [], acronyms=rules.acronyms, tagger=tagger)
    assert service.extract_keywords("URGENTE: STF CONDENA BOLSONARO A 27 ANOS DE PRISÃO")[0] == "stf"
    assert tagger.calls == 0


def test_extract_keywords_skips_dates_and_scores(settings, rules):
    service = EvidenceSearchService(settings, rules.stopwords, [], acronyms=rules.acronyms)
    keywords = service.extract_keywords("Brasil perde para o Japão de 3x2 no amistoso em 14/11/2026 na COP30")
    assert "3x2" not in keywords
    assert "14/11/2026" not in keywords
    assert "cop30" in keywords


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


def test_retries_with_specific_keywords_when_results_are_off_topic(settings, verification_config, rules):
    folha = FakeSearchClient([make_result("Sob chuva, ato contra Temer em Curitiba reúne 250 pessoas",
                                          "https://www1.folha.uol.com.br/poder/ato.shtml")])
    outcome = build_service(settings, verification_config, rules, {"folha": folha}).search(USER_CLAIM, USER_TITLE)
    assert len(folha.queries) == 2
    # A página repetida na segunda busca não é duplicada
    assert len(outcome.results[0].results) == 1


def test_does_not_retry_when_results_mention_specific_keywords(settings, verification_config, rules):
    service = build_service(settings, verification_config, rules, {})
    specific = service.extract_keywords(USER_CLAIM, USER_TITLE)[: settings.verification_min_keywords]
    folha = FakeSearchClient([make_result(" ".join(specific).title(), "https://www1.folha.uol.com.br/a.shtml")])
    build_service(settings, verification_config, rules, {"folha": folha}).search(USER_CLAIM, USER_TITLE)
    assert len(folha.queries) == 1

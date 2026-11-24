from datetime import date, timedelta

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
    assert keywords == ["donald", "tarifas", "produtos", "anunciou"]


def test_extract_keywords_keeps_sentence_order_over_word_length(settings, rules):
    # Por tamanho, "categoria" e "medalha" tirariam "ouro" da busca
    tagger = FakeTagger({"conquistou": "VERB", "acima": "ADV"})
    service = EvidenceSearchService(settings, rules.stopwords, [], tagger=tagger)
    keywords = service.extract_keywords("Beatriz Souza conquistou ouro acima da categoria medalha")
    assert keywords == ["beatriz", "souza", "ouro", "categoria"]


def test_extract_keywords_keeps_first_and_last_part_of_compound_names(settings, rules):
    # Sem a regra, "Donald John Trump" tiraria "tarifa" da busca
    tagger = FakeTagger({"Donald": "PROPN", "John": "PROPN", "Trump": "PROPN", "anuncia": "VERB"})
    service = EvidenceSearchService(settings, rules.stopwords, [], tagger=tagger)
    keywords = service.extract_keywords("Donald John Trump anuncia tarifa")
    assert keywords == ["donald", "trump", "tarifa", "anuncia"]


def test_extract_keywords_joins_names_through_connectors_and_keeps_acronyms(settings, rules):
    names = {"Banco": "PROPN", "Central": "PROPN", "Brasil": "PROPN", "ONU": "PROPN", "do": "ADP", "e": "CCONJ"}
    tagger = FakeTagger(names)
    service = EvidenceSearchService(settings, rules.stopwords, [], acronyms=rules.acronyms, tagger=tagger)
    keywords = service.extract_keywords("Banco Central do Brasil ONU e taxa")
    assert "central" not in keywords
    assert {"banco", "brasil", "onu", "taxa"} == set(keywords)


def test_extract_keywords_ignores_generic_terms(settings, rules):
    tagger = FakeTagger({"rompeu": "VERB"})
    service = EvidenceSearchService(settings, rules.stopwords, [], tagger=tagger, generic_terms=rules.generic_terms)
    keywords = service.extract_keywords("Mina rompeu domingo Maceió após anos")
    assert keywords == ["mina", "maceio", "apos", "rompeu"]


def test_extract_keywords_tags_shouting_text_in_lowercase(settings, rules):
    # Em maiúsculas o modelo erra a classe ("TRUMP ANUNCIA" vira organização): ele recebe o texto em minúsculas
    tagger = FakeTagger({"urgente": "ADJ", "condena": "VERB"})
    service = EvidenceSearchService(settings, rules.stopwords, [], acronyms=rules.acronyms, tagger=tagger,
                                    generic_terms=rules.generic_terms, fact_verb_stems=rules.fact_verb_stems)
    keywords = service.extract_keywords("URGENTE STF CONDENA BOLSONARO A 27 ANOS DE PRISÃO")
    assert keywords == ["stf", "condena", "bolsonaro", "prisao"]


def test_fact_verb_competes_with_nouns_by_position(settings, rules):
    # "condena" diz o fato; "prisao" e "trama" aparecem em qualquer notícia do julgamento
    tagger = FakeTagger({"STF": "PROPN", "Bolsonaro": "PROPN", "condena": "VERB", "por": "ADP"})
    service = EvidenceSearchService(settings, rules.stopwords, [], acronyms=rules.acronyms, tagger=tagger,
                                    fact_verb_stems=rules.fact_verb_stems)
    keywords = service.extract_keywords("STF condena Bolsonaro por trama golpista e prisão")
    assert keywords == ["stf", "condena", "bolsonaro", "trama"]


def test_other_verbs_stay_behind_the_nouns(settings, rules):
    tagger = FakeTagger({"Donald": "PROPN", "anunciou": "VERB", "sobre": "ADP"})
    service = EvidenceSearchService(settings, rules.stopwords, [], tagger=tagger, fact_verb_stems=rules.fact_verb_stems)
    assert service.extract_keywords("Donald anunciou tarifas sobre produtos importados")[-1] != "anunciou"


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


def test_news_date_adds_a_search_within_the_publication_window(settings, verification_config, rules):
    # A eliminação de 2022 ficava enterrada por matérias de 2026 com os mesmos nomes
    old = make_result("Brasil é eliminado pela Croácia nos pênaltis", "https://www.cnnbrasil.com.br/croacia")
    cnn = FakeSearchClient([old])
    folha = FakeSearchClient([], supports_period=False)
    service = build_service(settings, verification_config, rules, {"cnn_brasil": cnn, "folha": folha})
    service.search("Brasil perde para a Croácia nos pênaltis", reference_date=date(2022, 12, 9))
    window = timedelta(days=settings.evidence_support_window_days)
    # A busca sem data continua: checagens de boatos antigos são de anos antes
    assert cnn.periods[0] is None
    assert (date(2022, 12, 9) - window, date(2022, 12, 9) + window) in cnn.periods
    assert all(period is None for period in folha.periods)


def test_no_news_date_searches_without_period(settings, verification_config, rules):
    cnn = FakeSearchClient([])
    service = build_service(settings, verification_config, rules, {"cnn_brasil": cnn})
    service.search("Brasil perde para a Croácia nos pênaltis")
    assert all(period is None for period in cnn.periods)

from datetime import date

import pytest

from app.core.exceptions import ServiceUnavailableError
from app.services.evidence_service import EvidenceService, to_hypothesis
from tests.fakes import FakeEntailmentModel, FakeTagger, make_outcome, make_result

CLAIM = "Ivermectina tem 70% de eficácia contra a Covid, diz estudo"


def build(evidence_service, verification_config, results):
    return evidence_service.build_evidences(make_outcome(verification_config, results), CLAIM)


def test_fact_check_false_contradicts(evidence_service, verification_config):
    evidences = build(evidence_service, verification_config, {"lupa": [
        make_result("É falso que a ivermectina tenha 70% de eficácia contra a Covid",
                    "https://www.agencialupa.org/checagem/x"),
        make_result("Paracetamol não causa autismo", "https://www.agencialupa.org/checagem/y"),
    ]})
    assert len(evidences) == 1
    evidence = evidences[0]
    assert evidence.stance == "contradicts"
    assert evidence.verdict == "false"
    assert evidence.category == "fact_checker"
    assert evidence.source_name == "Agência Lupa"
    assert evidence.position == 1
    assert evidence.accessed_at is not None


def test_provider_default_verdict_applies_without_pattern(evidence_service, verification_config):
    # Provedor que só publica desmentidos pode declarar um veredito padrão
    provider = verification_config.sources[0].model_copy(update={"default_verdict": "false"})
    result = make_result("Ivermectina tem eficácia de 70% contra Covid", "https://www.agencialupa.org/x")
    assert evidence_service.detect_verdict(provider, result, "fact_checker") == "false"


def test_news_outlet_report_supports_only_when_highly_relevant(evidence_service, verification_config, settings):
    evidences = build(evidence_service, verification_config, {"folha": [
        make_result("Ivermectina tem 70% de eficácia contra Covid, diz estudo", "https://www1.folha.uol.com.br/a.shtml"),
        make_result("Estudo sobre ivermectina e Covid gera debate entre médicos", "https://www1.folha.uol.com.br/b.shtml"),
    ]})
    by_url = {e.url: e for e in evidences}
    assert by_url["https://www1.folha.uol.com.br/a.shtml"].stance == "supports"
    weak = by_url.get("https://www1.folha.uol.com.br/b.shtml")
    assert weak is None or weak.stance == "neutral"


def test_same_page_from_two_providers_is_deduplicated(evidence_service, verification_config):
    evidences = build(evidence_service, verification_config, {
        "lupa": [make_result("É falso que ivermectina tem 70% de eficácia contra Covid",
                             "https://www.agencialupa.org/checagem/x")],
        "veja": [make_result("Ivermectina tem 70% de eficácia contra a Covid",
                                   "https://agencialupa.org/checagem/x/")],
    })
    assert len(evidences) == 1


def test_same_organization_and_republications_are_not_independent(evidence_service, verification_config):
    evidences = build(evidence_service, verification_config, {"folha": [
        make_result("Ivermectina tem 70% de eficácia contra Covid, diz estudo", "https://www1.folha.uol.com.br/a"),
        make_result("Ivermectina tem 70% de eficácia contra Covid, diz estudo", "https://www1.folha.uol.com.br/b"),
    ], "veja": [
        # Republicação do mesmo texto em outro veículo
        make_result("Ivermectina tem 70% de eficácia contra Covid, diz estudo", "https://g1.globo.com/c"),
    ]})
    independent = [e for e in evidences if e.independent]
    assert len(independent) == 1


def test_unregistered_domain_is_unknown(evidence_service, verification_config):
    evidences = build(evidence_service, verification_config, {"veja": [
        make_result("Ivermectina tem 70% de eficácia contra a Covid", "https://blog-desconhecido.net/x",
                    published_at=date(2020, 1, 1)),
    ]})
    assert evidences[0].category == "unknown"
    assert evidences[0].organization == "blog-desconhecido.net"


def test_detect_verdict_checks_false_before_true(evidence_service, verification_config):
    lupa = verification_config.sources[0]
    detect = evidence_service.detect_verdict
    assert detect(lupa, make_result("Não é verdade que X", "u"), "fact_checker") == "false"
    assert detect(lupa, make_result("É verdade que X", "u"), "fact_checker") == "true"
    assert detect(lupa, make_result("Entenda o caso X", "u"), "fact_checker") is None
    assert detect(lupa, make_result("É falso que X", "u"), "news_outlet") is None


SEEDORF_CLAIM = "Seedorf é o novo técnico do Corinthians"


def build_for(evidence_service, verification_config, claim, results):
    return evidence_service.build_evidences(make_outcome(verification_config, results), claim)


def test_headline_with_extra_details_supports_short_claim(evidence_service, verification_config):
    evidences = build_for(evidence_service, verification_config, SEEDORF_CLAIM, {
        "folha": [make_result("Corinthians anuncia Clarence Seedorf como novo técnico",
                              "https://www1.folha.uol.com.br/esporte/seedorf.shtml")],
        "veja": [make_result("Corinthians surpreende e anuncia Clarence Seedorf como novo técnico",
                             "https://veja.abril.com.br/esporte/seedorf")],
    })
    assert [e.stance for e in evidences] == ["supports", "supports"]


def test_denial_in_headline_does_not_support(evidence_service, verification_config):
    evidences = build_for(evidence_service, verification_config, SEEDORF_CLAIM, {
        "folha": [make_result("Corinthians nega que Seedorf seja o novo técnico",
                              "https://www1.folha.uol.com.br/esporte/nega.shtml")],
    })
    assert evidences[0].stance == "neutral"


def test_denial_present_in_claim_does_not_block(evidence_service, verification_config):
    claim = "Corinthians nega que Seedorf seja o novo técnico"
    evidences = build_for(evidence_service, verification_config, claim, {
        "folha": [make_result("Corinthians nega que Seedorf seja o novo técnico do time",
                              "https://www1.folha.uol.com.br/esporte/nega.shtml")],
    })
    assert evidences[0].stance == "supports"


def test_short_claim_inside_unrelated_headline_does_not_support(evidence_service, verification_config):
    evidences = build_for(evidence_service, verification_config, "Lula morreu", {
        "folha": [make_result("Lula visita hospital onde paciente morreu",
                              "https://www1.folha.uol.com.br/poder/visita.shtml")],
    })
    assert all(e.stance != "supports" for e in evidences)


CAMARA_CLAIM = "Rafael Câmara contratado para formula 1 em 2027"
CAMARA_HEADLINE = "Brasileiro Rafael Câmara é anunciado como piloto da Haas na F1 em 2027"


def with_model(settings, verification_config, registry, rules, scores):
    return EvidenceService(settings, verification_config, registry, rules.stopwords,
                           entailment_model=FakeEntailmentModel(scores))


def test_model_supports_paraphrased_headline(settings, verification_config, registry, rules):
    service = with_model(settings, verification_config, registry, rules, {CAMARA_HEADLINE: 0.79})
    evidences = build_for(service, verification_config, CAMARA_CLAIM, {
        "folha": [make_result(CAMARA_HEADLINE, "https://www1.folha.uol.com.br/esporte/camara.shtml")],
    })
    assert evidences[0].stance == "supports"
    assert evidences[0].entailment == 0.79
    # Sem o modelo, as palavras diferentes ("anunciado", "F1") não bastam para sustentar
    without_model = build_for(evidence_service_without_model(settings, verification_config, registry, rules),
                              verification_config, CAMARA_CLAIM,
                              {"folha": [make_result(CAMARA_HEADLINE, "https://www1.folha.uol.com.br/esporte/c.shtml")]})
    assert [e.stance for e in without_model] == ["neutral"]


def test_model_overrides_word_overlap_for_support(settings, verification_config, registry, rules):
    # Mesmas palavras, outro fato: o modelo não vê implicação
    headline = "Corinthians anuncia Clarence Seedorf como novo técnico"
    service = with_model(settings, verification_config, registry, rules, {headline: 0.05})
    evidences = build_for(service, verification_config, SEEDORF_CLAIM, {
        "folha": [make_result(headline, "https://www1.folha.uol.com.br/esporte/seedorf.shtml")],
    })
    assert evidences[0].stance == "neutral"


def test_denial_blocks_support_even_with_model(settings, verification_config, registry, rules):
    headline = "Corinthians nega que Seedorf seja o novo técnico"
    service = with_model(settings, verification_config, registry, rules, {headline: 0.9})
    evidences = build_for(service, verification_config, SEEDORF_CLAIM, {
        "folha": [make_result(headline, "https://www1.folha.uol.com.br/esporte/nega.shtml")],
    })
    assert evidences[0].stance == "neutral"


def evidence_service_without_model(settings, verification_config, registry, rules):
    return EvidenceService(settings, verification_config, registry, rules.stopwords)


def test_model_needs_the_claim_proper_nouns_in_the_headline(settings, verification_config, registry, rules):
    # Mesmas entidades (Real Madrid), outro protagonista: o modelo erra, a exigência dos nomes evita
    headline = "Courtois renova com o Real Madrid e fica até 2027, diz jornal"
    service = with_model(settings, verification_config, registry, rules, {headline: 0.92})
    evidences = build_for(service, verification_config, "Neymar foi contratado pelo Real Madrid", {
        "veja": [make_result(headline, "https://veja.abril.com.br/esportes/courtois")],
    })
    assert all(e.stance != "supports" for e in evidences)
    assert all(e.entailment is None for e in evidences)


def test_model_needs_a_fact_word_besides_the_names(settings, verification_config, registry, rules):
    # Mesmos nomes (Brasil, Japão), outro fato: o modelo real deu 0,7 de implicação para esta checagem
    headline = "É falso que Brasil virou desenho animado no Japão com Iron Tariffs"
    excerpt = "Circulam mensagens afirmando que o caso das tarifas teria virado um desenho animado no Japão."
    service = with_model(settings, verification_config, registry, rules, {f"{headline}. {excerpt}": 0.71})
    evidences = build_for(service, verification_config, "Brasil perde para o Japão de 3x2 no amistoso", {
        "boatos": [make_result(headline, "https://www.boatos.org/brasil/desenho-japao.html", excerpt=excerpt)],
    })
    assert all(e.stance == "neutral" for e in evidences)
    assert all(e.entailment is None for e in evidences)


def test_identical_headline_supports_even_when_model_disagrees(settings, verification_config, registry, rules):
    # O modelo real deu 0,48 para esta manchete contra ela mesma
    headline = "Vini Jr faz hat-trick e Brasil encerra 2026 com 7 a 0 sobre Singapura"
    service = with_model(settings, verification_config, registry, rules, {headline: 0.48})
    evidences = build_for(service, verification_config, headline, {
        "veja": [make_result(headline, "https://veja.abril.com.br/esportes/vini-singapura")],
    })
    assert evidences[0].stance == "supports"


def test_model_compares_the_claim_without_date_and_with_written_score(settings, verification_config, registry, rules):
    headline = "Brasil leva gol no fim e perde duelo com o Japão"
    excerpt = "Um gol nos acréscimos definiu a vitória por 3 a 2 do Japão sobre o Brasil em amistoso."
    model = FakeEntailmentModel({excerpt: 0.94})
    service = EvidenceService(settings, verification_config, registry, rules.stopwords, entailment_model=model)
    evidences = build_for(service, verification_config, "Brasil perde para o Japão de 3x2 no amistoso em 14/11/2026", {
        "folha": [make_result(headline, "https://www1.folha.uol.com.br/esporte/japao.shtml", excerpt=excerpt)],
    })
    assert model.hypotheses == ["Brasil perde para o Japão de 3 a 2 no amistoso"]
    assert evidences[0].stance == "supports"


@pytest.mark.parametrize(("claim", "hypothesis"), [
    ("Brasil perde para o Japão de 3x2 no dia 14/11/2026", "Brasil perde para o Japão de 3 a 2"),
    ("Lula sanciona a lei em 05/01/2025 em Brasília", "Lula sanciona a lei em Brasília"),
    ("Seedorf é o novo técnico do Corinthians", "Seedorf é o novo técnico do Corinthians"),
])
def test_to_hypothesis(claim, hypothesis):
    assert to_hypothesis(claim) == hypothesis


def test_model_reads_the_excerpt_along_with_the_headline(settings, verification_config, registry, rules):
    headline = "Paquetá aprova empate do Flamengo com o Estudiantes"
    excerpt = "Equipes ficaram no 0 a 0 pela semifinal da Copa Libertadores"
    # A frase do resumo vai separada da manchete; vale a maior nota
    service = with_model(settings, verification_config, registry, rules, {headline: 0.1, excerpt: 0.99})
    evidences = build_for(service, verification_config, "Flamengo empata com Estudiantes pela libertadores", {
        "veja": [make_result(headline, "https://veja.abril.com.br/esportes/paqueta", excerpt=excerpt)],
    })
    assert evidences[0].stance == "supports"


def test_false_verdict_in_news_outlet_headline_never_supports(settings, verification_config, registry, rules):
    headline = "É falso que o Japão declarou ivermectina mais eficaz do que vacinas contra a covid"
    service = with_model(settings, verification_config, registry, rules, {headline: 0.82})
    evidences = build_for(service, verification_config, "Japão declarou ivermectina mais eficaz do que vacinas", {
        "folha": [make_result(headline, "https://www1.folha.uol.com.br/equilibrioesaude/japao.shtml")],
    })
    assert evidences[0].stance == "neutral"


def test_question_headline_has_no_verdict(evidence_service, verification_config):
    lupa = verification_config.sources[0]
    result = make_result("É verdade que o TSE destruiu 200 mil urnas eletrônicas?", "u",
                         excerpt="É verdade que o TSE se livrou de urnas?")
    assert evidence_service.detect_verdict(lupa, result, "fact_checker") is None


def test_debunk_wording_is_false_verdict(evidence_service, verification_config):
    lupa = verification_config.sources[0]
    detect = evidence_service.detect_verdict
    photo = make_result("Foto de Flávio com elefante abatido foi gerada por IA", "u")
    assert detect(lupa, photo, "fact_checker") == "false"
    assert detect(lupa, make_result("Vídeos de doação circulam fora de contexto", "u"), "fact_checker") == "false"
    assert detect(lupa, make_result("Post distorce falas de Lula sobre analfabetismo", "u"), "fact_checker") == "false"


SEEDORF_HEADLINE = "Corinthians anuncia Clarence Seedorf como novo técnico"


def seedorf_evidence(service, verification_config, published_at, reference_date=None):
    outcome = make_outcome(verification_config, {"folha": [
        make_result(SEEDORF_HEADLINE, "https://www1.folha.uol.com.br/esporte/seedorf.shtml", published_at=published_at),
    ]})
    return service.build_evidences(outcome, SEEDORF_CLAIM, reference_date=reference_date)[0]


def test_old_report_does_not_support_new_fact(evidence_service, verification_config, settings):
    # Análise em 06/10/2026: matéria de 2 anos antes é só contexto
    old = seedorf_evidence(evidence_service, verification_config, date(2024, 10, 6))
    assert old.stance == "neutral"
    assert old.out_of_window


def test_report_close_to_news_date_supports(evidence_service, verification_config):
    evidence = seedorf_evidence(evidence_service, verification_config, date(2024, 10, 6),
                                reference_date=date(2024, 10, 1))
    assert evidence.stance == "supports"
    assert not evidence.out_of_window


def test_fact_check_verdict_ignores_the_window(evidence_service, verification_config):
    outcome = make_outcome(verification_config, {"lupa": [
        make_result("É falso que a ivermectina tenha 70% de eficácia contra a Covid",
                    "https://www.agencialupa.org/checagem/x", published_at=date(2020, 5, 1)),
    ]})
    assert evidence_service.build_evidences(outcome, CLAIM)[0].stance == "contradicts"


def test_busy_model_returns_service_unavailable(settings, verification_config, registry, rules):
    service = EvidenceService(settings.model_copy(update={"nli_queue_timeout_seconds": 0.01}), verification_config,
                              registry, rules.stopwords, entailment_model=FakeEntailmentModel({}))
    service.inference_slots.acquire()  # outra análise ocupando o modelo
    with pytest.raises(ServiceUnavailableError):
        build_for(service, verification_config, SEEDORF_CLAIM, {
            "folha": [make_result(SEEDORF_HEADLINE, "https://www1.folha.uol.com.br/esporte/seedorf.shtml")],
        })


def test_fact_check_with_extra_words_counts_when_model_matches(settings, verification_config, registry, rules):
    headline = "É falso que Tony Ramos morreu aos 89 anos em setembro de 2026 e morte foi anunciada na TV"
    # O modelo compara o boato, sem o "É falso que", com a afirmação
    rumor = "Tony Ramos morreu aos 89 anos em setembro de 2026 e morte foi anunciada na TV"
    claim = "Tony Ramos morreu aos 89 anos"
    # Boato e afirmação dizem o mesmo: a implicação vale nos dois sentidos
    service = with_model(settings, verification_config, registry, rules, {(claim, rumor): 0.61, (rumor, claim): 0.8})
    evidences = build_for(service, verification_config, claim, {
        "boatos": [make_result(headline, "https://www.boatos.org/entretenimento/tony-ramos.html")],
    })
    assert evidences[0].verdict == "false"
    assert evidences[0].stance == "contradicts"
    # A nota veio do modelo, comparando o boato e não a manchete com o veredito
    assert evidences[0].entailment == 0.61


@pytest.mark.parametrize(
    ("title", "rumor"),
    [
        ("É falso que as vacinas contra a Covid-19 vêm com aids", "as vacinas contra a Covid-19 vêm com aids"),
        ("Silvio Santos morreu hoje aos 86 anos #boato", "Silvio Santos morreu hoje aos 86 anos"),
        # Checagem escrita como negação: sem o "não" sobra o boato
        ("Caminhões com doações ao RS não estão sendo retidos", "Caminhões com doações ao RS estão sendo retidos"),
        # O "não" do próprio boato fica quando o veredito já foi retirado
        ("É falso que a vacina não protege contra a gripe", "a vacina não protege contra a gripe"),
    ],
)
def test_rumor_is_the_headline_without_the_verdict(evidence_service, title, rumor):
    assert evidence_service.rumor(title, "false") == rumor


def test_fact_check_about_other_fact_of_the_same_people_does_not_contradict(evidence_service, verification_config):
    # Mesmos nomes, outro fato: antes os nomes bastavam e um fato verdadeiro virava "provavelmente falso"
    evidences = build_for(evidence_service, verification_config, "STF condena Bolsonaro a 27 anos de prisão", {
        "lupa": [make_result("Bolsonaro no STF é falso", "https://www.agencialupa.org/checagem/stf")],
    })
    assert evidences[0].verdict == "false"
    assert evidences[0].stance == "neutral"


def test_negated_fact_check_headline_is_matched_by_the_model(settings, verification_config, registry, rules):
    headline = "Caminhões com doações ao Rio Grande do Sul não estão sendo retidos por falta de nota fiscal"
    rumor = "Caminhões com doações ao Rio Grande do Sul estão sendo retidos por falta de nota fiscal"
    claim = "Os caminhões com doação pro Rio Grande do Sul estão sendo parados sem nota fiscal"
    service = with_model(settings, verification_config, registry, rules, {(claim, rumor): 0.9, (rumor, claim): 0.95})
    evidences = build_for(service, verification_config, claim, {
        "lupa": [make_result(headline, "https://www.agencialupa.org/checagem/rs", excerpt="É falso que caminhões...")],
    })
    assert evidences[0].verdict == "false"
    assert evidences[0].entailment == 0.9
    assert evidences[0].stance == "contradicts"


def test_rumor_that_only_presupposes_the_claim_does_not_contradict(settings, verification_config, registry, rules):
    # A carta é o boato; a morte é verdadeira. O boato implica a morte, mas a morte não implica o boato
    headline = "É falso que o papa Francisco enviou uma carta ao padre antes de morrer"
    rumor = "o papa Francisco enviou uma carta ao padre antes de morrer"
    claim = "O papa Francisco morreu"
    service = with_model(settings, verification_config, registry, rules, {rumor: 0.99, (claim, rumor): 0.02})
    evidences = build_for(service, verification_config, claim, {
        "boatos": [make_result(headline, "https://www.boatos.org/religiao/papa-carta.html")],
    })
    assert all(evidence.stance != "contradicts" for evidence in evidences)


def test_fact_check_with_other_age_is_another_fact(evidence_service, verification_config):
    # Boato antigo da morte (86 anos) não desmente a morte real (93 anos): fato verdadeiro virava falso
    evidences = build_for(evidence_service, verification_config, "Silvio Santos morre aos 93 anos em São Paulo", {
        "boatos": [make_result("Silvio Santos morreu hoje aos 86 anos #boato", "https://www.boatos.org/silvio.html")],
    })
    assert all(evidence.stance != "contradicts" for evidence in evidences)


def test_same_number_written_differently_still_contradicts(evidence_service, verification_config):
    evidences = build_for(evidence_service, verification_config, "WhatsApp vai cobrar 0,37 centavos por mensagem", {
        "boatos": [
            make_result("WhatsApp vai cobrar 0.37 centavos por mensagem #boato", "https://www.boatos.org/zap.html")
        ],
    })
    assert evidences[0].stance == "contradicts"


@pytest.mark.parametrize(
    "title",
    [
        "O que já checamos sobre a COP30 em Belém; confira",
        "Metanol em bebidas alcoólicas: confira o que já checamos",
        "Debate entre presidenciáveis na TV: checamos em tempo real",
    ],
)
def test_roundup_page_has_no_verdict(evidence_service, verification_config, title):
    # O resumo dessas páginas fala em "falso" sobre outras afirmações; a página em si não julga nada
    lupa = verification_config.sources[0]
    result = make_result(title, "https://www.agencialupa.org/x", excerpt="É falso que a COP30 foi cancelada")
    assert evidence_service.detect_verdict(lupa, result, "fact_checker") is None


def test_roundup_page_does_not_contradict_the_claim(evidence_service, verification_config):
    evidences = build_for(evidence_service, verification_config, "A COP30 vai ser em Belém", {
        "lupa": [make_result("O que já checamos sobre a COP30 em Belém; confira", "https://www.agencialupa.org/cop",
                             excerpt="É falso que a COP30 foi transferida")],
    })
    assert all(evidence.stance != "contradicts" for evidence in evidences)


IMA_CLAIM = "Depois que tomei a vacina da covid o ímã gruda no braço"
IMA_HEADLINE = "Ímã gruda no braço de quem tomou vacina contra Covid-19 #boato"


def test_sentence_start_is_not_a_required_name(settings, verification_config, registry, rules):
    # Sem a classe gramatical, "Depois" (começo da frase) era exigido na manchete e a checagem nem ia ao modelo
    service = EvidenceService(settings, verification_config, registry, rules.stopwords,
                              entailment_model=FakeEntailmentModel({}), tagger=FakeTagger({"Depois": "ADV"}))
    assert service.required_names([IMA_CLAIM, ""], ("vacina", "covid")).all == {"vacin", "covid"}


def test_tagger_keeps_names_at_the_sentence_start(settings, verification_config, registry, rules):
    tags = {"Neymar": "PROPN", "Real": "PROPN", "Madrid": "PROPN", "foi": "AUX", "pelo": "ADP"}
    service = EvidenceService(settings, verification_config, registry, rules.stopwords, tagger=FakeTagger(tags))
    assert service.required_names(["Neymar foi contratado pelo Real Madrid", ""], ()).all == {"neyma", "real", "madri"}


def test_shouting_text_falls_back_to_the_keywords(settings, verification_config, registry, rules):
    # Em maiúsculas toda palavra parecia nome: a manchete teria de repetir a frase inteira
    service = evidence_service_without_model(settings, verification_config, registry, rules)
    assert service.required_names(["VACINAS CAUSAM AUTISMO", ""], ("vacinas", "autismo")).all == {"vacin", "autis"}


def test_model_alone_does_not_link_a_fact_check_about_another_subject(settings, verification_config, registry, rules):
    # O modelo deu 0,80 para esta checagem contra "A notícia que parou o Brasil" e o fato verdadeiro virou falso
    headline = "Notícia sobre fábricas da Coca-Cola fechadas não tem relação com unidades da empresa no Brasil"
    claim = "A notícia que parou o Brasil nesta sexta"
    rumor = with_model(settings, verification_config, registry, rules, {}).rumor(headline, "false")
    service = with_model(settings, verification_config, registry, rules, {(claim, rumor): 0.8})
    evidences = build_for(service, verification_config, claim, {
        "boatos": [make_result(headline, "https://www.boatos.org/brasil/coca-cola.html")],
    })
    assert all(e.stance != "contradicts" for e in evidences)


def test_fact_check_of_the_opposite_rumor_does_not_contradict(evidence_service, verification_config):
    # O boato julgado falso é "não morreu": a checagem confirma a morte, não a desmente
    claim = "A cantora Marília Mendonça morreu na queda do avião"
    evidences = build_for(evidence_service, verification_config, claim, {
        "boatos": [make_result("Marília Mendonça não morreu e está bem #boato", "https://www.boatos.org/x/marilia.html")],
    })
    assert all(e.stance != "contradicts" for e in evidences)


def test_agency_denial_with_negation_still_contradicts(evidence_service, verification_config):
    # O "não" é da agência ("cobrança é golpe"): o boato julgado é a taxação, a mesma afirmação
    evidences = build_for(evidence_service, verification_config, "A Receita Federal está taxando o Pix", {
        "lupa": [make_result("Receita Federal não está taxando o Pix; cobrança é golpe", "https://www.agencialupa.org/pix")],
    })
    assert evidences[0].stance == "contradicts"



def test_subject_is_required_but_not_every_place(settings, verification_config, registry, rules):
    # Exigir todos os nomes barrava a reportagem da queda em Vinhedo por não citar "Cascavel"
    tags = {"Voepass": "PROPN", "Vinhedo": "PROPN", "Cascavel": "PROPN"}
    service = EvidenceService(settings, verification_config, registry, rules.stopwords, tagger=FakeTagger(tags))
    required = service.required_names(["Avião da Voepass que saiu de Cascavel cai em Vinhedo", ""], ())
    assert required.subject == {"voepa"}
    assert required.covered_by(service.stems("Avião da Voepass cai em Vinhedo"), 0.5)
    # Sem o sujeito não basta: "Courtois renova com o Real Madrid" não fala de Neymar
    assert not required.covered_by(service.stems("Avião cai entre Cascavel e Vinhedo"), 0.5)


def test_fact_check_still_needs_every_name_to_contradict(settings, verification_config, registry, rules):
    # O modelo deu 0,92 para esta checagem de outro fato; com a regra relaxada ela desmentia um fato verdadeiro
    tags = {"OMS": "PROPN", "África": "PROPN", "Mpox": "PROPN"}
    service = EvidenceService(settings, verification_config, registry, rules.stopwords, tagger=FakeTagger(tags))
    required = service.required_names(["A OMS declarou emergência pela mpox por uma variante nova na África", ""], ())
    headline = "É falso que OMS tenha decretado lockdown por causa de Mpox"
    assert service.names_covered(required, headline, verdict=None)
    assert not service.names_covered(required, headline, verdict="false")


def fact_clause_service(settings, verification_config, registry, rules, tags):
    return EvidenceService(settings, verification_config, registry, rules.stopwords, tagger=FakeTagger(tags))


def test_fact_clause_drops_the_details_after_the_fact(settings, verification_config, registry, rules):
    # Os detalhes que a manchete não traz derrubavam a nota do modelo (0,03 -> 0,85)
    tags = {"Alemão": "PROPN", "Penha": "PROPN", "operação": "NOUN", "deixou": "VERB", "nos": "ADP"}
    service = fact_clause_service(settings, verification_config, registry, rules, tags)
    claim = "Uma operação policial nos complexos do Alemão e da Penha deixou mortos, na zona norte do Rio"
    assert service.fact_clause(claim) == "Uma operação policial nos complexos do Alemão e da Penha deixou mortos"


def test_fact_clause_keeps_the_apposition_and_decimal_numbers(settings, verification_config, registry, rules):
    tags = {"Estados": "PROPN", "Unidos": "PROPN", "Donald": "PROPN", "Trump": "PROPN", "anunciou": "VERB",
            "Guaíba": "PROPN", "atinge": "VERB"}
    service = fact_clause_service(settings, verification_config, registry, rules, tags)
    claim = "O presidente dos Estados Unidos, Donald Trump, anunciou uma tarifa, diz a Casa Branca"
    assert service.fact_clause(claim) == "O presidente dos Estados Unidos, Donald Trump, anunciou uma tarifa"
    assert service.fact_clause("Guaíba atinge 5,35 metros, diz órgão") == "Guaíba atinge 5,35 metros"


def test_fact_clause_skips_openings_not_linked_to_the_fact(settings, verification_config, registry, rules):
    # "Que notícia triste gente" tem verbo, mas nem nome nem palavra-chave: não é o fato
    tags = {"é": "AUX", "Preta": "PROPN", "Gil": "PROPN", "morreu": "VERB"}
    service = fact_clause_service(settings, verification_config, registry, rules, tags)
    claim = "Que notícia triste é essa, a Preta Gil morreu nos Estados Unidos"
    assert service.fact_clause(claim, ("preta", "morreu")) == claim



def test_fact_check_of_a_different_rumor_needs_entailment_both_ways(settings, verification_config, registry, rules):
    # O modelo viu 0,92 da emergência para o lockdown; o lockdown não implica a emergência
    headline = "É falso que OMS tenha decretado lockdown por causa de Mpox"
    claim = "A OMS declarou emergência mundial por causa da mpox"
    rumor = with_model(settings, verification_config, registry, rules, {}).rumor(headline, "false")
    service = with_model(settings, verification_config, registry, rules, {(claim, rumor): 0.92, (rumor, claim): 0.1})
    evidences = build_for(service, verification_config, claim, {
        "boatos": [make_result(headline, "https://www.boatos.org/saude/mpox.html")],
    })
    assert all(e.stance != "contradicts" for e in evidences)


def test_word_overlap_alone_does_not_skip_the_names_rule(settings, verification_config, registry, rules):
    # Sem os nomes o modelo nem compara, e 0,71 de semelhança desmentia o ouro de Beatriz Souza
    tags = {"Beatriz": "PROPN", "Souza": "PROPN", "Brasil": "PROPN", "Paris": "PROPN", "Bolsonaro": "PROPN"}
    service = EvidenceService(settings, verification_config, registry, rules.stopwords,
                              entailment_model=FakeEntailmentModel({}), tagger=FakeTagger(tags))
    claim = "Beatriz Souza ganhou a medalha de ouro no judô, primeiro ouro do Brasil em Paris"
    evidences = build_for(service, verification_config, claim, {
        "lupa": [make_result("É falso que Beatriz Souza dedicou medalha de ouro no judô a Bolsonaro",
                             "https://www.agencialupa.org/checagem/beatriz")],
    })
    assert all(e.stance != "contradicts" for e in evidences)


def test_question_headline_does_not_support(settings, verification_config, registry, rules):
    # A pergunta não afirma; o modelo deu 0,64 e o boato da taxação do Pix virou verdadeiro
    headline = "Como a Receita vai monitorar o Pix acima de R$ 5 mil? Entenda as mudanças"
    service = with_model(settings, verification_config, registry, rules, {headline: 0.64})
    evidences = build_for(service, verification_config, "A Receita Federal vai cobrar imposto do Pix acima de 5 mil", {
        "veja": [make_result(headline, "https://veja.abril.com.br/economia/pix")],
    })
    assert all(e.stance != "supports" for e in evidences)

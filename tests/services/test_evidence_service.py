from datetime import date

from tests.fakes import make_outcome, make_result

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
        "cnn_brasil": [make_result("Ivermectina tem 70% de eficácia contra a Covid",
                                   "https://agencialupa.org/checagem/x/")],
    })
    assert len(evidences) == 1


def test_same_organization_and_republications_are_not_independent(evidence_service, verification_config):
    evidences = build(evidence_service, verification_config, {"folha": [
        make_result("Ivermectina tem 70% de eficácia contra Covid, diz estudo", "https://www1.folha.uol.com.br/a"),
        make_result("Ivermectina tem 70% de eficácia contra Covid, diz estudo", "https://www1.folha.uol.com.br/b"),
    ], "cnn_brasil": [
        # Republicação do mesmo texto em outro veículo
        make_result("Ivermectina tem 70% de eficácia contra Covid, diz estudo", "https://g1.globo.com/c"),
    ]})
    independent = [e for e in evidences if e.independent]
    assert len(independent) == 1


def test_unregistered_domain_is_unknown(evidence_service, verification_config):
    evidences = build(evidence_service, verification_config, {"cnn_brasil": [
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

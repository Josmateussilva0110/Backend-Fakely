from datetime import date

from app.models.news_analysis_model import Classification, VerificationStatus
from app.services.verification_service import VERIFICATION_DISABLED_NOTICE
from tests.conftest import NEUTRAL_TEXT, SENSATIONAL_TEXT
from tests.fakes import FakeSearchService, make_outcome, make_result

TITLE = "Ivermectina tem 70% de eficácia contra a Covid, diz estudo"
TEXT = "Um estudo mostra que a ivermectina tem 70% de eficácia contra a Covid. Compartilhe com todos!"
LUPA_FALSE = make_result("É falso que a ivermectina tenha 70% de eficácia contra a Covid",
                         "https://www.agencialupa.org/checagem/x")
FOLHA_MATCH = make_result("Ivermectina tem 70% de eficácia contra Covid, diz estudo",
                          "https://www1.folha.uol.com.br/x.shtml")


def test_sensational_text_without_evidence_is_inconclusive(make_verification_service, verification_config):
    service = make_verification_service(FakeSearchService(make_outcome(verification_config, {"lupa": []})))
    report = service.verify(SENSATIONAL_TEXT)
    # Estilo de escrita não decide: vira alerta, e a ausência de evidências não é falsidade
    assert report.classification == Classification.INCONCLUSIVE
    assert report.style_indicators
    assert "não indica que a afirmação seja falsa" in report.justification


def test_trusted_site_alone_does_not_make_it_true(make_verification_service, verification_config):
    service = make_verification_service(FakeSearchService(make_outcome(verification_config, {"folha": []})))
    report = service.verify(NEUTRAL_TEXT, url="https://g1.globo.com/saude/x.html")
    assert report.classification == Classification.INCONCLUSIVE
    assert report.source_assessment.recognized is True
    assert any("não confirma a afirmação" in item for item in report.limitations)


def test_fact_check_contradiction_is_likely_false(make_verification_service, verification_config):
    service = make_verification_service(FakeSearchService(make_outcome(verification_config, {"lupa": [LUPA_FALSE]})))
    report = service.verify(TEXT, title=TITLE)
    assert report.classification == Classification.LIKELY_FALSE
    assert report.claim == TITLE
    assert report.verification_status == VerificationStatus.COMPLETE
    assert "Agência Lupa" in report.justification
    assert report.evidences[0].url == LUPA_FALSE.url
    assert report.method == "evidence_rules"
    assert report.model_version == "evidence-rules-1"


def test_news_outlet_report_is_likely_true(make_verification_service, verification_config):
    service = make_verification_service(FakeSearchService(make_outcome(verification_config, {"folha": [FOLHA_MATCH]})))
    report = service.verify(TEXT, title=TITLE)
    assert report.classification == Classification.LIKELY_TRUE
    assert "Folha de S.Paulo" in report.justification


def test_conflict_between_sources_is_inconclusive(make_verification_service, verification_config):
    outcome = make_outcome(verification_config, {"lupa": [LUPA_FALSE], "folha": [FOLHA_MATCH]})
    report = make_verification_service(FakeSearchService(outcome)).verify(TEXT, title=TITLE)
    assert report.classification == Classification.INCONCLUSIVE
    assert "divergem" in report.justification


def test_all_sources_failed_is_inconclusive_and_not_cached(make_verification_service, verification_config):
    outcome = make_outcome(verification_config, {}, failed=("Agência Lupa", "Folha de S.Paulo"))
    search = FakeSearchService(outcome)
    service = make_verification_service(search)
    report = service.verify(TEXT, title=TITLE)
    assert report.classification == Classification.INCONCLUSIVE
    assert report.verification_status == VerificationStatus.FAILED
    assert "não foi concluída" in report.justification
    service.verify(TEXT, title=TITLE)
    assert search.calls == 2


def test_partial_failure_is_reported(make_verification_service, verification_config):
    outcome = make_outcome(verification_config, {"lupa": [LUPA_FALSE]}, failed=("Folha de S.Paulo",))
    report = make_verification_service(FakeSearchService(outcome)).verify(TEXT, title=TITLE)
    assert report.verification_status == VerificationStatus.PARTIAL
    assert any("Não foi possível consultar: Folha de S.Paulo" in item for item in report.limitations)


def test_outdated_evidence_is_flagged(make_verification_service, verification_config):
    old = make_result(LUPA_FALSE.title, LUPA_FALSE.url, published_at=date(2015, 1, 1))
    report = make_verification_service(
        FakeSearchService(make_outcome(verification_config, {"lupa": [old]}))
    ).verify(TEXT, title=TITLE)
    assert any("desatualizadas" in item for item in report.limitations)


def test_short_text_is_inconclusive_without_search(make_verification_service, verification_config):
    search = FakeSearchService(make_outcome(verification_config, {"lupa": [LUPA_FALSE]}))
    report = make_verification_service(search).verify("Isso é falso")
    assert report.classification == Classification.INCONCLUSIVE
    assert report.claim is None
    assert report.verification_status == VerificationStatus.SKIPPED
    assert search.calls == 0


def test_search_disabled_is_inconclusive(make_verification_service):
    report = make_verification_service().verify(TEXT, title=TITLE)
    assert report.classification == Classification.INCONCLUSIVE
    assert report.limitations[0] == VERIFICATION_DISABLED_NOTICE


def test_result_is_cached_and_refresh_bypasses_cache(make_verification_service, verification_config):
    search = FakeSearchService(make_outcome(verification_config, {"lupa": [LUPA_FALSE]}))
    service = make_verification_service(search)
    first = service.verify(TEXT, title=TITLE)
    assert service.verify(TEXT, title=TITLE) is first
    service.verify(TEXT, title=TITLE, refresh=True)
    assert search.calls == 2


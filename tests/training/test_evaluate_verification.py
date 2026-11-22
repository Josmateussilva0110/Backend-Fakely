from datetime import UTC, date, datetime

import pytest

from app.integrations.search_client import SearchError
from app.models.news_analysis_model import Classification, Confidence, Intent, VerificationStatus
from app.services.decision_service import DecisionService
from app.services.evidence_service import EvidenceItem
from app.services.verification_service import VerificationReport
from tests.claim_dataset import ClaimCase
from tests.fakes import FakeSearchClient, make_result
from training.evaluate_verification import (
    STAGE_CLAIM_WRONG,
    STAGE_CONFLICT,
    STAGE_INVERTED,
    STAGE_NO_RESULTS,
    STAGE_OK,
    STAGE_OPINION_CHECKED,
    STAGE_OUT_OF_WINDOW,
    STAGE_UNRELATED,
    STAGE_VERDICT_MISSING,
    STAGE_VERDICT_UNMATCHED,
    STAGE_WEAK,
    CaseOutcome,
    RecordingSearchClient,
    SearchStore,
    diagnose,
    format_errors,
    parse_event_date,
)

CLAIM = "O governo vai taxar o Pix"
CASE = ClaimCase(
    id="whatsapp-pix",
    category="whatsapp",
    event="Boato do Pix (janeiro de 2025)",
    veracity="false",
    split="train",
    intent="claim",
    title=None,
    text="URGENTE!!! O governo vai taxar o Pix. Compartilhem",
    expected_claim=CLAIM,
    expected_keywords=("pix", "taxar"),
)


def make_report(
    classification=Classification.INCONCLUSIVE,
    claim=CLAIM,
    evidences=(),
    checked=("Lupa",),
    status=VerificationStatus.COMPLETE,
) -> VerificationReport:
    return VerificationReport(
        claim=claim,
        intent=Intent.CLAIM,
        classification=classification,
        confidence=Confidence.LOW,
        justification="",
        limitations=(),
        style_indicators=(),
        keywords=("pix", "taxar"),
        evidences=tuple(evidences),
        sources_checked=checked,
        sources_failed=(),
        verification_status=status,
        source_assessment=None,
        method="x",
        model_version="x",
    )


def make_evidence(
    stance: str,
    out_of_window: bool = False,
    organization: str = "lupa",
    category: str = "fact_checker",
    verdict: str | None = None,
    entailment: float | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        position=1,
        source_name=organization,
        organization=organization,
        category=category,
        retrieved_via="x",
        title="t",
        url=f"https://{organization}.org/1",
        excerpt="",
        published_at=None,
        accessed_at=datetime.now(UTC),
        relevance=0.9,
        stance=stance,
        verdict=verdict,
        independent=True,
        entailment=entailment,
        out_of_window=out_of_window,
    )


OLD_REPORT = make_evidence("neutral", out_of_window=True, organization="cnn", category="news_outlet", entailment=0.9)
UNMATCHED_CHECK = make_evidence("neutral", verdict="false")
CHECK_WITHOUT_VERDICT = make_evidence("neutral")


@pytest.fixture
def decisions(settings):
    return DecisionService(settings)


def test_parse_event_date_reads_full_date_and_month():
    assert parse_event_date("Morte do papa Francisco (21/04/2025)") == date(2025, 4, 21)
    assert parse_event_date("Mortes por metanol (setembro/outubro de 2025)") == date(2025, 9, 15)
    assert parse_event_date("Boato da vacina com chip (2020-2021)") is None


def test_recorded_search_is_reused_without_network(tmp_path):
    client = FakeSearchClient([make_result("Pix não será taxado", "https://lupa.org/1")])
    store = SearchStore(tmp_path / "cache.json", "reuse")
    RecordingSearchClient("lupa", client, store).search("pix taxar", 5)
    store.save()

    replay = SearchStore(tmp_path / "cache.json", "offline")
    results = RecordingSearchClient("lupa", FakeSearchClient(error=True), replay).search("pix taxar", 5)
    assert [r.title for r in results] == ["Pix não será taxado"]
    assert len(client.queries) == 1


def test_offline_mode_fails_on_unrecorded_search(tmp_path):
    store = SearchStore(tmp_path / "cache.json", "offline")
    with pytest.raises(SearchError):
        RecordingSearchClient("lupa", FakeSearchClient(), store).search("outra busca", 5)


def test_failed_search_is_not_recorded(tmp_path):
    store = SearchStore(tmp_path / "cache.json", "reuse")
    with pytest.raises(SearchError):
        RecordingSearchClient("lupa", FakeSearchClient(error=True), store).search("pix", 5)
    assert store.data == {}


def test_diagnose_correct_case(decisions):
    report = make_report(Classification.LIKELY_FALSE)
    assert diagnose(CASE, report, Classification.LIKELY_FALSE, True, 3, decisions) == STAGE_OK


@pytest.mark.parametrize(
    ("claim_hit", "raw_results", "evidences", "stage"),
    [
        (False, 3, (), STAGE_CLAIM_WRONG),
        (True, 0, (), STAGE_NO_RESULTS),
        (True, 3, (), STAGE_UNRELATED),
        (True, 3, (UNMATCHED_CHECK,), STAGE_VERDICT_UNMATCHED),
        (True, 3, (CHECK_WITHOUT_VERDICT,), STAGE_VERDICT_MISSING),
        (True, 3, (make_evidence("neutral", category="news_outlet"),), STAGE_WEAK),
        (True, 3, (make_evidence("supports"), make_evidence("contradicts", organization="aosfatos")), STAGE_CONFLICT),
    ],
)
def test_diagnose_finds_first_failing_stage(decisions, claim_hit, raw_results, evidences, stage):
    report = make_report(evidences=evidences)
    assert diagnose(CASE, report, Classification.LIKELY_FALSE, claim_hit, raw_results, decisions) == stage


def test_diagnose_inverted_decision_is_a_serious_error(decisions):
    report = make_report(Classification.LIKELY_TRUE, evidences=(make_evidence("supports"),))
    stage = diagnose(CASE, report, Classification.LIKELY_FALSE, True, 3, decisions)
    assert stage == STAGE_INVERTED
    assert CaseOutcome(CASE, report, Classification.LIKELY_FALSE, True, 3, stage).serious_error


def test_diagnose_opinion_with_classification(decisions):
    opinion = ClaimCase(**{**CASE.__dict__, "veracity": "opinion", "expected_claim": None, "expected_keywords": ()})
    report = make_report(Classification.LIKELY_FALSE)
    assert diagnose(opinion, report, Classification.INCONCLUSIVE, False, 3, decisions) == STAGE_OPINION_CHECKED


def test_old_report_explains_true_fact_before_unrelated_fact_check(decisions):
    # Para um fato verdadeiro, a reportagem antiga que o sustentaria é a causa, não a checagem de outro boato
    report = make_report(evidences=(UNMATCHED_CHECK, OLD_REPORT))
    assert diagnose(CASE, report, Classification.LIKELY_TRUE, True, 3, decisions) == STAGE_OUT_OF_WINDOW
    assert diagnose(CASE, report, Classification.LIKELY_FALSE, True, 3, decisions) == STAGE_VERDICT_UNMATCHED


def test_validation_errors_are_counted_but_never_listed():
    hidden = ClaimCase(**{**CASE.__dict__, "id": "validation-pix", "split": "validation"})
    wrong = make_report(Classification.INCONCLUSIVE)
    outcomes = [
        CaseOutcome(CASE, wrong, Classification.LIKELY_FALSE, True, 3, STAGE_UNRELATED),
        CaseOutcome(hidden, wrong, Classification.LIKELY_FALSE, True, 3, STAGE_UNRELATED),
    ]
    report = "\n".join(format_errors(outcomes))
    assert CASE.id in report
    assert "validation-pix" not in report
    assert "1 de validação ou teste não listados" in report

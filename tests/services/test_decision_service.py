from datetime import UTC, datetime

import pytest

from app.models.news_analysis_model import Classification, Confidence, VerificationStatus
from app.services.decision_service import DecisionService
from app.services.evidence_service import EvidenceItem

CHECKED = ("Agência Lupa",)


def make_evidence(stance: str, organization: str = "lupa", category: str = "fact_checker",
                  relevance: float = 0.9, independent: bool = True,
                  entailment: float | None = None) -> EvidenceItem:
    return EvidenceItem(
        position=1, source_name=organization, organization=organization, category=category,
        retrieved_via="x", title="t", url=f"https://{organization}.org/1", excerpt="", published_at=None,
        accessed_at=datetime.now(UTC), relevance=relevance, stance=stance,
        verdict=None, independent=independent, entailment=entailment,
    )


@pytest.fixture
def decision_service(settings):
    return DecisionService(settings)


def test_no_source_checked_is_inconclusive(decision_service):
    decision = decision_service.decide([make_evidence("contradicts")], sources_checked=())
    assert decision.classification == Classification.INCONCLUSIVE
    assert decision.reason == "verification_unavailable"


def test_no_evidence_is_inconclusive_not_false(decision_service):
    decision = decision_service.decide([], CHECKED)
    assert decision.classification == Classification.INCONCLUSIVE
    assert decision.reason == "no_evidence"


def test_contradiction_is_likely_false(decision_service):
    decision = decision_service.decide([make_evidence("contradicts")], CHECKED)
    assert decision.classification == Classification.LIKELY_FALSE
    assert len(decision.contradicting) == 1


def test_support_is_likely_true(decision_service):
    decision = decision_service.decide([make_evidence("supports", "folha", "news_outlet")], CHECKED)
    assert decision.classification == Classification.LIKELY_TRUE


def test_conflicting_evidence_is_inconclusive(decision_service):
    evidences = [make_evidence("contradicts"), make_evidence("supports", "folha", "news_outlet")]
    decision = decision_service.decide(evidences, CHECKED)
    assert decision.classification == Classification.INCONCLUSIVE
    assert decision.reason == "conflicting_evidence"


@pytest.mark.parametrize("evidence", [
    make_evidence("contradicts", relevance=0.6),
    make_evidence("contradicts", category="unknown"),
    make_evidence("contradicts", independent=False),
    make_evidence("neutral"),
])
def test_weak_evidence_does_not_decide(decision_service, evidence):
    decision = decision_service.decide([evidence], CHECKED)
    assert decision.classification == Classification.INCONCLUSIVE
    assert decision.reason == "insufficient_evidence"


def test_minimum_independent_sources_is_configurable(settings):
    strict = DecisionService(settings.model_copy(update={"decision_min_supporting_sources": 2}))
    one = [make_evidence("supports", "folha", "news_outlet")]
    two = one + [make_evidence("supports", "g1", "news_outlet")]
    assert strict.decide(one, CHECKED).classification == Classification.INCONCLUSIVE
    assert strict.decide(two, CHECKED).classification == Classification.LIKELY_TRUE


def test_report_supported_by_model_is_decisive_despite_low_word_overlap(decision_service, settings):
    evidence = make_evidence("supports", organization="folha", category="news_outlet",
                             relevance=0.5, entailment=settings.nli_min_entailment)
    assert decision_service.decide([evidence], CHECKED).classification == Classification.LIKELY_TRUE


def supporting(*organizations):
    return [make_evidence("supports", organization=o, category="news_outlet") for o in organizations]


def test_confidence_is_high_with_agreeing_sources_and_all_sites(decision_service):
    decision = decision_service.decide(supporting("folha", "veja"), CHECKED)
    assert decision_service.confidence(decision, VerificationStatus.COMPLETE) == Confidence.HIGH


def test_confidence_is_medium_with_one_source_or_partial_search(decision_service):
    one = decision_service.decide(supporting("folha"), CHECKED)
    assert decision_service.confidence(one, VerificationStatus.COMPLETE) == Confidence.MEDIUM
    two = decision_service.decide(supporting("folha", "veja"), CHECKED)
    assert decision_service.confidence(two, VerificationStatus.PARTIAL) == Confidence.MEDIUM


def test_inconclusive_has_low_confidence(decision_service):
    decision = decision_service.decide([], CHECKED)
    assert decision_service.confidence(decision, VerificationStatus.COMPLETE) == Confidence.LOW

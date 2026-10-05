from app.ml.news_classifier import NewsClassifier
from app.models.news_model import Classification
from app.services.news_analysis_service import NewsAnalysisService
from app.services.source_verification_service import VerificationMatch, VerificationResult
from tests.conftest import NEUTRAL_TEXT, SENSATIONAL_TEXT


def test_short_text_is_inconclusive(analysis_service):
    result = analysis_service.analyze("Isso é falso")
    assert result.classification == Classification.INCONCLUSIVE
    assert result.confidence is None


def test_sensational_text_is_likely_false(analysis_service):
    result = analysis_service.analyze(SENSATIONAL_TEXT)
    assert result.classification == Classification.LIKELY_FALSE
    assert result.confidence >= 0.6
    assert result.method == "heuristic"
    assert result.verification is None


def test_neutral_text_without_source_is_inconclusive(analysis_service):
    assert analysis_service.analyze(NEUTRAL_TEXT).classification == Classification.INCONCLUSIVE


def test_neutral_text_with_trusted_source_is_likely_true(analysis_service):
    result = analysis_service.analyze(NEUTRAL_TEXT, url="https://g1.globo.com/saude/x.html")
    assert result.classification == Classification.LIKELY_TRUE


def test_analysis_is_cached(analysis_service):
    first = analysis_service.analyze(SENSATIONAL_TEXT)
    assert analysis_service.analyze(SENSATIONAL_TEXT) is first


def test_combines_model_when_trained(settings, rules):
    texts = [SENSATIONAL_TEXT, SENSATIONAL_TEXT + " agora", NEUTRAL_TEXT, NEUTRAL_TEXT + " hoje"]
    classifier = NewsClassifier().train(texts, [1, 1, 0, 0], rules.stopwords)
    service = NewsAnalysisService(settings, rules, classifier)

    result = service.analyze(SENSATIONAL_TEXT)
    assert result.method == "model+heuristic"
    assert result.probability_fake > 0.5


class FakeVerificationService:
    def __init__(self, result: VerificationResult | None):
        self.result = result

    def verify(self, text, title=None):
        return self.result


def make_match(source_type: str, verdict: str | None, similarity: float = 0.8) -> VerificationMatch:
    return VerificationMatch(
        source_key="x", source_name="Fonte X", source_type=source_type, title="Título",
        url="https://x.org/1", excerpt="", published_at=None, similarity=similarity, verdict=verdict,
    )


def make_verification(*matches, failed=()) -> VerificationResult:
    return VerificationResult(
        keywords=("a", "b"), matches=tuple(sorted(matches, key=lambda m: m.similarity, reverse=True)), sources_checked=("Fonte X",), sources_failed=failed,
    )


def build(settings, rules, verification):
    return NewsAnalysisService(settings, rules, NewsClassifier(), FakeVerificationService(verification))


def test_fact_check_false_overrides_neutral_text(settings, rules):
    service = build(settings, rules, make_verification(make_match("fact_checker", "false")))
    result = service.analyze(NEUTRAL_TEXT, url="https://g1.globo.com/x")
    assert result.classification == Classification.LIKELY_FALSE
    assert result.method == "heuristic+verification"
    assert any("aponta como falso" in e for e in result.evidence)


def test_fact_check_true_lowers_probability(settings, rules):
    service = build(settings, rules, make_verification(make_match("fact_checker", "true")))
    assert service.analyze(SENSATIONAL_TEXT).classification == Classification.LIKELY_TRUE


def test_news_outlet_match_supports_true(settings, rules):
    service = build(settings, rules, make_verification(make_match("news_outlet", None)))
    # Texto neutro sem fonte seria inconclusivo; a matéria em veículo confiável o torna verdadeiro
    assert service.analyze(NEUTRAL_TEXT).classification == Classification.LIKELY_TRUE


def test_result_with_failed_source_is_not_cached(settings, rules):
    service = build(settings, rules, make_verification(failed=("Fonte Y",)))
    first = service.analyze(NEUTRAL_TEXT)
    assert service.analyze(NEUTRAL_TEXT) is not first
    assert any("Não foi possível consultar" in e for e in first.evidence)


def test_weak_fact_check_does_not_decide(settings, rules):
    service = build(settings, rules, make_verification(make_match("fact_checker", "false", similarity=0.6)))
    assert service.analyze(NEUTRAL_TEXT).classification == Classification.INCONCLUSIVE


def test_better_news_match_prevails_over_fact_check(settings, rules):
    # Caso real: matéria idêntica na Folha e checagem de outro assunto na Lupa
    verification = make_verification(
        make_match("news_outlet", None, similarity=1.0),
        make_match("fact_checker", "false", similarity=0.75),
    )
    result = build(settings, rules, verification).analyze(NEUTRAL_TEXT)
    assert result.classification == Classification.LIKELY_TRUE
    assert any("Notícia semelhante" in e for e in result.evidence)

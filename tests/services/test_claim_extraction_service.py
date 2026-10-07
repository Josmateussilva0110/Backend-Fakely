import pytest

from app.services.claim_extraction_service import ClaimExtractionService
from tests.conftest import SENSATIONAL_TEXT


@pytest.fixture
def claim_service(rules, settings):
    return ClaimExtractionService(rules, settings.min_words, max_candidates=3, max_length=settings.claim_max_length)


def test_descriptive_title_is_the_main_claim(claim_service):
    result = claim_service.extract("Texto qualquer da notícia com várias palavras.", "Vacina X foi proibida no Brasil")
    assert result.main_claim == "Vacina X foi proibida no Brasil"


def test_picks_the_most_specific_sentence_when_title_is_vague(claim_service):
    text = (
        "Compartilhe antes que apaguem! "
        "O Ministério da Saúde suspendeu a vacina X em 12 estados nesta semana. "
        "Veja mais detalhes aqui."
    )
    result = claim_service.extract(text, "BOMBA!!!")
    assert result.main_claim == "O Ministério da Saúde suspendeu a vacina X em 12 estados nesta semana."
    assert len(result.candidates) <= 3


def test_call_to_action_is_not_preferred(claim_service):
    result = claim_service.extract(SENSATIONAL_TEXT)
    assert result.main_claim is not None
    assert not result.main_claim.lower().startswith("urgente")


def test_short_text_has_no_claim(claim_service):
    assert claim_service.extract("Isso é falso").main_claim is None


def test_claim_is_truncated(rules, settings):
    service = ClaimExtractionService(rules, settings.min_words, max_candidates=1, max_length=20)
    assert len(service.extract("Palavra " * 50).main_claim) == 20

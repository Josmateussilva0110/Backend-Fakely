import pytest

from app.services.claim_extraction_service import ClaimExtractionService
from tests.conftest import SENSATIONAL_TEXT


@pytest.fixture
def claim_service(rules, settings):
    return ClaimExtractionService(rules, settings.min_words, settings.claim_max_length)


def test_descriptive_title_is_the_main_claim(claim_service):
    claim = claim_service.extract("Texto qualquer da notícia com várias palavras.", "Vacina X foi proibida no Brasil")
    assert claim == "Vacina X foi proibida no Brasil"


def test_picks_the_most_specific_sentence_when_title_is_vague(claim_service):
    text = (
        "Compartilhe antes que apaguem! "
        "O Ministério da Saúde suspendeu a vacina X em 12 estados nesta semana. "
        "Veja mais detalhes aqui."
    )
    claim = claim_service.extract(text, "BOMBA!!!")
    assert claim == "O Ministério da Saúde suspendeu a vacina X em 12 estados nesta semana."


def test_call_to_action_is_not_preferred(claim_service):
    claim = claim_service.extract(SENSATIONAL_TEXT)
    assert claim is not None
    assert not claim.lower().startswith("urgente")


def test_short_text_has_no_claim(claim_service):
    assert claim_service.extract("Isso é falso") is None


def test_claim_is_truncated(rules, settings):
    service = ClaimExtractionService(rules, settings.min_words, max_length=20)
    assert len(service.extract("Palavra " * 50)) == 20

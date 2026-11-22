import pytest

from app.models.news_analysis_model import Intent
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.intent_service import IntentService
from tests.fakes import FakeTagger


@pytest.fixture
def claim_service(rules, settings):
    return ClaimExtractionService(rules, settings.min_words, settings.claim_max_length)


def detect(rules, claim_service, text, tagger=None):
    return IntentService(rules, claim_service, tagger).detect(text, claim_service.extract(text))


@pytest.mark.parametrize(
    "text",
    [
        "Quem ganhou a eleição para presidente em 2022?",
        "Quantos anos de prisão o Bolsonaro pegou?",
        "Em qual cidade foi a COP30?",
        "Por que o X foi bloqueado no Brasil?",
    ],
)
def test_open_question_has_no_claim(rules, claim_service, text):
    result = detect(rules, claim_service, text)
    assert result.intent == Intent.OPEN_QUESTION
    assert result.claim is None


@pytest.mark.parametrize(
    "text",
    ["É verdade que o governo vai taxar o Pix a partir de janeiro?", "O Flávio Dino virou ministro do STF mesmo?"],
)
def test_yes_no_question_keeps_the_claim(rules, claim_service, text):
    result = detect(rules, claim_service, text)
    assert result.intent == Intent.YES_NO_QUESTION
    assert result.claim


def test_short_question_at_the_start_does_not_make_a_question(rules, claim_service):
    text = "Vocês sabiam? A Marielle Franco era casada com o Marcinho VP e a mídia está escondendo isso."
    assert detect(rules, claim_service, text).intent != Intent.YES_NO_QUESTION


def test_opinion(rules, claim_service):
    assert detect(rules, claim_service, "Na minha opinião o Botafogo merecia ganhar mais títulos.").intent == (
        Intent.OPINION
    )


def test_future_claim_is_a_possible_prediction(rules, claim_service):
    result = detect(rules, claim_service, "O dólar vai passar de 10 reais até o fim do ano, guardem o dinheiro.")
    assert result.intent == Intent.PREDICTION
    # A previsão continua com a frase: ela ainda é verificada (pode ser boato já desmentido)
    assert result.claim


def test_past_claim_is_a_claim(rules, claim_service):
    assert detect(rules, claim_service, "O papa Francisco morreu nesta segunda-feira no Vaticano.").intent == (
        Intent.CLAIM
    )


CHIP_TEXT = "A vacina tem chip, o sinal espalha o vírus e a máscara causa falta de oxigênio."
CHIP_TAGS = {"tem": "VERB", "espalha": "VERB", "causa": "VERB", "e": "CCONJ", "o": "DET", "a": "DET"}


def test_three_independent_claims_are_multiple_claims(rules, claim_service):
    result = detect(rules, claim_service, CHIP_TEXT, FakeTagger(CHIP_TAGS))
    assert result.intent == Intent.MULTIPLE_CLAIMS
    # A frase verificada não muda: dividir sem análise sintática corta frases normais
    assert result.claim == claim_service.extract(CHIP_TEXT)


def test_two_clauses_are_not_enough(rules, claim_service):
    text = "O Banco Central lançou o Pix e as transferências entre pessoas são gratuitas."
    tags = {"lançou": "VERB", "são": "AUX", "e": "CCONJ", "O": "DET", "as": "DET"}
    assert detect(rules, claim_service, text, FakeTagger(tags)).intent == Intent.CLAIM


def test_dependent_clause_does_not_start_a_new_claim(rules, claim_service):
    text = "Lula passou por uma cirurgia em São Paulo, para drenar uma hemorragia, e teve alta, depois voltou."
    tags = {
        "passou": "VERB",
        "para": "ADP",
        "drenar": "VERB",
        "e": "CCONJ",
        "teve": "VERB",
        "depois": "ADV",
        "voltou": "VERB",
    }
    assert detect(rules, claim_service, text, FakeTagger(tags)).intent != Intent.MULTIPLE_CLAIMS


def test_without_tagger_there_is_no_multiple_claims(rules, claim_service):
    assert detect(rules, claim_service, CHIP_TEXT).intent == Intent.CLAIM

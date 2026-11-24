import pytest

from app.core.config import get_settings
from app.ml.part_of_speech_tagger import PartOfSpeechTagger
from app.services.analysis_rules_service import load_analysis_rules
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.evidence_search_service import EvidenceSearchService
from app.services.intent_service import IntentService
from app.services.text_processing_service import normalize_text
from tests.claim_dataset import (
    CATEGORIES,
    CHECKABLE_VERACITIES,
    INTENTS,
    SPLITS,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    VERACITIES,
    compute_metrics,
    evaluate_case,
    format_report,
    format_summary,
    has_keyword,
    in_split,
    keyword_alternatives,
    load_cases,
)
from tests.conftest import SENSATIONAL_TEXT

# Piso do conjunto de avaliação (sem melhorias: 72% e 55%; com regras: 95% e 69%); suba ao melhorar.
# Com 150 casos (48 novos de notícia, WhatsApp e título chamativo) o resultado ficou em 92,7% e 69,3%:
# os 102 antigos seguem em 95% e 76%, a queda vem dos casos novos, mais difíceis.
# Palavras-chave na ordem da frase e sem termos genéricos (datas, "anos"): 69,3% -> 74,6%.
# Nome composto só com a primeira e a última palavra ("Donald Trump", não "Donald John Trump"): 75,8%.
# Com 250 casos o piso vale só para o treinamento (139). Entraram previsões e perguntas abertas, que ainda
# viram afirmação (0% até existir a etapa de intenção): treinamento em 86% e 77%.
# Com a etapa de intenção, previsões e perguntas abertas deixam de virar afirmação: 86% -> 90%;
# intenção certa em 88% do treinamento.
# Com 320 casos (178 no treinamento) entram boatos com "vai + verbo" e títulos vagos com nome próprio,
# mais difíceis: afirmação 90% -> 88,8%, intenção 88% -> 86,5%; palavras-chave 77% -> 79,8%.
# Verbos do fato ("condena", "morreu") disputando a busca com os nomes e texto em maiúsculas passando pelo
# modelo em minúsculas: palavras-chave 79,8% -> 82,5%; título vago com "Brasil": afirmação 88,8% -> 89,3%.
MIN_DATASET_CLAIM_ACCURACY = 0.89
MIN_DATASET_KEYWORD_RECALL = 0.82
MIN_DATASET_INTENT_ACCURACY = 0.86


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


def test_question_becomes_a_statement(claim_service):
    claim = claim_service.extract("É verdade que o governo vai taxar o Pix a partir de janeiro?")
    assert claim == "O governo vai taxar o Pix a partir de janeiro"


def test_question_drops_the_alternative(claim_service):
    claim = claim_service.extract("O avião que caiu em Vinhedo era da Voepass ou da Latam?")
    assert claim == "O avião que caiu em Vinhedo era da Voepass"


def test_alarm_label_before_colon_is_removed(claim_service):
    claim = claim_service.extract("Atenção aposentados: o governo vai confiscar a poupança na segunda.")
    assert claim == "O governo vai confiscar a poupança na segunda."


def test_label_without_alarm_is_kept(claim_service):
    claim = claim_service.extract("Agora é lei: celular está proibido nas escolas do Brasil.")
    assert claim == "Agora é lei: celular está proibido nas escolas do Brasil."


def test_vague_title_loses_to_the_body(claim_service):
    text = "O papa Francisco morreu nesta segunda-feira, aos 88 anos, no Vaticano."
    assert claim_service.extract(text, "Você não vai acreditar no que aconteceu hoje") == text


def test_title_without_name_or_number_loses_to_the_body(claim_service):
    text = "A CBF confirmou Carlo Ancelotti como técnico da seleção brasileira."
    assert claim_service.extract(text, "Ninguém esperava essa reviravolta no futebol") == text


def test_vague_title_is_used_when_body_is_too_short(claim_service):
    assert claim_service.extract("Veja.", "Ninguém esperava essa reviravolta no futebol") is not None


def test_shouting_text_ignores_the_alarm_opening(claim_service):
    claim = claim_service.extract("URGENTE: STF CONDENA BOLSONARO A 27 ANOS DE PRISÃO. ACABOU DE SAIR O RESULTADO")
    assert claim == "STF CONDENA BOLSONARO A 27 ANOS DE PRISÃO."


def test_opinion_has_no_claim(claim_service):
    text = "Na minha opinião o Ancelotti não vai dar certo na seleção brasileira."
    assert claim_service.extract(text) is None
    assert claim_service.is_opinion(text)


def test_fact_next_to_an_opinion_is_kept(claim_service):
    text = "Acho um absurdo tudo isso. O governo federal vai taxar as compras internacionais em 20%."
    assert claim_service.extract(text) == "O governo federal vai taxar as compras internacionais em 20%."
    assert not claim_service.is_opinion(text)


# Conjunto de avaliação com textos reais (tests/data/claim_extraction_cases.json).
# Para ver o relatório por categoria: pytest tests/services/test_claim_extraction_service.py -k dataset -s
@pytest.fixture(scope="module")
def dataset_results():
    settings = get_settings()
    rules = load_analysis_rules(settings.analysis_rules_path)
    claim_service = ClaimExtractionService(rules, settings.min_words, settings.claim_max_length)
    # Mesma configuração da API: siglas e classe gramatical (spaCy) na escolha das palavras-chave
    tagger = PartOfSpeechTagger.load(settings.pos_tagger_model)
    search_service = EvidenceSearchService(
        settings,
        rules.stopwords,
        [],
        acronyms=rules.acronyms,
        tagger=tagger,
        generic_terms=rules.generic_terms,
        fact_verb_stems=rules.fact_verb_stems,
    )
    try:
        intent_service = IntentService(rules, claim_service, tagger)
        return [
            evaluate_case(case, claim_service, intent_service, search_service, rules.stopwords) for case in load_cases()
        ]
    finally:
        search_service.close()


def test_dataset_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 100
    assert len({case.id for case in cases}) == len(cases)
    assert {case.category for case in cases} == set(CATEGORIES)
    for case in cases:
        assert case.veracity in VERACITIES, case.id
        assert case.split in SPLITS, case.id
        assert case.intent in INTENTS, case.id
        assert case.checkable == (case.veracity in CHECKABLE_VERACITIES), case.id
        assert bool(case.expected_keywords) == case.checkable, case.id
        # Palavras-chave esperadas já normalizadas, como as devolvidas pela busca
        assert all(k == normalize_text(k) for k in case.expected_keywords), case.id
        assert all(all(keyword_alternatives(k)) for k in case.expected_keywords), case.id


@pytest.mark.parametrize(
    ("split", "share"), [(TRAIN_SPLIT, (0.5, 0.65)), (VALIDATION_SPLIT, (0.2, 0.3)), (TEST_SPLIT, (0.15, 0.25))]
)
def test_dataset_splits_cover_every_category(split, share):
    # Cada grupo precisa ter todos os tipos de texto, senão não mede nada sobre eles
    cases = load_cases(split)
    assert {case.category for case in cases} == set(CATEGORIES)
    low, high = share
    assert low <= len(cases) / len(load_cases()) <= high


def test_dataset_metrics_do_not_regress(dataset_results):
    """Piso com o resultado atual: ao melhorar a interpretação, suba os valores."""
    # O piso vale para o treinamento; a validação só aparece em totais e o teste nem isso
    metrics = compute_metrics(in_split(dataset_results, TRAIN_SPLIT))
    # Na saída só a tabela; os erros caso a caso aparecem quando o piso é violado
    print("\n" + format_summary(dataset_results))
    report = format_report(dataset_results)
    assert metrics.claim_accuracy >= MIN_DATASET_CLAIM_ACCURACY, report
    assert metrics.keyword_recall >= MIN_DATASET_KEYWORD_RECALL, report
    assert metrics.intent_accuracy >= MIN_DATASET_INTENT_ACCURACY, report


def test_expected_keyword_accepts_any_alternative():
    assert has_keyword("cirurgia|operado", ("lula", "operado"))
    assert not has_keyword("cirurgia|operado", ("lula", "hemorragia"))
    assert has_keyword("lula", ("lula",))


def test_title_whose_only_name_is_brasil_loses_to_the_body(claim_service):
    text = "A cantora Marília Mendonça morreu na queda do avião em que viajava para um show em Minas Gerais."
    assert claim_service.extract(text, "A notícia que parou o Brasil nesta sexta") == text

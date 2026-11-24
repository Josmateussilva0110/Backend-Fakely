import pytest

from app.services.text_processing_service import (
    describe_style_signals,
    extract_features,
    extract_quantities,
    normalize_text,
    parse_number,
    quantities_conflict,
)


def test_normalize_text_removes_accents_urls_and_spaces():
    assert normalize_text("  Atenção!   Veja https://x.com/a  AGORA ") == "atencao! veja agora"


def test_extract_features_detects_sensationalism(rules):
    text = "URGENTE!!! Compartilhe antes que apaguem: a mídia esconde a CURA DEFINITIVA!"
    features = extract_features(text, rules)
    assert "urgente" in features.sensational_terms
    assert "antes que apaguem" in features.sensational_terms
    assert features.repeated_punctuation_count >= 1
    assert features.uppercase_ratio > 0.3


def test_describe_style_signals(rules):
    text = "URGENTE!!! Compartilhe antes que apaguem: a mídia esconde a CURA DEFINITIVA!"
    signals = describe_style_signals(extract_features(text, rules), rules)
    assert any(s.startswith("Termos sensacionalistas") for s in signals)
    assert "Uso excessivo de letras maiúsculas" in signals

    neutral = "O governo anunciou uma nova medida para a saúde"
    assert describe_style_signals(extract_features(neutral, rules), rules) == []


def quantities(text: str) -> dict[str, set[float]]:
    return extract_quantities(normalize_text(text))


@pytest.mark.parametrize(
    ("raw", "value"), [("1.518", 1518), ("5.000", 5000), ("0,37", 0.37), ("0.37", 0.37), ("2,5", 2.5)]
)
def test_parse_number_reads_brazilian_and_american_formats(raw, value):
    assert parse_number(raw) == value


def test_extracts_age_percent_and_money():
    assert quantities("Pelé morre aos 82 anos") == {"age": {82}}
    assert quantities("tarifa de 50% sobre produtos") == {"percent": {50}}
    assert quantities("isenção até R$ 5 mil") == quantities("isenção até 5.000 reais") == {"money": {5000}}


def test_numbers_without_unit_are_ignored():
    # "27 anos de prisão" e "acima de 5 mil" não dizem o que medem com segurança
    assert quantities("27 anos e 3 meses de prisão; acima de 5 mil") == {}


def test_conflict_only_when_both_texts_have_the_same_kind():
    assert quantities_conflict(quantities("aos 93 anos"), quantities("aos 86 anos"))
    assert not quantities_conflict(quantities("aos 93 anos"), quantities("morreu em São Paulo"))
    assert not quantities_conflict(quantities("aos 93 anos"), quantities("50% dos votos"))
    assert not quantities_conflict(quantities("R$ 5 mil"), quantities("5.000 reais"))

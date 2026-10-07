from app.services.text_processing_service import (
    describe_style_signals, extract_features, normalize_text, preprocess_text,
)


def test_normalize_text_removes_accents_urls_and_spaces():
    assert normalize_text("  Atenção!   Veja https://x.com/a  AGORA ") == "atencao! veja agora"


def test_preprocess_text_removes_stopwords(rules):
    processed = preprocess_text("O governo anunciou uma nova medida para a saúde", rules.stopwords)
    assert "o" in processed.tokens
    assert "o" not in processed.content_tokens
    assert "saude" in processed.content_tokens


def test_extract_features_detects_sensationalism(rules):
    text = "URGENTE!!! Compartilhe antes que apaguem: a mídia esconde a CURA DEFINITIVA!"
    features = extract_features(preprocess_text(text, rules.stopwords), rules)
    assert "urgente" in features.sensational_terms
    assert "antes que apaguem" in features.sensational_terms
    assert features.repeated_punctuation_count >= 1
    assert features.uppercase_ratio > 0.3


def test_describe_style_signals(rules):
    text = "URGENTE!!! Compartilhe antes que apaguem: a mídia esconde a CURA DEFINITIVA!"
    signals = describe_style_signals(extract_features(preprocess_text(text, rules.stopwords), rules), rules)
    assert any(s.startswith("Termos sensacionalistas") for s in signals)
    assert "Uso excessivo de letras maiúsculas" in signals

    neutral = "O governo anunciou uma nova medida para a saúde"
    assert describe_style_signals(extract_features(preprocess_text(neutral, rules.stopwords), rules), rules) == []

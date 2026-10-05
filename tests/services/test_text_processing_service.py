from app.services.text_processing_service import (
    extract_domain, extract_features, normalize_text, preprocess_text,
)


def test_normalize_text_removes_accents_urls_and_spaces():
    assert normalize_text("  Atenção!   Veja https://x.com/a  AGORA ") == "atencao! veja agora"


def test_preprocess_text_removes_stopwords(rules):
    processed = preprocess_text("O governo anunciou uma nova medida para a saúde", rules.stopwords)
    assert "o" in processed.tokens
    assert "o" not in processed.content_tokens
    assert "saude" in processed.content_tokens


def test_extract_domain():
    assert extract_domain("https://www.g1.globo.com/noticia", None) == "g1.globo.com"
    assert extract_domain(None, "folha.uol.com.br") == "folha.uol.com.br"
    assert extract_domain(None, "Jornal da Cidade") is None
    assert extract_domain(None, None) is None


def test_extract_features_detects_sensationalism(rules):
    text = "URGENTE!!! Compartilhe antes que apaguem: a mídia esconde a CURA DEFINITIVA!"
    features = extract_features(preprocess_text(text, rules.stopwords), rules)
    assert "urgente" in features.sensational_terms
    assert "antes que apaguem" in features.sensational_terms
    assert features.repeated_punctuation_count >= 1
    assert features.uppercase_ratio > 0.3


def test_trusted_source_matches_subdomain(rules):
    features = extract_features(
        preprocess_text("texto", rules.stopwords), rules, url="https://noticias.uol.com.br/x"
    )
    assert features.trusted_source is True

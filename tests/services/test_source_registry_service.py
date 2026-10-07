def test_find_matches_subdomain_and_most_specific_section(registry):
    assert registry.find("https://noticias.uol.com.br/x").key == "uol"
    assert registry.find("https://noticias.uol.com.br/confere/x").key == "uol_confere"
    assert registry.find("https://g1.globo.com/fato-ou-fake/noticia/x").category == "fact_checker"
    assert registry.find("https://g1.globo.com/saude/x").category == "news_outlet"
    assert registry.find("https://www.ibge.gov.br/x").key == "ibge"
    assert registry.find("https://blog.net/x") is None


def test_assess_domain_informed_by_user(registry):
    assessment = registry.assess(None, "folha.uol.com.br")
    assert assessment.recognized is True
    assert assessment.name == "Folha de S.Paulo"

    unknown = registry.assess("https://www.blogqualquer.net/x", None)
    assert unknown.domain == "blogqualquer.net"
    assert unknown.recognized is False

    assert registry.assess(None, "Jornal da Cidade") is None
    assert registry.assess(None, None) is None

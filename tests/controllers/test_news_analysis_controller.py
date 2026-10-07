import pytest

from app.core.dependencies import get_content_extraction_service, get_verification_service
from app.main import app
from app.services.content_extraction_service import ContentExtractionService
from tests.conftest import NEUTRAL_TEXT, SENSATIONAL_TEXT
from tests.fakes import FakeSearchService, make_outcome, make_result
from tests.services.test_content_extraction_service import PARAGRAPH, FakeWebPageClient

URL = "/api/v1/news-analyses"
TITLE = "Ivermectina tem 70% de eficácia contra a Covid, diz estudo"
LUPA_FALSE = make_result("É falso que a ivermectina tenha 70% de eficácia contra a Covid",
                         "https://www.agencialupa.org/checagem/x")


@pytest.fixture
def with_search(make_verification_service, verification_config):
    """Usa o pipeline com busca falsa; devolve o FakeSearchService para os testes ajustarem."""

    def install(results: dict, failed: tuple = ()) -> FakeSearchService:
        search = FakeSearchService(make_outcome(verification_config, results, failed))
        service = make_verification_service(search)
        app.dependency_overrides[get_verification_service] = lambda: service
        return search

    return install


def create(client, **payload):
    return client.post(URL, json={"text": SENSATIONAL_TEXT, **payload})


def test_health_and_security_headers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_docs_disabled(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_create_returns_report_with_evidences(client, with_search):
    with_search({"lupa": [LUPA_FALSE]})
    response = client.post(URL, json={"text": NEUTRAL_TEXT, "title": TITLE, "source": "blogqualquer.net"})
    assert response.status_code == 201
    body = response.json()
    assert response.headers["Location"].endswith(f"{URL}/{body['id']}")
    assert body["classification"] == "likely_false"
    assert body["classification_label"] == "Possivelmente falsa"
    assert body["claim"] == TITLE
    assert "Agência Lupa" in body["justification"]
    assert body["verification_status"] == "complete"
    assert body["source_assessment"] == {
        "domain": "blogqualquer.net", "recognized": False, "name": None, "category": None,
    }
    evidence = body["evidences"][0]
    assert evidence["url"] == LUPA_FALSE.url
    assert evidence["stance"] == "contradicts"
    assert evidence["category"] == "fact_checker"
    assert "organization" not in evidence
    # Nenhuma pontuação é apresentada como probabilidade
    assert "confidence" not in body and "probability_fake" not in body
    assert body["disclaimer"]

    stored = client.get(f"{URL}/{body['id']}").json()
    assert stored["evidences"] == body["evidences"]
    assert stored["news"]["id"] == body["news"]["id"]


def test_without_evidence_result_is_inconclusive(client, with_search):
    with_search({"lupa": []})
    body = create(client).json()
    assert body["classification"] == "inconclusive"
    assert body["style_indicators"]
    assert body["evidences"] == []


def test_search_disabled_is_inconclusive(client):
    body = create(client).json()
    assert body["classification"] == "inconclusive"
    assert body["verification_status"] == "skipped"


def test_create_from_url_only(client, with_search):
    with_search({"lupa": []})
    service = ContentExtractionService(FakeWebPageClient(), 10_000)
    app.dependency_overrides[get_content_extraction_service] = lambda: service
    body = client.post(URL, json={"url": "https://g1.globo.com/saude/boletim.ghtml"}).json()
    news = client.get(f"/api/v1/news/{body['news']['id']}").json()
    assert news["text"] == PARAGRAPH
    assert news["title"] == "Boletim de vacinação é divulgado"
    assert news["published_at"] == "2026-10-05"


def test_url_that_cannot_be_read_returns_422(client):
    service = ContentExtractionService(None, 10_000)
    app.dependency_overrides[get_content_extraction_service] = lambda: service
    response = client.post(URL, json={"url": "http://127.0.0.1/admin"})
    assert response.status_code == 422
    assert client.get(URL).json()["total"] == 0


def test_create_requires_text_or_url(client):
    response = client.post(URL, json={"title": "Só título"})
    assert response.status_code == 422
    assert response.json()["errors"][0]["message"] == "Informe o texto ou a URL da notícia."


def test_create_rejects_blank_text(client):
    for text in ["", "    "]:
        response = client.post(URL, json={"text": text})
        assert response.status_code == 422
        assert response.json()["errors"][0]["field"] == "text"
    assert client.get(URL).json()["total"] == 0  # RN08: nada foi analisado/salvo


def test_create_rejects_client_controlled_fields(client):
    response = client.post(URL, json={"text": SENSATIONAL_TEXT, "classification": "likely_true"})
    assert response.status_code == 422
    assert response.json()["errors"] == [{"field": "classification", "message": "Campo não permitido."}]


def test_validation_messages_are_in_portuguese(client):
    errors = client.post(URL, json={"text": "texto", "url": "nao-e-url"}).json()["errors"]
    assert {e["field"]: e["message"] for e in errors}["url"] == "URL inválida."
    errors = client.get(URL, params={"page_size": 1000}).json()["errors"]
    assert errors[0]["message"] == "Deve ser menor ou igual a 100."


def test_create_rejects_future_date(client):
    assert create(client, published_at="2999-01-01").status_code == 422


def test_list_paginates_and_filters(client, with_search):
    with_search({"lupa": [LUPA_FALSE]})
    for _ in range(3):
        client.post(URL, json={"text": NEUTRAL_TEXT, "title": TITLE})
    with_search({"lupa": []})
    create(client)

    page = client.get(URL, params={"page_size": 2, "classification": "likely_false"}).json()
    assert page["total"] == 3
    assert page["pages"] == 2
    assert len(page["items"]) == 2
    assert "evidences" not in page["items"][0]
    assert page["items"][0]["news"]["title"] == TITLE


def test_list_rejects_invalid_params(client):
    assert client.get(URL, params={"page_size": 1000}).status_code == 422
    assert client.get(URL, params={"order_by": "classification"}).status_code == 422
    assert client.get(URL, params={"unknown": "x"}).status_code == 422


def test_get_missing_returns_404(client):
    response = client.get(f"{URL}/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Análise não encontrada."

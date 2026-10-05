from app.core.dependencies import get_news_analysis_service
from app.main import app
from app.ml.news_classifier import NewsClassifier
from app.services.news_analysis_service import NewsAnalysisService
from tests.conftest import ADMIN_HEADERS, NEUTRAL_TEXT, SENSATIONAL_TEXT
from tests.services.test_news_analysis_service import (
    FakeVerificationService, make_match, make_verification,
)

URL = "/api/v1/news"


def create(client, **payload):
    return client.post(URL, json={"text": SENSATIONAL_TEXT, **payload})


def test_health_and_security_headers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_health_reports_database_down(client):
    from sqlalchemy.exc import OperationalError

    from app.database.session import get_db

    class BrokenSession:
        def execute(self, *args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    app.dependency_overrides[get_db] = lambda: BrokenSession()
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "unavailable"}


def test_docs_disabled(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_create_analyzes_and_persists(client):
    response = create(client, title="Bomba", source="blogqualquer.net")
    assert response.status_code == 201
    body = response.json()
    assert body["classification"] == "likely_false"
    assert body["classification_label"] == "Possivelmente falsa"
    assert 0 <= body["confidence"] <= 1
    assert body["disclaimer"]
    assert response.headers["Location"].endswith(f"{URL}/{body['id']}")

    detail = client.get(f"{URL}/{body['id']}").json()
    assert detail["source"] == "blogqualquer.net"
    assert detail["features"]["sensational_terms"]


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
    errors = client.post(URL, json={"url": "nao-e-url"}).json()["errors"]
    messages = {e["field"]: e["message"] for e in errors}
    assert messages["text"] == "Campo obrigatório."
    assert messages["url"] == "URL inválida."
    errors = client.get(URL, params={"page_size": 1000}).json()["errors"]
    assert errors[0]["message"] == "Deve ser menor ou igual a 100."


def test_create_rejects_invalid_url_and_future_date(client):
    assert create(client, url="not-a-url").status_code == 422
    assert create(client, published_at="2999-01-01").status_code == 422


def test_list_paginates_and_filters(client):
    for _ in range(3):
        create(client)
    client.post(URL, json={"text": NEUTRAL_TEXT, "url": "https://g1.globo.com/x"})

    page = client.get(URL, params={"page_size": 2, "classification": "likely_false"}).json()
    assert page["total"] == 3
    assert page["pages"] == 2
    assert len(page["items"]) == 2
    assert "text" not in page["items"][0]


def test_list_rejects_invalid_params(client):
    assert client.get(URL, params={"page_size": 1000}).status_code == 422
    assert client.get(URL, params={"order_by": "text"}).status_code == 422
    assert client.get(URL, params={"unknown": "x"}).status_code == 422


def test_get_missing_returns_404(client):
    response = client.get(f"{URL}/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Notícia não encontrada."


def test_update_requires_admin_key(client):
    news_id = create(client).json()["id"]
    assert client.patch(f"{URL}/{news_id}", json={"title": "x"}).status_code == 403
    assert client.patch(f"{URL}/{news_id}", json={"title": "x"},
                        headers={"X-API-Key": "wrong"}).status_code == 403


def test_update_reanalyzes(client):
    news_id = create(client).json()["id"]
    response = client.patch(
        f"{URL}/{news_id}",
        json={"text": NEUTRAL_TEXT, "url": "https://g1.globo.com/x"},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["classification"] == "likely_true"


def test_update_rejects_empty_payload_and_null_text(client):
    news_id = create(client).json()["id"]
    assert client.patch(f"{URL}/{news_id}", json={}, headers=ADMIN_HEADERS).status_code == 422
    assert client.patch(f"{URL}/{news_id}", json={"text": None}, headers=ADMIN_HEADERS).status_code == 422


def test_delete(client):
    news_id = create(client).json()["id"]
    assert client.delete(f"{URL}/{news_id}").status_code == 403
    assert client.delete(f"{URL}/{news_id}", headers=ADMIN_HEADERS).status_code == 204
    assert client.get(f"{URL}/{news_id}").status_code == 404


def test_create_returns_and_persists_verification(client, settings, rules):
    verification = make_verification(make_match("fact_checker", "false"))
    service = NewsAnalysisService(settings, rules, NewsClassifier(), FakeVerificationService(verification))
    app.dependency_overrides[get_news_analysis_service] = lambda: service

    body = client.post(URL, json={"text": NEUTRAL_TEXT}).json()
    assert body["classification"] == "likely_false"
    assert body["verification"]["matches"][0]["verdict"] == "false"
    assert body["verification"]["matches"][0]["url"] == "https://x.org/1"
    assert "source_key" not in body["verification"]["matches"][0]

    stored = client.get(f"{URL}/{body['id']}").json()
    assert stored["verification"]["sources_checked"] == ["Fonte X"]


def test_verification_is_null_when_disabled(client):
    assert create(client).json()["verification"] is None

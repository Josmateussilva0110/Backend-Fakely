from app.database.session import get_db
from app.main import app
from tests.conftest import ADMIN_HEADERS, NEUTRAL_TEXT, SENSATIONAL_TEXT

URL = "/api/v1/news"


def create_news(client, **payload) -> int:
    body = client.post("/api/v1/news-analyses", json={"text": SENSATIONAL_TEXT, **payload}).json()
    return body["news"]["id"]


def test_health_reports_database_down(client):
    from sqlalchemy.exc import OperationalError

    class BrokenSession:
        def execute(self, *args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    app.dependency_overrides[get_db] = lambda: BrokenSession()
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "unavailable"}


def test_post_on_news_is_not_allowed(client):
    # A criação acontece via POST /news-analyses
    assert client.post(URL, json={"text": NEUTRAL_TEXT}).status_code == 405


def test_list_and_detail(client):
    news_id = create_news(client, source="Blog.net")
    create_news(client)
    page = client.get(URL, params={"page_size": 1, "source": "blog.net"}).json()
    assert page["total"] == 1
    assert "text" not in page["items"][0]
    assert client.get(f"{URL}/{news_id}").json()["text"] == SENSATIONAL_TEXT


def test_list_rejects_invalid_params(client):
    assert client.get(URL, params={"order_by": "text"}).status_code == 422
    assert client.get(URL, params={"created_from": "2026-10-02T00:00:00Z",
                                   "created_to": "2026-10-01T00:00:00Z"}).status_code == 422


def test_get_missing_returns_404(client):
    response = client.get(f"{URL}/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Notícia não encontrada."


def test_update_requires_admin_key(client):
    news_id = create_news(client)
    assert client.patch(f"{URL}/{news_id}", json={"title": "x"}).status_code == 403
    assert client.patch(f"{URL}/{news_id}", json={"title": "x"},
                        headers={"X-API-Key": "wrong"}).status_code == 403


def test_update_keeps_previous_analyses(client):
    news_id = create_news(client)
    response = client.patch(f"{URL}/{news_id}", json={"text": NEUTRAL_TEXT}, headers=ADMIN_HEADERS)
    assert response.status_code == 200
    assert response.json()["text"] == NEUTRAL_TEXT
    assert client.get(f"{URL}/{news_id}/analyses").json()["total"] == 1


def test_update_rejects_empty_payload_and_null_text(client):
    news_id = create_news(client)
    assert client.patch(f"{URL}/{news_id}", json={}, headers=ADMIN_HEADERS).status_code == 422
    assert client.patch(f"{URL}/{news_id}", json={"text": None}, headers=ADMIN_HEADERS).status_code == 422


def test_reanalysis_creates_new_analysis(client):
    news_id = create_news(client)
    response = client.post(f"{URL}/{news_id}/analyses")
    assert response.status_code == 201
    assert response.headers["Location"].endswith(f"/news-analyses/{response.json()['id']}")
    analyses = client.get(f"{URL}/{news_id}/analyses").json()
    assert analyses["total"] == 2
    assert client.post(f"{URL}/999/analyses").status_code == 404
    assert client.get(f"{URL}/999/analyses").status_code == 404


def test_delete_removes_news_and_analyses(client):
    news_id = create_news(client)
    analysis_id = client.get(f"{URL}/{news_id}/analyses").json()["items"][0]["id"]
    assert client.delete(f"{URL}/{news_id}").status_code == 403
    assert client.delete(f"{URL}/{news_id}", headers=ADMIN_HEADERS).status_code == 204
    assert client.get(f"{URL}/{news_id}").status_code == 404
    assert client.get(f"/api/v1/news-analyses/{analysis_id}").status_code == 404

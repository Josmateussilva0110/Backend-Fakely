"""Testa a API rodando de ponta a ponta (inclui consultas reais às fontes de verificação).

Uso (Linux, macOS ou Windows, com o venv ativo):
    python scripts/smoke_test.py [BASE_URL]        (padrão: http://localhost:8000)

Faz até 8 análises; o limite padrão é 10 por minuto (RATE_LIMIT_CREATE).
A ADMIN_API_KEY é lida do ambiente ou do .env, para testar PATCH/DELETE.
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT_DIR = Path(__file__).resolve().parent.parent
TIMEOUT_SECONDS = 60

# Consoles do Windows podem usar outra codificação; garante a saída dos acentos
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class SmokeTest:
    def __init__(self, base_url: str, admin_key: str | None):
        self.base_url = base_url.rstrip("/")
        self.api = f"{self.base_url}/api/v1"
        self.admin_key = admin_key
        self.client = httpx.Client(timeout=TIMEOUT_SECONDS)
        self.passed = 0
        self.failed = 0
        self.body: dict = {}
        self.analysis_id = self.news_id = None

    def request(self, method: str, path: str, json: dict | None = None, headers: dict | None = None) -> int:
        url = path if path.startswith("http") else f"{self.api}{path}"
        response = self.client.request(method, url, json=json, headers=headers)
        try:
            self.body = response.json()
        except ValueError:
            self.body = {}
        return response.status_code

    def get(self, field: str):
        """Lê um campo do último corpo, ex.: 'evidences.0.title'."""
        value = self.body
        for part in field.split("."):
            if isinstance(value, list):
                value = value[int(part)] if int(part) < len(value) else None
            elif isinstance(value, dict):
                value = value.get(part)
            if value is None:
                return None
        return value

    def check(self, name: str, status: int, expected_status: int, expected_classification: str | None = None):
        classification = self.get("classification")
        ok = status == expected_status and (
            expected_classification is None or classification == expected_classification
        )
        if ok:
            self.passed += 1
            print(f"  PASSOU  {name}")
            return
        self.failed += 1
        detail = f"status {status}, esperado {expected_status}"
        if expected_classification:
            detail += f'; classificação "{classification}", esperado "{expected_classification}"'
        print(f"  FALHOU  {name} ({detail})")

    def show_analysis(self) -> None:
        print(f"          → {self.get('classification_label')} (verificação: {self.get('verification_status')})")
        if claim := self.get("claim"):
            print(f"          → afirmação: {claim[:90]}")
        if evidence := self.get("evidences.0.title"):
            print(f"          → evidência: {self.get('evidences.0.source_name')}: {evidence[:90]}")

    def analyze(self, name: str, payload: dict, expected_classification: str) -> None:
        self.check(name, self.request("POST", "/news-analyses", payload), 201, expected_classification)
        self.show_analysis()

    def run(self) -> bool:
        print(f"API: {self.base_url}")
        if not self.ensure_api_is_up():
            return False
        self.run_analyses()
        self.run_validations()
        self.run_queries()
        self.run_security()
        print(f"\nResultado: {self.passed} passaram, {self.failed} falharam")
        return self.failed == 0

    def ensure_api_is_up(self) -> bool:
        try:
            status = self.request("GET", f"{self.base_url}/health")
        except httpx.HTTPError:
            print(f"A API não respondeu em {self.base_url}/health. Ela está rodando?")
            return False
        if status == 503:
            print("A API está no ar, mas o banco de dados está indisponível.")
            print("Suba o PostgreSQL com: docker compose up -d db   (e depois: alembic upgrade head)")
            return False
        if status != 200:
            print(f"A API respondeu {status} em /health.")
            return False
        return True

    def run_analyses(self) -> None:
        print("\n== Análises (consultam as fontes reais; o resultado pode variar com o conteúdo delas)")
        self.analyze("Alegação falsa checada pela Lupa", {
            "title": "Ivermectina tem 70% de eficácia contra a Covid",
            "text": "Um estudo mostra que a ivermectina tem 70% de eficácia contra a Covid. Compartilhe com todos!",
        }, "likely_false")
        self.analysis_id, self.news_id = self.get("id"), self.get("news.id")

        self.analyze("Notícia publicada na Folha", {
            "title": "Quatro em cada dez brasileiros usaram remédios sem eficácia contra a Covid, diz estudo",
            "text": "Levantamento mostra que 40% dos brasileiros usaram medicamentos sem eficácia "
                    "comprovada contra a Covid durante a pandemia.",
        }, "likely_true")
        self.analyze("Sensacionalista sem evidências não vira falsa", {
            "title": "BOMBA!!!",
            "text": "URGENTE!!! COMPARTILHE ANTES QUE APAGUEM! A mídia esconde a cura definitiva que eles "
                    "não querem que você saiba. Alerta: vão proibir o remédio milagroso!",
            "source": "blogqualquer.net",
        }, "inconclusive")
        self.analyze("Site conhecido, sozinho, não vira verdadeira", {
            "text": "O Ministério da Saúde divulgou nesta segunda-feira o boletim semanal com os dados de vacinação.",
            "url": "https://g1.globo.com/saude/noticia/boletim.ghtml",
        }, "inconclusive")
        status = self.request("POST", "/news-analyses", {"text": "Isso é falso"})
        self.check("Texto curto (RN07)", status, 201, "inconclusive")

    def run_validations(self) -> None:
        print("\n== Validações (devem retornar 422)")
        valid_text = "texto com palavras suficientes aqui"
        cases = [
            ("Texto vazio (RN02)", {"text": "    "}),
            ("Sem texto e sem URL", {}),
            ("URL interna bloqueada (SSRF)", {"url": "http://169.254.169.254/latest/meta-data"}),
            ("URL inválida", {"text": valid_text, "url": "nao-e-url"}),
            ("Data no futuro", {"text": valid_text, "published_at": "2999-01-01"}),
            ("Campo controlado pelo servidor", {"text": valid_text, "classification": "likely_true"}),
        ]
        for name, payload in cases:
            self.check(name, self.request("POST", "/news-analyses", payload), 422)
        print(f"          → mensagem: {self.get('errors.0.message')}")

    def run_queries(self) -> None:
        print("\n== Consultas")
        self.check("Listagem de análises", self.request("GET", "/news-analyses?page=1&page_size=5"), 200)
        print(f"          → total: {self.get('total')} | páginas: {self.get('pages')}")
        self.check("Filtro por classificação", self.request("GET", "/news-analyses?classification=likely_false"), 200)
        if self.analysis_id:
            self.check(f"Detalhe da análise {self.analysis_id}",
                       self.request("GET", f"/news-analyses/{self.analysis_id}"), 200)
            self.check(f"Análises da notícia {self.news_id}",
                       self.request("GET", f"/news/{self.news_id}/analyses"), 200)
        self.check("Listagem de notícias", self.request("GET", "/news?page=1&page_size=5"), 200)
        self.check("Análise inexistente", self.request("GET", "/news-analyses/999999999"), 404)
        self.check("page_size acima do máximo", self.request("GET", "/news-analyses?page_size=1000"), 422)
        self.check("Ordenação fora da lista branca", self.request("GET", "/news-analyses?order_by=text"), 422)
        self.check("Parâmetro desconhecido", self.request("GET", "/news?campo=x"), 422)

    def run_security(self) -> None:
        print("\n== Segurança")
        response = self.client.get(f"{self.base_url}/health")
        has_headers = response.headers.get("x-content-type-options") == "nosniff"
        self.check("Headers de segurança presentes", 200 if has_headers else 0, 200)
        if self.news_id:
            self.check("PATCH sem chave", self.request("PATCH", f"/news/{self.news_id}", {"title": "x"}), 403)
            self.check("DELETE com chave errada", self.request(
                "DELETE", f"/news/{self.news_id}", headers={"X-API-Key": "chave-errada"}), 403)
        if not self.admin_key:
            print("  (ADMIN_API_KEY não definida: testes de PATCH/DELETE com chave ignorados)")
            return
        headers = {"X-API-Key": self.admin_key}
        self.request("POST", "/news-analyses",
                     {"text": "Notícia temporária criada pelo smoke test para validar edição e remoção."})
        temp_id = self.get("news.id")
        self.check("PATCH com chave", self.request(
            "PATCH", f"/news/{temp_id}", {"title": "Título editado"}, headers), 200)
        self.check("DELETE com chave", self.request("DELETE", f"/news/{temp_id}", headers=headers), 204)
        self.check("Removida não existe mais", self.request("GET", f"/news/{temp_id}"), 404)


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BASE_URL", "http://localhost:8000")
    # Chave admin lida do ambiente ou do .env, sem exibi-la
    admin_key = os.environ.get("ADMIN_API_KEY") or dotenv_values(ROOT_DIR / ".env").get("ADMIN_API_KEY")
    sys.exit(0 if SmokeTest(base_url, admin_key or None).run() else 1)


if __name__ == "__main__":
    main()

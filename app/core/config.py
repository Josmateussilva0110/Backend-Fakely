from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    app_name: str = "Fake News Detection API"
    environment: Literal["development", "test", "production"] = "development"
    api_prefix: str = "/api/v1"
    # 127.0.0.1 por padrão: só aceita conexões da própria máquina
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    # Documentação fechada por padrão; habilite só em desenvolvimento
    docs_enabled: bool = False
    cors_origins: list[str] = []

    database_url: str

    # Chave para operações administrativas (PATCH/DELETE); sem chave, elas ficam desabilitadas
    admin_api_key: SecretStr | None = None

    rate_limit_enabled: bool = True
    rate_limit_create: str = "10/minute"
    rate_limit_storage_uri: str = "memory://"

    analysis_rules_path: Path = BASE_DIR / "app" / "data" / "analysis_rules.json"
    source_registry_path: Path = BASE_DIR / "app" / "data" / "source_registry.json"
    # Identifica a versão das regras de decisão gravada em cada análise (auditoria)
    analysis_version: str = "evidence-rules-1"
    max_text_length: int = 100_000
    # Abaixo deste número de palavras não há afirmação verificável (RN07)
    min_words: int = 5
    claim_max_length: int = 500

    # Busca de evidências (Lupa, Boatos.org, Folha e, com chave, Google Fact Check)
    verification_enabled: bool = True
    verification_sources_path: Path = BASE_DIR / "app" / "data" / "verification_sources.json"
    verification_user_agent: str = "FakeNewsDetectionMVP/1.0 (projeto academico)"
    verification_timeout_seconds: float = 5.0
    verification_total_timeout_seconds: float = 12.0
    verification_max_response_bytes: int = 2_000_000
    verification_max_results: int = 5
    verification_max_keywords: int = 4
    verification_min_keywords: int = 2
    google_fact_check_api_key: SecretStr | None = None
    google_fact_check_language_code: str = "pt-BR"

    # Evidências: relevância mínima para listar e para pesar na decisão
    evidence_min_relevance: float = 0.5
    evidence_min_overlap: int = 2
    evidence_excerpt_max_length: int = 300
    evidence_outdated_after_days: int = 730
    decision_min_relevance: float = 0.7
    # Suficiência: fontes independentes necessárias para concluir
    decision_min_supporting_sources: int = 1
    decision_min_contradicting_sources: int = 1

    # Leitura da página quando o usuário envia só a URL (com proteção contra SSRF)
    page_fetch_enabled: bool = True
    page_fetch_timeout_seconds: float = 8.0
    page_fetch_max_bytes: int = 3_000_000
    page_fetch_max_redirects: int = 3

    default_page_size: int = 20
    max_page_size: int = 100

    analysis_cache_ttl_seconds: int = 3600
    analysis_cache_max_entries: int = 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

import os

# Configuração isolada para testes (antes de importar a aplicação)
os.environ.update({
    "ENVIRONMENT": "test",
    "DATABASE_URL": "sqlite://",
    "RATE_LIMIT_ENABLED": "false",
    "ADMIN_API_KEY": "test-admin-key",
    "DOCS_ENABLED": "false",
    # Testes nunca acessam a internet; a verificação usa clientes falsos
    "VERIFICATION_ENABLED": "false",
    "PAGE_FETCH_ENABLED": "false",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.database.session import Base, enable_sqlite_foreign_keys, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import evidence_model, news_analysis_model, news_model  # noqa: E402,F401
from app.services.analysis_rules_service import load_analysis_rules  # noqa: E402
from app.services.evidence_service import EvidenceService  # noqa: E402
from app.services.source_registry_service import load_source_registry  # noqa: E402
from app.services.verification_config_service import load_verification_config  # noqa: E402
from app.services.verification_service import VerificationService  # noqa: E402

ADMIN_HEADERS = {"X-API-Key": "test-admin-key"}

NEUTRAL_TEXT = (
    "O Ministério da Saúde divulgou nesta segunda-feira o boletim semanal com os dados "
    "de vacinação. Segundo o órgão, a cobertura vacinal aumentou em relação ao mês anterior."
)
SENSATIONAL_TEXT = (
    "URGENTE!!! COMPARTILHE ANTES QUE APAGUEM! A mídia esconde a cura definitiva que "
    "eles não querem que você saiba. Alerta: vão proibir o remédio milagroso!"
)


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def rules(settings):
    return load_analysis_rules(settings.analysis_rules_path)


@pytest.fixture
def registry(settings):
    return load_source_registry(settings.source_registry_path)


@pytest.fixture
def verification_config(settings):
    return load_verification_config(settings.verification_sources_path)


@pytest.fixture
def evidence_service(settings, verification_config, registry, rules):
    return EvidenceService(settings, verification_config, registry, rules.stopwords)


@pytest.fixture
def make_verification_service(settings, rules, registry, evidence_service):
    """Pipeline com um serviço de busca falso (testes nunca acessam a internet)."""

    def build(search_service=None, **settings_overrides) -> VerificationService:
        return VerificationService(
            settings.model_copy(update=settings_overrides), rules, registry,
            search_service=search_service,
            evidence_service=evidence_service if search_service else None,
        )

    return build


@pytest.fixture
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    enable_sqlite_foreign_keys(engine)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    yield session
    session.close()


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

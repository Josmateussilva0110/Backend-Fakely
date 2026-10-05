import os

# Configuração isolada para testes (antes de importar a aplicação)
os.environ.update({
    "ENVIRONMENT": "test",
    "DATABASE_URL": "sqlite://",
    "MODEL_PATH": "/nonexistent/model.joblib",
    "RATE_LIMIT_ENABLED": "false",
    "ADMIN_API_KEY": "test-admin-key",
    "DOCS_ENABLED": "false",
    # Testes nunca acessam a internet; a verificação usa clientes falsos
    "VERIFICATION_ENABLED": "false",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.database.session import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.ml.news_classifier import NewsClassifier  # noqa: E402
from app.models import news_model  # noqa: E402,F401
from app.services.analysis_rules_service import load_analysis_rules  # noqa: E402
from app.services.news_analysis_service import NewsAnalysisService  # noqa: E402

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
def analysis_service(settings, rules):
    return NewsAnalysisService(settings, rules, NewsClassifier())


@pytest.fixture
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
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

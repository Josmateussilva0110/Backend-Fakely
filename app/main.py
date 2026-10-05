import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.controllers import news_controller
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.security import add_security_headers, limiter
from app.database.session import get_db
from app.ml.news_classifier import NewsClassifier
from app.services.analysis_rules_service import AnalysisRules, load_analysis_rules
from app.services.news_analysis_service import NewsAnalysisService
from app.services.source_verification_service import SourceVerificationService
from app.services.verification_config_service import load_verification_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger("app")


def log_startup_banner(settings: Settings, analysis_service: NewsAnalysisService) -> None:
    base_url = f"http://{settings.host}:{settings.port}"
    enabled = lambda flag: "ativada" if flag else "desativada"  # noqa: E731
    lines = [
        f"{settings.app_name} rodando",
        f"Endereço:     {base_url}",
        f"Porta:        {settings.port}",
        f"API:          {base_url}{settings.api_prefix}",
        f"Docs:         {base_url}/docs" if settings.docs_enabled else "Docs:         desativada",
        f"Ambiente:     {settings.environment}",
        f"Verificação:  {enabled(analysis_service.verification_service is not None)}",
        f"Modelo de ML: {'treinado' if analysis_service.classifier.is_trained else 'não treinado (só heurística)'}",
    ]
    width = max(len(line) for line in lines) + 4
    logger.info("=" * width)
    for line in lines:
        logger.info(f"  {line}")
    logger.info("=" * width)


def build_verification_service(
    settings: Settings, rules: AnalysisRules, http_client: httpx.Client
) -> SourceVerificationService | None:
    if not settings.verification_enabled:
        return None
    return SourceVerificationService.build(
        settings, load_verification_config(settings.verification_sources_path), rules.stopwords, http_client
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    rules = load_analysis_rules(settings.analysis_rules_path)
    # Recursos pesados e conexões HTTP criados uma única vez e reutilizados
    http_client = httpx.Client(
        timeout=settings.verification_timeout_seconds,
        follow_redirects=True,
        max_redirects=3,
        headers={"User-Agent": settings.verification_user_agent},
    )
    verification_service = build_verification_service(settings, rules, http_client)
    app.state.news_analysis_service = NewsAnalysisService(
        settings=settings,
        rules=rules,
        classifier=NewsClassifier.load(settings.model_path),
        verification_service=verification_service,
    )
    log_startup_banner(settings, app.state.news_analysis_service)
    yield
    if verification_service is not None:
        verification_service.close()
    http_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.1.0",
        lifespan=lifespan,
        debug=False,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.limiter = limiter

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Content-Type", "X-API-Key"],
        )
    add_security_headers(app)
    register_exception_handlers(app)

    @app.get("/health", tags=["status"])
    def health(db: Session = Depends(get_db)):
        try:
            db.execute(text("SELECT 1"))
        except SQLAlchemyError:
            logger.exception("Banco de dados indisponível")
            return JSONResponse(status_code=503, content={"status": "unavailable", "database": "unavailable"})
        return {"status": "ok", "database": "ok"}

    app.include_router(news_controller.router, prefix=settings.api_prefix)
    return app


app = create_app()

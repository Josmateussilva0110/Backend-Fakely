import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.controllers.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.security import add_security_headers, limiter
from app.database.session import get_db
from app.integrations.web_page_client import WebPageClient
from app.services.analysis_rules_service import load_analysis_rules
from app.services.content_extraction_service import ContentExtractionService
from app.services.evidence_search_service import EvidenceSearchService
from app.services.evidence_service import EvidenceService
from app.services.source_registry_service import SourceRegistry, load_source_registry
from app.services.verification_config_service import load_verification_config
from app.services.verification_service import VerificationService

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
# O httpx registra as URLs requisitadas em INFO; mantém só avisos para não poluir nem expor parâmetros
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("app")


def log_startup_banner(settings: Settings, verification_service: VerificationService) -> None:
    base_url = f"http://{settings.host}:{settings.port}"
    search = verification_service.search_service
    lines = [
        f"{settings.app_name} rodando",
        f"Endereço:     {base_url}",
        f"Porta:        {settings.port}",
        f"API:          {base_url}{settings.api_prefix}",
        f"Docs:         {base_url}/docs" if settings.docs_enabled else "Docs:         desativada",
        f"Ambiente:     {settings.environment}",
        f"Busca:        {', '.join(search.provider_names) if search else 'desativada'}",
        f"Leitura URL:  {'ativada' if settings.page_fetch_enabled else 'desativada'}",
    ]
    width = max(len(line) for line in lines) + 4
    logger.info("=" * width)
    for line in lines:
        logger.info(f"  {line}")
    logger.info("=" * width)


def build_verification_service(
    settings: Settings, registry: SourceRegistry, http_client: httpx.Client
) -> VerificationService:
    rules = load_analysis_rules(settings.analysis_rules_path)
    if not settings.verification_enabled:
        return VerificationService(settings, rules, registry)
    config = load_verification_config(settings.verification_sources_path)
    return VerificationService(
        settings,
        rules,
        registry,
        search_service=EvidenceSearchService.build(settings, config, rules.stopwords, http_client),
        evidence_service=EvidenceService(settings, config, registry, rules.stopwords),
    )


def build_content_extraction_service(settings: Settings, http_client: httpx.Client) -> ContentExtractionService:
    web_page_client = None
    if settings.page_fetch_enabled:
        web_page_client = WebPageClient(
            http_client,
            max_bytes=settings.page_fetch_max_bytes,
            max_redirects=settings.page_fetch_max_redirects,
            timeout_seconds=settings.page_fetch_timeout_seconds,
        )
    return ContentExtractionService(web_page_client, settings.max_text_length)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    registry = load_source_registry(settings.source_registry_path)
    # Recursos pesados e conexões HTTP criados uma única vez e reutilizados
    http_client = httpx.Client(
        timeout=settings.verification_timeout_seconds,
        follow_redirects=True,
        max_redirects=3,
        headers={"User-Agent": settings.verification_user_agent},
    )
    verification_service = build_verification_service(settings, registry, http_client)
    app.state.verification_service = verification_service
    app.state.content_extraction_service = build_content_extraction_service(settings, http_client)
    log_startup_banner(settings, verification_service)
    yield
    if verification_service.search_service is not None:
        verification_service.search_service.close()
    http_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="2.0.0",
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

    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.news_analysis_repository import NewsAnalysisRepository
from app.models.news_repository import NewsRepository
from app.services.content_extraction_service import ContentExtractionService
from app.services.news_analysis_service import NewsAnalysisService
from app.services.news_service import NewsService
from app.services.verification_service import VerificationService


def get_verification_service(request: Request) -> VerificationService:
    # Instância única criada no lifespan (regras, modelo e clientes HTTP carregados uma vez)
    return request.app.state.verification_service


def get_content_extraction_service(request: Request) -> ContentExtractionService:
    return request.app.state.content_extraction_service


def get_news_service(db: Session = Depends(get_db)) -> NewsService:
    return NewsService(NewsRepository(db))


def get_news_analysis_service(
    db: Session = Depends(get_db),
    verification_service: VerificationService = Depends(get_verification_service),
    content_service: ContentExtractionService = Depends(get_content_extraction_service),
) -> NewsAnalysisService:
    return NewsAnalysisService(
        NewsAnalysisRepository(db), NewsRepository(db), verification_service, content_service
    )

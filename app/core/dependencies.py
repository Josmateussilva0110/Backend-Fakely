from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.news_repository import NewsRepository
from app.services.news_analysis_service import NewsAnalysisService
from app.services.news_service import NewsService


def get_news_analysis_service(request: Request) -> NewsAnalysisService:
    # Instância única criada no lifespan (modelo e regras carregados uma vez)
    return request.app.state.news_analysis_service


def get_news_service(
    db: Session = Depends(get_db),
    analysis_service: NewsAnalysisService = Depends(get_news_analysis_service),
) -> NewsService:
    return NewsService(NewsRepository(db), analysis_service)

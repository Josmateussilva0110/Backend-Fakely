from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response, status

from app.core.config import get_settings
from app.core.dependencies import get_news_analysis_service
from app.core.security import limiter
from app.schemas.common import ErrorResponse, Page
from app.schemas.news_analysis.create import NewsAnalysisCreate
from app.schemas.news_analysis.filters import NewsAnalysisFilters
from app.schemas.news_analysis.response import NewsAnalysisResponse, NewsAnalysisSummaryResponse
from app.services.news_analysis_service import NewsAnalysisService

router = APIRouter(
    prefix="/news-analyses",
    tags=["news-analyses"],
    responses={422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)


def to_page(items, total: int, pages: int, filters: NewsAnalysisFilters) -> Page[NewsAnalysisSummaryResponse]:
    return Page[NewsAnalysisSummaryResponse](
        items=[NewsAnalysisSummaryResponse.model_validate(item) for item in items],
        total=total,
        page=filters.page,
        page_size=filters.page_size,
        pages=pages,
    )


@router.post("", response_model=NewsAnalysisResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(lambda: get_settings().rate_limit_create)
def create_news_analysis(
    request: Request,
    response: Response,
    data: NewsAnalysisCreate,
    service: NewsAnalysisService = Depends(get_news_analysis_service),
):
    analysis = service.create(data)
    response.headers["Location"] = str(request.url_for("get_news_analysis", analysis_id=analysis.id))
    return analysis


@router.get("", response_model=Page[NewsAnalysisSummaryResponse])
def list_news_analyses(
    filters: Annotated[NewsAnalysisFilters, Query()],
    service: NewsAnalysisService = Depends(get_news_analysis_service),
):
    return to_page(*service.list(filters), filters)


@router.get("/{analysis_id}", response_model=NewsAnalysisResponse, responses={404: {"model": ErrorResponse}})
def get_news_analysis(analysis_id: int, service: NewsAnalysisService = Depends(get_news_analysis_service)):
    return service.get_by_id(analysis_id)

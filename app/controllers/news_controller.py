from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response, status

from app.controllers.news_analysis_controller import to_page
from app.core.config import get_settings
from app.core.dependencies import get_news_analysis_service, get_news_service
from app.core.security import limiter, require_admin_key
from app.schemas.common import ErrorResponse, Page
from app.schemas.news.filters import NewsFilters
from app.schemas.news.response import NewsResponse, NewsSummaryResponse
from app.schemas.news.update import NewsUpdate
from app.schemas.news_analysis.filters import NewsAnalysisFilters
from app.schemas.news_analysis.response import NewsAnalysisResponse, NewsAnalysisSummaryResponse
from app.services.news_analysis_service import NewsAnalysisService
from app.services.news_service import NewsService

router = APIRouter(
    prefix="/news",
    tags=["news"],
    responses={422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)


@router.get("", response_model=Page[NewsSummaryResponse])
def list_news(
    filters: Annotated[NewsFilters, Query()],
    service: NewsService = Depends(get_news_service),
):
    items, total, pages = service.list(filters)
    return Page[NewsSummaryResponse](
        items=[NewsSummaryResponse.model_validate(item) for item in items],
        total=total,
        page=filters.page,
        page_size=filters.page_size,
        pages=pages,
    )


@router.get("/{news_id}", response_model=NewsResponse, responses={404: {"model": ErrorResponse}})
def get_news(news_id: int, service: NewsService = Depends(get_news_service)):
    return service.get_by_id(news_id)


@router.patch(
    "/{news_id}",
    response_model=NewsResponse,
    dependencies=[Depends(require_admin_key)],
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def update_news(news_id: int, data: NewsUpdate, service: NewsService = Depends(get_news_service)):
    return service.update(news_id, data)


@router.delete(
    "/{news_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin_key)],
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def delete_news(news_id: int, service: NewsService = Depends(get_news_service)):
    service.delete(news_id)


@router.get(
    "/{news_id}/analyses",
    response_model=Page[NewsAnalysisSummaryResponse],
    responses={404: {"model": ErrorResponse}},
)
def list_news_analyses_of_news(
    news_id: int,
    filters: Annotated[NewsAnalysisFilters, Query()],
    service: NewsAnalysisService = Depends(get_news_analysis_service),
):
    return to_page(*service.list(filters, news_id=news_id), filters)


@router.post(
    "/{news_id}/analyses",
    response_model=NewsAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}},
)
@limiter.limit(lambda: get_settings().rate_limit_create)
def reanalyze_news(
    request: Request,
    response: Response,
    news_id: int,
    service: NewsAnalysisService = Depends(get_news_analysis_service),
):
    analysis = service.reanalyze(news_id)
    response.headers["Location"] = str(request.url_for("get_news_analysis", analysis_id=analysis.id))
    return analysis

"""Agrupa os routers de todos os recursos."""

from fastapi import APIRouter

from app.controllers import news_analysis_controller, news_controller

api_router = APIRouter()
api_router.include_router(news_analysis_controller.router)
api_router.include_router(news_controller.router)

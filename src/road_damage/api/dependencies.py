"""FastAPI dependency providers."""

from fastapi import Request

from road_damage.api.config import BackendSettings
from road_damage.api.services.analysis_service import AnalysisService


def get_backend_settings(request: Request) -> BackendSettings:
    return request.app.state.backend_settings


def get_analysis_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service

"""Cheap liveness endpoint with no model or data access."""

from fastapi import APIRouter

from road_damage.api import API_VERSION
from road_damage.api.schemas.common import PUBLIC_ERROR_RESPONSES
from road_damage.api.schemas.health import HealthResponse


router = APIRouter(tags=["health"])


@router.get(
    "/health", response_model=HealthResponse, responses=PUBLIC_ERROR_RESPONSES
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="AI-Based Road Damage Detection API",
        api_version=API_VERSION,
    )

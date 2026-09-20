"""FastAPI application factory for secure Phase 5B video analysis."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from road_damage.api import API_PREFIX, API_VERSION
from road_damage.api.config import BackendSettings
from road_damage.api.errors import register_error_handlers
from road_damage.api.request_limits import (
    UploadRequestBodyLimitMiddleware,
    upload_request_body_limit,
)
from road_damage.api.routes import analyses, health, system
from road_damage.api.services.analysis_service import (
    AnalysisService,
    Phase5BAnalysisService,
)


def create_app(
    settings: BackendSettings | None = None,
    analysis_service: AnalysisService | None = None,
) -> FastAPI:
    """Construct an API without loading models, CUDA, datasets, or checkpoints."""
    resolved_settings = settings or BackendSettings.from_environment()
    docs_url = "/docs" if resolved_settings.docs_enabled else None
    openapi_url = "/openapi.json" if resolved_settings.docs_enabled else None
    application = FastAPI(
        title="AI-Based Road Damage Detection API",
        version=API_VERSION,
        description=(
            "Phase 5B secure video submission and background execution for "
            "the frozen Phase 4 road-damage application."
        ),
        debug=False,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
    )
    application.state.backend_settings = resolved_settings
    application.state.analysis_service = (
        analysis_service
        if analysis_service is not None
        else Phase5BAnalysisService(resolved_settings)
    )
    register_error_handlers(application)
    application.add_middleware(
        UploadRequestBodyLimitMiddleware,
        maximum_body_bytes=upload_request_body_limit(
            resolved_settings.max_upload_bytes
        ),
        upload_path=f"{API_PREFIX}/analyses",
    )

    if resolved_settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_origins),
            allow_credentials=resolved_settings.cors_allow_credentials,
            allow_methods=["GET", "POST"],
            allow_headers=["Accept", "Content-Type"],
        )

    application.include_router(health.router, prefix=API_PREFIX)
    application.include_router(system.router, prefix=API_PREFIX)
    application.include_router(analyses.router, prefix=API_PREFIX)
    return application


app = create_app()


__all__ = ["app", "create_app"]

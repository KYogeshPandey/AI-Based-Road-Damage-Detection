"""Shared, stable API contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiSchema(BaseModel):
    """Strict immutable base for response and boundary models."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CanonicalDamageClass(ApiSchema):
    class_id: int = Field(ge=0, le=3)
    class_name: str = Field(min_length=1)
    friendly_display_name: str = Field(min_length=1)


# API projection of the frozen four-class Phase 4 contract. This mapping is
# intentionally not configurable through the backend runtime environment.
CANONICAL_DAMAGE_CLASSES: tuple[CanonicalDamageClass, ...] = (
    CanonicalDamageClass(
        class_id=0,
        class_name="D00_longitudinal_crack",
        friendly_display_name="Longitudinal Crack",
    ),
    CanonicalDamageClass(
        class_id=1,
        class_name="D10_transverse_crack",
        friendly_display_name="Transverse Crack",
    ),
    CanonicalDamageClass(
        class_id=2,
        class_name="D20_alligator_crack",
        friendly_display_name="Alligator Crack",
    ),
    CanonicalDamageClass(
        class_id=3,
        class_name="D40_pothole",
        friendly_display_name="Pothole",
    ),
)


class ErrorDetail(ApiSchema):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: Any | None = None


class ErrorResponse(ApiSchema):
    error: ErrorDetail


PUBLIC_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Invalid upload request."},
    404: {"model": ErrorResponse, "description": "Requested resource was not found."},
    413: {"model": ErrorResponse, "description": "Upload exceeds the configured limit."},
    415: {"model": ErrorResponse, "description": "Unsupported uploaded media type."},
    422: {"model": ErrorResponse, "description": "Request validation failed."},
    500: {"model": ErrorResponse, "description": "Unexpected internal error."},
}


class SystemCapabilitiesResponse(ApiSchema):
    service: Literal["AI-Based Road Damage Detection API"]
    api_version: str = Field(min_length=1)
    canonical_classes: tuple[CanonicalDamageClass, ...]
    model_family: Literal["YOLOv8s"]
    operating_point_status: Literal["frozen_validation_selected"]
    phase4_outputs: tuple[str, ...]
    analysis_execution_enabled: Literal[True]
    current_phase: Literal["Phase 5B"]

"""Public, machine-independent system capabilities."""

from fastapi import APIRouter

from road_damage.api import API_VERSION
from road_damage.api.schemas.common import (
    CANONICAL_DAMAGE_CLASSES,
    PUBLIC_ERROR_RESPONSES,
    SystemCapabilitiesResponse,
)


router = APIRouter(tags=["system"])


@router.get(
    "/system",
    response_model=SystemCapabilitiesResponse,
    responses=PUBLIC_ERROR_RESPONSES,
)
def system_capabilities() -> SystemCapabilitiesResponse:
    return SystemCapabilitiesResponse(
        service="AI-Based Road Damage Detection API",
        api_version=API_VERSION,
        canonical_classes=CANONICAL_DAMAGE_CLASSES,
        model_family="YOLOv8s",
        operating_point_status="frozen_validation_selected",
        phase4_outputs=(
            "annotated_video.mp4",
            "run_manifest.json",
            "summary.json",
            "events.json",
            "frame_detections.jsonl",
            "completion.json",
        ),
        analysis_execution_enabled=False,
        current_phase="Phase 5A",
    )

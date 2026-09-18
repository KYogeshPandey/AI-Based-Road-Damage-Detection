"""Phase 5A analysis capability endpoint.

No submission or execution route is exposed in this phase.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from road_damage.api.dependencies import get_analysis_service
from road_damage.api.schemas.analysis import AnalysisExecutionCapability
from road_damage.api.schemas.common import PUBLIC_ERROR_RESPONSES
from road_damage.api.services.analysis_service import AnalysisService


router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.get(
    "/capabilities",
    response_model=AnalysisExecutionCapability,
    responses=PUBLIC_ERROR_RESPONSES,
)
def analysis_capabilities(
    service: Annotated[AnalysisService, Depends(get_analysis_service)],
) -> AnalysisExecutionCapability:
    return service.capability()

"""Secure Phase 5B video submission and job-status endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Request, UploadFile, status

from road_damage.api.dependencies import get_analysis_service
from road_damage.api.errors import InvalidUploadError
from road_damage.api.schemas.analysis import (
    AnalysisExecutionCapability,
    AnalysisJobResponse,
)
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


@router.post(
    "",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=PUBLIC_ERROR_RESPONSES,
)
async def create_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Road-video upload")],
    service: Annotated[AnalysisService, Depends(get_analysis_service)],
) -> AnalysisJobResponse:
    """Accept exactly one video file and schedule its owned staged copy."""
    try:
        form = await request.form()
        if set(form.keys()) != {"file"} or len(form.getlist("file")) != 1:
            raise InvalidUploadError("Only the multipart field 'file' is accepted.")
        job = await service.create_job(file)
    finally:
        await file.close()
        await request.close()
    background_tasks.add_task(service.execute_job, job.job_id)
    return job


@router.get(
    "/{job_id}",
    response_model=AnalysisJobResponse,
    responses=PUBLIC_ERROR_RESPONSES,
)
def get_analysis(
    job_id: UUID,
    service: Annotated[AnalysisService, Depends(get_analysis_service)],
) -> AnalysisJobResponse:
    return service.get_job(job_id)

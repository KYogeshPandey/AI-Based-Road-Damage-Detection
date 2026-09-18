"""Analysis application-service boundary.

Phase 5A defines lifecycle contracts only. This module deliberately does not
import or call the Phase 4 inference pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from road_damage.api.errors import (
    AnalysisExecutionUnavailableError,
    ResourceNotFoundError,
)
from road_damage.api.schemas.analysis import (
    AnalysisExecutionCapability,
    AnalysisJobResponse,
    AnalysisResultSummary,
    AnalysisSubmissionRequest,
    DamageEventResponse,
)


@runtime_checkable
class AnalysisService(Protocol):
    """Interface to be implemented by the Phase 5B execution layer."""

    def capability(self) -> AnalysisExecutionCapability: ...

    def create_job(self, request: AnalysisSubmissionRequest) -> AnalysisJobResponse: ...

    def get_job(self, job_id: UUID) -> AnalysisJobResponse: ...

    def get_result_summary(self, job_id: UUID) -> AnalysisResultSummary: ...

    def get_events(self, job_id: UUID) -> tuple[DamageEventResponse, ...]: ...


class AnalysisExecutionDisabledService:
    """Honest Phase 5A implementation: schemas exist, execution does not."""

    def capability(self) -> AnalysisExecutionCapability:
        return AnalysisExecutionCapability(
            submission_enabled=False,
            execution_enabled=False,
            current_phase="Phase 5A",
            message=(
                "Analysis execution is intentionally disabled in Phase 5A; "
                "no submission endpoint is exposed."
            ),
        )

    def create_job(self, request: AnalysisSubmissionRequest) -> AnalysisJobResponse:
        del request
        raise AnalysisExecutionUnavailableError()

    def get_job(self, job_id: UUID) -> AnalysisJobResponse:
        del job_id
        raise ResourceNotFoundError("Analysis job")

    def get_result_summary(self, job_id: UUID) -> AnalysisResultSummary:
        del job_id
        raise ResourceNotFoundError("Analysis result")

    def get_events(self, job_id: UUID) -> tuple[DamageEventResponse, ...]:
        del job_id
        raise ResourceNotFoundError("Analysis events")

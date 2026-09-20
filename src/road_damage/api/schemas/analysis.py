"""Analysis lifecycle and Phase 4 result-projection contracts."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, field_validator, model_validator

from road_damage.api.paths import UnsafeApiPathError, validate_public_filename
from road_damage.api.schemas.common import ApiSchema, CANONICAL_DAMAGE_CLASSES


_CLASS_NAME_BY_ID = {
    item.class_id: item.class_name for item in CANONICAL_DAMAGE_CLASSES
}
_FRIENDLY_NAME_BY_ID = {
    item.class_id: item.friendly_display_name for item in CANONICAL_DAMAGE_CLASSES
}


class AnalysisJobStatus(str, Enum):
    """Application job lifecycle, separate from Phase 4 run status."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class PipelineRunStatus(str, Enum):
    """Frozen Phase 4 output lifecycle vocabulary."""

    COMPLETED = "COMPLETED"
    FAILED_TECHNICAL = "FAILED_TECHNICAL"
    INTERRUPTED = "INTERRUPTED"


class DamageEventStatus(str, Enum):
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"


class AnalysisSubmissionRequest(ApiSchema):
    """Safe display-filename contract retained from the Phase 5A boundary."""

    input_filename: str = Field(min_length=1, max_length=255)

    @field_validator("input_filename")
    @classmethod
    def filename_only(cls, value: str) -> str:
        try:
            return validate_public_filename(value)
        except UnsafeApiPathError as exc:
            raise ValueError(str(exc)) from exc


class AnalysisProgress(ApiSchema):
    frames_processed: int = Field(ge=0)
    total_frames: int | None = Field(default=None, ge=0)
    percent_complete: float | None = Field(default=None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_frame_bounds(self) -> Self:
        if self.total_frames is not None and self.frames_processed > self.total_frames:
            raise ValueError("frames_processed cannot exceed total_frames.")
        return self


class AnalysisArtifactAvailability(ApiSchema):
    summary: bool = False
    events: bool = False
    annotated_video: bool = False
    run_manifest: bool = False


class AnalysisJobError(ApiSchema):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class AnalysisJobResponse(ApiSchema):
    job_id: UUID
    status: AnalysisJobStatus
    created_at: AwareDatetime
    updated_at: AwareDatetime
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    progress: AnalysisProgress
    input_filename: str = Field(min_length=1, max_length=255)
    availability: AnalysisArtifactAvailability
    error: AnalysisJobError | None = None

    @field_validator("input_filename")
    @classmethod
    def filename_only(cls, value: str) -> str:
        try:
            return validate_public_filename(value)
        except UnsafeApiPathError as exc:
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at.")
        if self.started_at is not None:
            if self.started_at < self.created_at:
                raise ValueError("started_at cannot precede created_at.")
            if self.started_at > self.updated_at:
                raise ValueError("started_at cannot follow updated_at.")
        if self.completed_at is not None:
            if self.started_at is None:
                raise ValueError("completed_at requires started_at.")
            if self.completed_at < self.started_at:
                raise ValueError("completed_at cannot precede started_at.")
            if self.completed_at > self.updated_at:
                raise ValueError("completed_at cannot follow updated_at.")

        if self.status is AnalysisJobStatus.QUEUED:
            if self.started_at is not None:
                raise ValueError("QUEUED jobs cannot contain started_at.")
            if self.completed_at is not None:
                raise ValueError("QUEUED jobs cannot contain completed_at.")
            if self.error is not None:
                raise ValueError("QUEUED jobs cannot contain an error.")
        elif self.status is AnalysisJobStatus.RUNNING:
            if self.started_at is None:
                raise ValueError("RUNNING jobs require started_at.")
            if self.completed_at is not None:
                raise ValueError("RUNNING jobs cannot contain completed_at.")
            if self.error is not None:
                raise ValueError("RUNNING jobs cannot contain an error.")
        elif self.status is AnalysisJobStatus.COMPLETED:
            if self.started_at is None:
                raise ValueError("COMPLETED jobs require started_at.")
            if self.completed_at is None:
                raise ValueError("COMPLETED jobs require completed_at.")
            if self.error is not None:
                raise ValueError("COMPLETED jobs cannot contain an error.")
        elif self.status is AnalysisJobStatus.FAILED:
            if self.started_at is None:
                raise ValueError("FAILED jobs require started_at.")
            if self.completed_at is None:
                raise ValueError("FAILED jobs require completed_at.")
            if self.error is None:
                raise ValueError("FAILED jobs require an error.")
        elif self.status is AnalysisJobStatus.INTERRUPTED:
            if self.started_at is None:
                raise ValueError("INTERRUPTED jobs require started_at.")
            if self.completed_at is None:
                raise ValueError("INTERRUPTED jobs require completed_at.")
        return self


class PerClassResult(ApiSchema):
    class_id: int = Field(ge=0, le=3)
    class_name: str
    raw_detection_observations: int = Field(ge=0)
    event_total: int = Field(ge=0)
    confirmed_event_total: int = Field(ge=0)
    tentative_event_total: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_class_and_counts(self) -> Self:
        if self.class_name != _CLASS_NAME_BY_ID[self.class_id]:
            raise ValueError("class_name does not match class_id.")
        if self.confirmed_event_total + self.tentative_event_total != self.event_total:
            raise ValueError("Per-class confirmed and tentative counts must sum to total.")
        return self


class AnalysisTimingSummary(ApiSchema):
    processing_elapsed_seconds: float = Field(ge=0.0)
    model_inference_seconds: float = Field(ge=0.0)
    effective_inference_fps: float = Field(ge=0.0)
    overall_processing_fps: float = Field(ge=0.0)


class AnalysisResultSummary(ApiSchema):
    run_id: str = Field(min_length=1)
    status: PipelineRunStatus
    frames_inferred: int = Field(ge=0)
    raw_detection_observation_total: int = Field(ge=0)
    temporal_event_total: int = Field(ge=0)
    confirmed_event_total: int = Field(ge=0)
    tentative_event_total: int = Field(ge=0)
    per_class: tuple[PerClassResult, ...]
    timing: AnalysisTimingSummary
    event_interpretation: Literal[
        "Heuristic temporal damage events; not ground-truth physical-world unique objects."
    ] = "Heuristic temporal damage events; not ground-truth physical-world unique objects."

    @model_validator(mode="after")
    def validate_totals(self) -> Self:
        if self.confirmed_event_total + self.tentative_event_total != self.temporal_event_total:
            raise ValueError("Confirmed and tentative event totals must sum to event total.")
        if tuple(item.class_id for item in self.per_class) != (0, 1, 2, 3):
            raise ValueError("per_class must contain class IDs 0, 1, 2, 3 in order.")
        if sum(item.raw_detection_observations for item in self.per_class) != self.raw_detection_observation_total:
            raise ValueError("Per-class raw observations must sum to the raw total.")
        if sum(item.event_total for item in self.per_class) != self.temporal_event_total:
            raise ValueError("Per-class event counts must sum to the event total.")
        disposition_errors: list[str] = []
        if (
            sum(item.confirmed_event_total for item in self.per_class)
            != self.confirmed_event_total
        ):
            disposition_errors.append(
                "Per-class confirmed event counts must sum to the confirmed total."
            )
        if (
            sum(item.tentative_event_total for item in self.per_class)
            != self.tentative_event_total
        ):
            disposition_errors.append(
                "Per-class tentative event counts must sum to the tentative total."
            )
        if disposition_errors:
            raise ValueError(" ".join(disposition_errors))
        return self


class DamageEventResponse(ApiSchema):
    event_id: str = Field(pattern=r"^RD[0-9]{4,}$")
    class_id: int = Field(ge=0, le=3)
    class_name: str
    friendly_display_name: str
    status: DamageEventStatus
    first_frame: int = Field(ge=0)
    last_frame: int = Field(ge=0)
    first_timestamp_ms: int = Field(ge=0)
    last_timestamp_ms: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    observation_count: int = Field(ge=1)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    max_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("event_id")
    @classmethod
    def positive_event_sequence(cls, value: str) -> str:
        if int(value[2:]) < 1:
            raise ValueError("event_id must use a positive RD sequence.")
        return value

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.class_name != _CLASS_NAME_BY_ID[self.class_id]:
            raise ValueError("class_name does not match class_id.")
        if self.friendly_display_name != _FRIENDLY_NAME_BY_ID[self.class_id]:
            raise ValueError("friendly_display_name does not match class_id.")
        if self.last_frame < self.first_frame:
            raise ValueError("last_frame cannot precede first_frame.")
        if self.last_timestamp_ms < self.first_timestamp_ms:
            raise ValueError("last_timestamp_ms cannot precede first_timestamp_ms.")
        expected_duration = (self.last_timestamp_ms - self.first_timestamp_ms) / 1000
        if abs(self.duration_seconds - expected_duration) > 1e-9:
            raise ValueError("duration_seconds does not match event timestamps.")
        if self.mean_confidence > self.max_confidence:
            raise ValueError("mean_confidence cannot exceed max_confidence.")
        return self


class AnalysisExecutionCapability(ApiSchema):
    submission_enabled: Literal[True]
    execution_enabled: Literal[True]
    current_phase: Literal["Phase 5B"]
    message: str = Field(min_length=1)
    supported_video_extensions: tuple[str, ...]
    maximum_upload_bytes: int = Field(ge=1)

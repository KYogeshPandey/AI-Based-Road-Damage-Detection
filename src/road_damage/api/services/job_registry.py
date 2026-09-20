"""Thread-safe in-memory Phase 5B analysis-job registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import UUID

from road_damage.api.schemas.analysis import (
    AnalysisArtifactAvailability,
    AnalysisJobError,
    AnalysisJobResponse,
    AnalysisJobStatus,
    AnalysisProgress,
)


class UnknownAnalysisJobError(KeyError):
    """Raised when a job is absent from the process-local registry."""


class InvalidJobTransitionError(RuntimeError):
    """Raised when code attempts an invalid lifecycle transition."""


@dataclass(frozen=True, slots=True)
class AnalysisJobPaths:
    job_directory: Path
    input_video: Path
    analysis_directory: Path


@dataclass(frozen=True, slots=True)
class AnalysisJobRecord:
    response: AnalysisJobResponse
    paths: AnalysisJobPaths


_ALLOWED_TRANSITIONS = {
    AnalysisJobStatus.QUEUED: {
        AnalysisJobStatus.RUNNING,
        AnalysisJobStatus.FAILED,
    },
    AnalysisJobStatus.RUNNING: {
        AnalysisJobStatus.COMPLETED,
        AnalysisJobStatus.FAILED,
        AnalysisJobStatus.INTERRUPTED,
    },
    AnalysisJobStatus.COMPLETED: set(),
    AnalysisJobStatus.FAILED: set(),
    AnalysisJobStatus.INTERRUPTED: set(),
}


def _now_not_before(value: datetime) -> datetime:
    return max(datetime.now(timezone.utc), value)


class AnalysisJobRegistry:
    """Store immutable records and replace them atomically on transitions."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[UUID, AnalysisJobRecord] = {}

    def add(self, record: AnalysisJobRecord) -> None:
        with self._lock:
            job_id = record.response.job_id
            if job_id in self._records:
                raise InvalidJobTransitionError("Analysis job already exists.")
            self._records[job_id] = record

    def get(self, job_id: UUID) -> AnalysisJobRecord:
        with self._lock:
            try:
                return self._records[job_id]
            except KeyError as exc:
                raise UnknownAnalysisJobError(str(job_id)) from exc

    def transition(
        self,
        job_id: UUID,
        target: AnalysisJobStatus,
        *,
        progress: AnalysisProgress | None = None,
        availability: AnalysisArtifactAvailability | None = None,
        error: AnalysisJobError | None = None,
    ) -> AnalysisJobRecord:
        with self._lock:
            current = self.get(job_id)
            source = current.response.status
            if target not in _ALLOWED_TRANSITIONS[source]:
                raise InvalidJobTransitionError(
                    f"Invalid analysis-job transition: {source.value} -> {target.value}."
                )

            now = _now_not_before(current.response.updated_at)
            started_at = current.response.started_at
            completed_at = current.response.completed_at
            resolved_error = error
            if target is AnalysisJobStatus.RUNNING:
                started_at = now
                completed_at = None
                resolved_error = None
            elif target in {
                AnalysisJobStatus.COMPLETED,
                AnalysisJobStatus.FAILED,
                AnalysisJobStatus.INTERRUPTED,
            }:
                # QUEUED -> FAILED is reserved for a scheduling failure. It is
                # recorded as an attempted job so the existing lifecycle schema
                # remains internally consistent.
                started_at = started_at or now
                completed_at = now
                if target is AnalysisJobStatus.COMPLETED:
                    resolved_error = None
                elif target is AnalysisJobStatus.FAILED and resolved_error is None:
                    raise InvalidJobTransitionError("FAILED requires a safe job error.")

            payload = current.response.model_dump()
            payload.update(
                {
                    "status": target,
                    "updated_at": now,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "progress": progress or current.response.progress,
                    "availability": availability or current.response.availability,
                    "error": resolved_error,
                }
            )
            response = AnalysisJobResponse.model_validate(payload)
            replacement = AnalysisJobRecord(response=response, paths=current.paths)
            self._records[job_id] = replacement
            return replacement

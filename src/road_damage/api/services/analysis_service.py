"""Phase 5B upload staging and lazy Phase 4 execution service."""

from __future__ import annotations

import importlib
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from uuid import UUID, uuid4

from road_damage.api.config import BackendSettings
from road_damage.api.errors import (
    AnalysisNotFoundError,
    AnalysisResultRetrievalUnavailableError,
    ApiApplicationError,
    InvalidUploadError,
    UnsupportedVideoTypeError,
    UploadStorageError,
    UploadTooLargeError,
    VideoValidationFailedError,
)
from road_damage.api.paths import (
    UnsafeApiPathError,
    resolve_job_directory,
    validate_public_filename,
)
from road_damage.api.schemas.analysis import (
    AnalysisArtifactAvailability,
    AnalysisExecutionCapability,
    AnalysisJobError,
    AnalysisJobResponse,
    AnalysisJobStatus,
    AnalysisProgress,
    AnalysisResultSummary,
    DamageEventResponse,
)
from road_damage.api.services.completion_validation import (
    validate_completed_phase4_run,
    validate_interrupted_phase4_run,
)
from road_damage.api.services.job_registry import (
    AnalysisJobPaths,
    AnalysisJobRecord,
    AnalysisJobRegistry,
    InvalidJobTransitionError,
    UnknownAnalysisJobError,
)
from road_damage.api.services.video_validation import (
    StagedVideoValidationError,
    validate_staged_video,
)


LOGGER = logging.getLogger(__name__)
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class AsyncUpload(Protocol):
    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


AnalysisCallable = Callable[[Path, Path], Mapping[str, Any]]
AnalysisLoader = Callable[[], AnalysisCallable]
VideoValidator = Callable[[Path], Any]
JobIdFactory = Callable[[], UUID]


def _load_phase4_analysis() -> AnalysisCallable:
    """Import the frozen Phase 4 application service only for job execution."""
    module = importlib.import_module("road_damage.inference.video_analysis")
    analysis = getattr(module, "analyze_video", None)
    if not callable(analysis):
        raise RuntimeError("Phase 4 analysis entry point is unavailable.")
    return analysis


@runtime_checkable
class AnalysisService(Protocol):
    def capability(self) -> AnalysisExecutionCapability: ...

    async def create_job(self, upload: AsyncUpload) -> AnalysisJobResponse: ...

    def execute_job(self, job_id: UUID) -> None: ...

    def get_job(self, job_id: UUID) -> AnalysisJobResponse: ...

    def get_result_summary(self, job_id: UUID) -> AnalysisResultSummary: ...

    def get_events(self, job_id: UUID) -> tuple[DamageEventResponse, ...]: ...


class Phase5BAnalysisService:
    """Stage validated uploads and execute the shared Phase 4 service once."""

    def __init__(
        self,
        settings: BackendSettings,
        *,
        registry: AnalysisJobRegistry | None = None,
        analysis_loader: AnalysisLoader = _load_phase4_analysis,
        video_validator: VideoValidator = validate_staged_video,
        job_id_factory: JobIdFactory = uuid4,
    ) -> None:
        self._settings = settings
        self._registry = registry or AnalysisJobRegistry()
        self._analysis_loader = analysis_loader
        self._video_validator = video_validator
        self._job_id_factory = job_id_factory

    @property
    def registry(self) -> AnalysisJobRegistry:
        return self._registry

    def capability(self) -> AnalysisExecutionCapability:
        return AnalysisExecutionCapability(
            submission_enabled=True,
            execution_enabled=True,
            current_phase="Phase 5B",
            message="Secure video submission and background analysis execution are enabled.",
            supported_video_extensions=SUPPORTED_VIDEO_EXTENSIONS,
            maximum_upload_bytes=self._settings.max_upload_bytes,
        )

    async def create_job(self, upload: AsyncUpload) -> AnalysisJobResponse:
        """Stream one upload into a UUID-owned directory and validate it."""
        original_filename = upload.filename
        try:
            safe_filename = validate_public_filename(original_filename)  # type: ignore[arg-type]
        except UnsafeApiPathError as exc:
            raise InvalidUploadError("A safe base filename is required.") from exc
        if len(safe_filename) > 255:
            raise InvalidUploadError("The upload filename is too long.")

        suffix = Path(safe_filename).suffix.casefold()
        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            raise UnsupportedVideoTypeError()

        job_id = self._job_id_factory()
        job_directory = resolve_job_directory(self._settings.output_root, job_id)
        input_directory = job_directory / "input"
        staged_video = input_directory / f"video{suffix}"
        analysis_directory = job_directory / "analysis"
        accepted = False
        job_directory_owned = False
        try:
            job_directory.mkdir(parents=True, exist_ok=False)
            job_directory_owned = True
            input_directory.mkdir(exist_ok=False)
            byte_count = 0
            with staged_video.open("xb") as destination:
                while True:
                    chunk = await upload.read(self._settings.upload_chunk_bytes)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise InvalidUploadError()
                    byte_count += len(chunk)
                    if byte_count > self._settings.max_upload_bytes:
                        raise UploadTooLargeError(self._settings.max_upload_bytes)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if byte_count == 0:
                raise InvalidUploadError("The uploaded video is empty.")

            self._video_validator(staged_video)
            now = datetime.now(timezone.utc)
            response = AnalysisJobResponse(
                job_id=job_id,
                status=AnalysisJobStatus.QUEUED,
                created_at=now,
                updated_at=now,
                progress=AnalysisProgress(
                    frames_processed=0,
                    total_frames=None,
                    percent_complete=0.0,
                ),
                input_filename=safe_filename,
                availability=AnalysisArtifactAvailability(),
            )
            self._registry.add(
                AnalysisJobRecord(
                    response=response,
                    paths=AnalysisJobPaths(
                        job_directory=job_directory,
                        input_video=staged_video,
                        analysis_directory=analysis_directory,
                    ),
                )
            )
            accepted = True
            return response
        except ApiApplicationError:
            raise
        except StagedVideoValidationError as exc:
            raise VideoValidationFailedError() from exc
        except InvalidJobTransitionError as exc:
            LOGGER.exception("Could not register a newly reserved analysis job.")
            raise UploadStorageError() from exc
        except OSError as exc:
            LOGGER.exception("Failed to store an analysis upload.")
            raise UploadStorageError() from exc
        finally:
            if not accepted and job_directory_owned and job_directory.exists():
                self._remove_incomplete_job(job_directory, job_id)

    def execute_job(self, job_id: UUID) -> None:
        """Run one accepted job using the single shared Phase 4 entry point."""
        try:
            running = self._registry.transition(
                job_id,
                AnalysisJobStatus.RUNNING,
                progress=AnalysisProgress(
                    frames_processed=0,
                    total_frames=None,
                    percent_complete=None,
                ),
            )
        except (UnknownAnalysisJobError, InvalidJobTransitionError):
            LOGGER.exception("Could not start analysis job %s.", job_id)
            return

        try:
            analyze_video = self._analysis_loader()
            result = analyze_video(
                running.paths.input_video,
                running.paths.analysis_directory,
            )
            if not isinstance(result, Mapping):
                raise RuntimeError("Phase 4 returned an invalid result.")

            availability = self._artifact_availability(running.paths.analysis_directory)
            status = str(result.get("status", ""))
            if status == "COMPLETED":
                verified = validate_completed_phase4_run(
                    running.paths.analysis_directory,
                    running.paths.input_video,
                    result,
                )
                self._registry.transition(
                    job_id,
                    AnalysisJobStatus.COMPLETED,
                    progress=AnalysisProgress(
                        frames_processed=verified.frames_inferred,
                        total_frames=None,
                        percent_complete=100.0,
                    ),
                    availability=availability,
                )
            elif status == "INTERRUPTED":
                verified = validate_interrupted_phase4_run(
                    running.paths.analysis_directory, result
                )
                self._registry.transition(
                    job_id,
                    AnalysisJobStatus.INTERRUPTED,
                    progress=AnalysisProgress(
                        frames_processed=verified.frames_inferred,
                        total_frames=None,
                        percent_complete=None,
                    ),
                    availability=availability,
                )
            else:
                raise RuntimeError("Phase 4 did not report a completed analysis.")
        except Exception:
            LOGGER.exception("Analysis job %s failed.", job_id)
            availability = self._artifact_availability(running.paths.analysis_directory)
            try:
                self._registry.transition(
                    job_id,
                    AnalysisJobStatus.FAILED,
                    progress=AnalysisProgress(
                        frames_processed=0,
                        total_frames=None,
                        percent_complete=None,
                    ),
                    availability=availability,
                    error=AnalysisJobError(
                        code="ANALYSIS_EXECUTION_FAILED",
                        message="Video analysis failed.",
                    ),
                )
            except InvalidJobTransitionError:
                LOGGER.exception("Could not record failure for analysis job %s.", job_id)

    def get_job(self, job_id: UUID) -> AnalysisJobResponse:
        try:
            return self._registry.get(job_id).response
        except UnknownAnalysisJobError as exc:
            raise AnalysisNotFoundError() from exc

    def get_result_summary(self, job_id: UUID) -> AnalysisResultSummary:
        del job_id
        raise AnalysisResultRetrievalUnavailableError()

    def get_events(self, job_id: UUID) -> tuple[DamageEventResponse, ...]:
        del job_id
        raise AnalysisResultRetrievalUnavailableError()

    def _remove_incomplete_job(self, job_directory: Path, job_id: UUID) -> None:
        expected = resolve_job_directory(self._settings.output_root, job_id)
        if job_directory.resolve(strict=False) != expected:
            raise RuntimeError("Refusing cleanup outside the owned job directory.")
        shutil.rmtree(job_directory)

    @staticmethod
    def _artifact_availability(output: Path) -> AnalysisArtifactAvailability:
        return AnalysisArtifactAvailability(
            summary=(output / "summary.json").is_file(),
            events=(output / "events.json").is_file(),
            annotated_video=(output / "annotated_video.mp4").is_file(),
            run_manifest=(output / "run_manifest.json").is_file(),
        )

__all__ = [
    "AnalysisService",
    "Phase5BAnalysisService",
    "SUPPORTED_VIDEO_EXTENSIONS",
]

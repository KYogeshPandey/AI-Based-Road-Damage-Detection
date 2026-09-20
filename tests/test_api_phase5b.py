"""Phase 5B secure upload, lifecycle, and lazy integration regressions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from uuid import UUID, uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from road_damage.api.config import BackendSettings  # noqa: E402
from road_damage.api.schemas.analysis import (  # noqa: E402
    AnalysisArtifactAvailability,
    AnalysisJobResponse,
    AnalysisJobStatus,
    AnalysisProgress,
)
from road_damage.api.services.analysis_service import (  # noqa: E402
    Phase5BAnalysisService,
)
from road_damage.api.services.job_registry import (  # noqa: E402
    AnalysisJobPaths,
    AnalysisJobRecord,
    AnalysisJobRegistry,
    InvalidJobTransitionError,
)


class MemoryUpload:
    def __init__(self, filename: str | None, payload: bytes) -> None:
        self.filename = filename
        self._payload = payload
        self._offset = 0
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            size = len(self._payload)
        start = self._offset
        self._offset += size
        return self._payload[start : self._offset]


def create_tiny_avi(path: Path) -> bytes:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 24),
    )
    if not writer.isOpened():
        raise RuntimeError("Synthetic test video writer could not be opened.")
    try:
        for value in (0, 80, 160):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_completed_artifacts(output: Path, source: Path) -> dict[str, object]:
    """Write a minimal internally consistent frozen Phase 4 success fixture."""
    output.mkdir(parents=True, exist_ok=False)
    run_id = output.name
    input_sha = sha256_file(source)
    model_sha = "A" * 64
    frames = {"inferred": 3}
    summary: dict[str, object] = {
        "schema_version": "road_damage.video_analysis.v1.summary",
        "run_id": run_id,
        "status": "COMPLETED",
        "input_video": {"sha256": input_sha},
        "model": {"sha256": model_sha},
        "frames": frames,
        "events": {"total": 0},
        "internal_test_accessed": False,
        "training_executed": False,
        "threshold_tuning_performed": False,
    }
    events = {
        "schema_version": "road_damage.video_analysis.v1.events",
        "run_id": run_id,
        "event_count": 0,
        "events": [],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "events.json", events)
    (output / "frame_detections.jsonl").write_bytes(b"")
    (output / "annotated_video.mp4").write_bytes(b"synthetic-video")
    artifact_hashes = {
        name: sha256_file(output / name)
        for name in (
            "summary.json",
            "events.json",
            "frame_detections.jsonl",
            "annotated_video.mp4",
        )
    }
    manifest = {
        "schema_version": "road_damage.video_analysis.v1.run_manifest",
        "run_id": run_id,
        "status": "COMPLETED",
        "validation": {
            "model": {"sha256": model_sha},
            "input_video": {"sha256": input_sha},
        },
        "processing_request": {"annotated_video_enabled": True},
        "progress": {"frames_inferred": 3, "finalized_events": 0},
        "artifacts_sha256": artifact_hashes,
        "internal_test_accessed": False,
        "training_executed": False,
        "threshold_tuning_performed": False,
    }
    write_json(output / "run_manifest.json", manifest)
    completion = {
        "schema_version": "road_damage.video_analysis.v1.completion",
        "run_id": run_id,
        "status": "COMPLETED",
        "completed_utc": "2026-09-20T00:00:00+00:00",
        "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
        "artifacts_sha256": artifact_hashes,
        "frames_inferred": 3,
        "event_count": 0,
        "model_sha256": model_sha,
        "input_sha256": input_sha,
        "internal_test_accessed": False,
        "training_executed": False,
        "threshold_tuning_performed": False,
    }
    write_json(output / "completion.json", completion)
    return summary


class Phase5BServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(dir=PROJECT_ROOT / "outputs")
        self.output_root = Path(self.temporary.name) / "api"
        self.settings = BackendSettings(
            output_root=self.output_root,
            max_upload_bytes=1024,
            upload_chunk_bytes=7,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_upload_streams_in_configured_chunks_and_is_not_renamed_from_client(self) -> None:
        upload = MemoryUpload("My Road.AVI", b"0123456789")
        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            analysis_loader=Mock(side_effect=AssertionError("inference loaded")),
        )
        response = asyncio.run(service.create_job(upload))
        record = service.registry.get(response.job_id)

        self.assertEqual(response.status, AnalysisJobStatus.QUEUED)
        self.assertEqual(response.input_filename, "My Road.AVI")
        self.assertEqual(upload.read_sizes, [7, 7, 7])
        self.assertEqual(record.paths.input_video.name, "video.avi")
        self.assertEqual(record.paths.input_video.read_bytes(), b"0123456789")
        self.assertEqual(record.paths.analysis_directory.name, "analysis")
        self.assertFalse(record.paths.analysis_directory.exists())

    def test_empty_oversized_unsupported_and_unsafe_uploads_leave_no_job_directory(self) -> None:
        cases = (
            (MemoryUpload("empty.mp4", b""), 1024, "INVALID_UPLOAD"),
            (MemoryUpload("large.mp4", b"0123456789"), 4, "UPLOAD_TOO_LARGE"),
            (MemoryUpload("note.txt", b"abc"), 1024, "UNSUPPORTED_VIDEO_TYPE"),
            (MemoryUpload("../escape.mp4", b"abc"), 1024, "INVALID_UPLOAD"),
            (MemoryUpload("C:\\escape.mp4", b"abc"), 1024, "INVALID_UPLOAD"),
        )
        from road_damage.api.errors import ApiApplicationError

        for upload, maximum, code in cases:
            with self.subTest(code=code, filename=upload.filename):
                settings = BackendSettings(
                    output_root=self.output_root,
                    max_upload_bytes=maximum,
                    upload_chunk_bytes=3,
                )
                service = Phase5BAnalysisService(
                    settings, video_validator=lambda _path: None
                )
                with self.assertRaises(ApiApplicationError) as caught:
                    asyncio.run(service.create_job(upload))
                self.assertEqual(caught.exception.code, code)
                children = list(self.output_root.iterdir()) if self.output_root.exists() else []
                self.assertEqual(children, [])

    def test_invalid_video_preflight_cleans_partial_job_and_never_loads_model(self) -> None:
        from road_damage.api.errors import VideoValidationFailedError
        from road_damage.api.services.video_validation import StagedVideoValidationError

        loader = Mock(side_effect=AssertionError("inference loaded"))
        service = Phase5BAnalysisService(
            self.settings,
            video_validator=Mock(
                side_effect=StagedVideoValidationError("private technical detail")
            ),
            analysis_loader=loader,
        )
        with self.assertRaises(VideoValidationFailedError):
            asyncio.run(service.create_job(MemoryUpload("bad.mp4", b"not-video")))
        loader.assert_not_called()
        self.assertEqual(list(self.output_root.iterdir()), [])

    def test_execution_calls_phase4_once_with_only_owned_input_and_output(self) -> None:
        observed: dict[str, object] = {}
        service: Phase5BAnalysisService

        def analyze(source: Path, output: Path) -> dict[str, object]:
            observed["source"] = source
            observed["output"] = output
            observed["running"] = service.get_job(job.job_id).status
            return write_completed_artifacts(output, source)

        analyzer = Mock(side_effect=analyze)
        loader = Mock(return_value=analyzer)
        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            analysis_loader=loader,
        )
        job = asyncio.run(service.create_job(MemoryUpload("road.mp4", b"video")))
        loader.assert_not_called()
        service.execute_job(job.job_id)

        record = service.registry.get(job.job_id)
        self.assertEqual(observed["running"], AnalysisJobStatus.RUNNING)
        self.assertEqual(observed["source"], record.paths.input_video)
        self.assertEqual(observed["output"], record.paths.analysis_directory)
        loader.assert_called_once_with()
        analyzer.assert_called_once_with(
            record.paths.input_video, record.paths.analysis_directory
        )
        self.assertEqual(record.response.status, AnalysisJobStatus.COMPLETED)
        self.assertEqual(record.response.progress.frames_processed, 3)
        self.assertEqual(record.response.progress.percent_complete, 100.0)
        self.assertTrue(all(record.response.availability.model_dump().values()))

    def test_reported_completion_without_receipt_is_failed(self) -> None:
        def incomplete(_source: Path, output: Path) -> dict[str, object]:
            output.mkdir(parents=True)
            (output / "summary.json").write_text("{}", encoding="utf-8")
            return {"status": "COMPLETED", "frames": {"inferred": 1}}

        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            analysis_loader=lambda: incomplete,
        )
        job = asyncio.run(service.create_job(MemoryUpload("road.mp4", b"video")))
        service.execute_job(job.job_id)
        result = service.get_job(job.job_id)
        self.assertEqual(result.status, AnalysisJobStatus.FAILED)
        self.assertEqual(result.error.code, "ANALYSIS_EXECUTION_FAILED")
        self.assertTrue(result.availability.summary)
        self.assertFalse(result.availability.events)

    def test_execution_failure_is_safe_and_retains_accepted_input(self) -> None:
        def fail(_source: Path, _output: Path) -> dict[str, object]:
            raise RuntimeError(f"secret path {PROJECT_ROOT}")

        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            analysis_loader=lambda: fail,
        )
        job = asyncio.run(service.create_job(MemoryUpload("road.mp4", b"video")))
        service.execute_job(job.job_id)
        record = service.registry.get(job.job_id)
        self.assertEqual(record.response.status, AnalysisJobStatus.FAILED)
        self.assertEqual(record.response.error.message, "Video analysis failed.")
        self.assertNotIn(str(PROJECT_ROOT), record.response.model_dump_json())
        self.assertTrue(record.paths.input_video.is_file())

    def test_unknown_job_has_specific_safe_error(self) -> None:
        from road_damage.api.errors import AnalysisNotFoundError

        service = Phase5BAnalysisService(self.settings)
        with self.assertRaises(AnalysisNotFoundError) as caught:
            service.get_job(uuid4())
        self.assertEqual(caught.exception.code, "ANALYSIS_NOT_FOUND")

    def test_registry_rejects_terminal_state_mutation_and_does_not_leak_mutability(self) -> None:
        from datetime import datetime, timezone

        job_id = uuid4()
        now = datetime.now(timezone.utc)
        response = AnalysisJobResponse(
            job_id=job_id,
            status="QUEUED",
            created_at=now,
            updated_at=now,
            progress=AnalysisProgress(frames_processed=0),
            input_filename="road.mp4",
            availability=AnalysisArtifactAvailability(),
        )
        registry = AnalysisJobRegistry()
        registry.add(
            AnalysisJobRecord(
                response=response,
                paths=AnalysisJobPaths(Path("job"), Path("input"), Path("analysis")),
            )
        )
        registry.transition(job_id, AnalysisJobStatus.RUNNING)
        registry.transition(
            job_id,
            AnalysisJobStatus.COMPLETED,
            progress=AnalysisProgress(frames_processed=1, percent_complete=100),
        )
        with self.assertRaises(InvalidJobTransitionError):
            registry.transition(job_id, AnalysisJobStatus.FAILED)
        with self.assertRaises(Exception):
            registry.get(job_id).response.status = AnalysisJobStatus.FAILED


class Phase5BHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient
        from road_damage.api.app import create_app

        self.temporary = TemporaryDirectory(dir=PROJECT_ROOT / "outputs")
        self.output_root = Path(self.temporary.name) / "api"
        self.payload = create_tiny_avi(Path(self.temporary.name) / "tiny.avi")

        def analyze(_source: Path, output: Path) -> dict[str, object]:
            return write_completed_artifacts(output, _source)

        settings = BackendSettings(
            output_root=self.output_root,
            max_upload_bytes=len(self.payload) + 100,
            upload_chunk_bytes=13,
        )
        self.service = Phase5BAnalysisService(
            settings, analysis_loader=lambda: analyze
        )
        self.client = TestClient(
            create_app(settings, self.service), raise_server_exceptions=False
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_post_returns_queued_then_get_reports_completed_artifacts(self) -> None:
        response = self.client.post(
            "/api/v1/analyses",
            files={"file": ("road.avi", self.payload, "video/x-msvideo")},
        )
        self.assertEqual(response.status_code, 202)
        created = response.json()
        self.assertEqual(created["status"], "QUEUED")
        self.assertEqual(created["input_filename"], "road.avi")
        job_id = UUID(created["job_id"])

        status_response = self.client.get(f"/api/v1/analyses/{job_id}")
        self.assertEqual(status_response.status_code, 200)
        current = status_response.json()
        self.assertEqual(current["status"], "COMPLETED")
        self.assertEqual(current["progress"]["percent_complete"], 100.0)
        self.assertTrue(all(current["availability"].values()))
        record = self.service.registry.get(job_id)
        self.assertEqual(record.paths.input_video.parent.name, "input")
        self.assertEqual(record.paths.analysis_directory.parent.name, str(job_id))

    def test_openapi_declares_multipart_upload_and_job_lookup(self) -> None:
        schema = self.client.get("/openapi.json").json()
        operation = schema["paths"]["/api/v1/analyses"]["post"]
        self.assertIn("multipart/form-data", operation["requestBody"]["content"])
        self.assertIn("202", operation["responses"])
        self.assertIn("/api/v1/analyses/{job_id}", schema["paths"])

    def test_unknown_job_and_malformed_uuid_have_stable_errors(self) -> None:
        missing = self.client.get(f"/api/v1/analyses/{uuid4()}")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "ANALYSIS_NOT_FOUND")
        malformed = self.client.get("/api/v1/analyses/not-a-uuid")
        self.assertEqual(malformed.status_code, 422)
        self.assertEqual(malformed.json()["error"]["code"], "INVALID_REQUEST")

    def test_client_cannot_submit_scientific_or_path_controls(self) -> None:
        response = self.client.post(
            "/api/v1/analyses",
            files={"file": ("road.avi", self.payload, "video/x-msvideo")},
            data={"model": "last.pt", "confidence": "0.99", "output_path": "C:\\x"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "INVALID_UPLOAD")
        children = list(self.output_root.iterdir()) if self.output_root.exists() else []
        self.assertEqual(children, [])

    def test_real_opencv_preflight_rejects_non_video_bytes(self) -> None:
        response = self.client.post(
            "/api/v1/analyses",
            files={"file": ("fake.avi", b"not a video", "video/x-msvideo")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VIDEO_VALIDATION_FAILED")
        children = list(self.output_root.iterdir()) if self.output_root.exists() else []
        self.assertEqual(children, [])


if __name__ == "__main__":
    unittest.main()

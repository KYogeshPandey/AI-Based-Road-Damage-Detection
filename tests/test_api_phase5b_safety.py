"""Narrow Phase 5B safety regressions for receipts, ingress, and ownership."""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from road_damage.api.app import create_app  # noqa: E402
from road_damage.api.config import BackendSettings  # noqa: E402
from road_damage.api.errors import ApiApplicationError  # noqa: E402
from road_damage.api.request_limits import (  # noqa: E402
    MULTIPART_OVERHEAD_ALLOWANCE_BYTES,
    UploadRequestBodyLimitMiddleware,
    upload_request_body_limit,
)
from road_damage.api.schemas.analysis import (  # noqa: E402
    AnalysisArtifactAvailability,
    AnalysisJobResponse,
    AnalysisProgress,
)
from road_damage.api.services.analysis_service import (  # noqa: E402
    Phase5BAnalysisService,
)
from road_damage.api.services.job_registry import (  # noqa: E402
    UnknownAnalysisJobError,
)
from road_damage.api.services.video_validation import (  # noqa: E402
    StagedVideoValidationError,
)
from test_api_phase5b import (  # noqa: E402
    MemoryUpload,
    sha256_file,
    write_completed_artifacts,
    write_json,
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class CompletionReconciliationTests(unittest.TestCase):
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

    def _execute_with_mutation(self, mutation) -> object:
        def analyze(source: Path, output: Path) -> dict[str, object]:
            result = write_completed_artifacts(output, source)
            mutation(output, result)
            return result

        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            analysis_loader=lambda: analyze,
        )
        job = asyncio.run(service.create_job(MemoryUpload("road.mp4", b"video")))
        with self.assertLogs("road_damage.api.services.analysis_service", level="ERROR"):
            service.execute_job(job.job_id)
        return service.get_job(job.job_id)

    def _assert_failed(self, mutation) -> None:
        result = self._execute_with_mutation(mutation)
        self.assertEqual(result.status.value, "FAILED")
        self.assertEqual(result.error.code, "ANALYSIS_EXECUTION_FAILED")
        self.assertNotIn(str(PROJECT_ROOT), result.model_dump_json())

    def test_fully_reconciled_completion_becomes_completed(self) -> None:
        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            analysis_loader=lambda: (
                lambda source, output: write_completed_artifacts(output, source)
            ),
        )
        job = asyncio.run(service.create_job(MemoryUpload("road.mp4", b"video")))
        service.execute_job(job.job_id)
        result = service.get_job(job.job_id)
        self.assertEqual(result.status.value, "COMPLETED")
        self.assertEqual(result.progress.frames_processed, 3)

    def test_empty_completion_object_is_rejected(self) -> None:
        self._assert_failed(
            lambda output, _result: write_json(output / "completion.json", {})
        )

    def test_malformed_completion_json_is_rejected(self) -> None:
        self._assert_failed(
            lambda output, _result: (output / "completion.json").write_text(
                "{", encoding="utf-8"
            )
        )

    def test_noncompleted_receipt_status_is_rejected(self) -> None:
        def mutate(output: Path, _result: dict[str, object]) -> None:
            receipt = load_json(output / "completion.json")
            receipt["status"] = "INTERRUPTED"
            write_json(output / "completion.json", receipt)

        self._assert_failed(mutate)

    def test_completion_run_id_mismatch_is_rejected(self) -> None:
        def mutate(output: Path, _result: dict[str, object]) -> None:
            receipt = load_json(output / "completion.json")
            receipt["run_id"] = "another-run"
            write_json(output / "completion.json", receipt)

        self._assert_failed(mutate)

    def test_summary_run_id_mismatch_is_rejected(self) -> None:
        def mutate(output: Path, _result: dict[str, object]) -> None:
            summary = load_json(output / "summary.json")
            summary["run_id"] = "another-run"
            write_json(output / "summary.json", summary)

        self._assert_failed(mutate)

    def test_manifest_run_id_mismatch_is_rejected(self) -> None:
        def mutate(output: Path, _result: dict[str, object]) -> None:
            manifest = load_json(output / "run_manifest.json")
            manifest["run_id"] = "another-run"
            write_json(output / "run_manifest.json", manifest)

        self._assert_failed(mutate)

    def test_manifest_sha_mismatch_is_rejected(self) -> None:
        def mutate(output: Path, _result: dict[str, object]) -> None:
            receipt = load_json(output / "completion.json")
            receipt["run_manifest_sha256"] = "B" * 64
            write_json(output / "completion.json", receipt)

        self._assert_failed(mutate)

    def test_summary_artifact_sha_mismatch_is_rejected(self) -> None:
        self._assert_failed(
            lambda output, _result: (output / "summary.json").write_text(
                (output / "summary.json").read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
        )

    def test_other_required_artifact_sha_mismatch_is_rejected(self) -> None:
        self._assert_failed(
            lambda output, _result: (output / "annotated_video.mp4").write_bytes(
                b"tampered"
            )
        )

    def test_missing_required_artifact_is_rejected(self) -> None:
        self._assert_failed(
            lambda output, _result: (output / "events.json").unlink()
        )

    def test_malformed_summary_is_rejected(self) -> None:
        self._assert_failed(
            lambda output, _result: (output / "summary.json").write_text(
                "not-json", encoding="utf-8"
            )
        )

    def test_malformed_manifest_is_rejected(self) -> None:
        self._assert_failed(
            lambda output, _result: (output / "run_manifest.json").write_text(
                "not-json", encoding="utf-8"
            )
        )

    def test_input_identity_mismatch_is_rejected(self) -> None:
        def mutate(output: Path, _result: dict[str, object]) -> None:
            receipt = load_json(output / "completion.json")
            receipt["input_sha256"] = "C" * 64
            write_json(output / "completion.json", receipt)

        self._assert_failed(mutate)

    def test_active_phase4_run_lock_is_rejected(self) -> None:
        self._assert_failed(
            lambda output, _result: (output / ".run.lock").write_text(
                "reserved\n", encoding="utf-8"
            )
        )

    def test_failed_technical_representation_never_completes(self) -> None:
        def analyze(_source: Path, output: Path) -> dict[str, object]:
            output.mkdir(parents=True)
            run_id = output.name
            result = {
                "schema_version": "road_damage.video_analysis.v1.summary",
                "run_id": run_id,
                "status": "FAILED_TECHNICAL",
                "frames": {"inferred": 1},
            }
            write_json(output / "summary.json", result)
            write_json(
                output / "run_manifest.json",
                {
                    "schema_version": "road_damage.video_analysis.v1.run_manifest",
                    "run_id": run_id,
                    "status": "FAILED_TECHNICAL",
                },
            )
            return result

        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            analysis_loader=lambda: analyze,
        )
        job = asyncio.run(service.create_job(MemoryUpload("road.mp4", b"video")))
        with self.assertLogs("road_damage.api.services.analysis_service", level="ERROR"):
            service.execute_job(job.job_id)
        self.assertEqual(service.get_job(job.job_id).status.value, "FAILED")

    def test_consistent_interrupted_representation_maps_to_interrupted(self) -> None:
        def analyze(_source: Path, output: Path) -> dict[str, object]:
            output.mkdir(parents=True)
            run_id = output.name
            result = {
                "schema_version": "road_damage.video_analysis.v1.summary",
                "run_id": run_id,
                "status": "INTERRUPTED",
                "frames": {"inferred": 2},
            }
            write_json(output / "summary.json", result)
            write_json(
                output / "run_manifest.json",
                {
                    "schema_version": "road_damage.video_analysis.v1.run_manifest",
                    "run_id": run_id,
                    "status": "INTERRUPTED",
                },
            )
            return result

        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            analysis_loader=lambda: analyze,
        )
        job = asyncio.run(service.create_job(MemoryUpload("road.mp4", b"video")))
        service.execute_job(job.job_id)
        result = service.get_job(job.job_id)
        self.assertEqual(result.status.value, "INTERRUPTED")
        self.assertEqual(result.progress.frames_processed, 2)


class RecordingAsgiApplication:
    def __init__(self) -> None:
        self.calls = 0
        self.body = b""

    async def __call__(self, _scope, receive, send) -> None:
        self.calls += 1
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            self.body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


async def invoke_asgi(
    app,
    chunks: list[bytes],
    *,
    content_length: int | None,
    content_type: bytes = b"application/octet-stream",
) -> list[dict[str, object]]:
    headers = [(b"content-type", content_type)]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "POST",
        "path": "/api/v1/analyses",
        "raw_path": b"/api/v1/analyses",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8000),
        "root_path": "",
    }
    await app(scope, receive, send)
    return sent


class UploadIngressLimitTests(unittest.TestCase):
    def test_oversized_content_length_never_reaches_route_or_service(self) -> None:
        with TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as temporary:
            output_root = Path(temporary) / "api"
            settings = BackendSettings(
                output_root=output_root,
                max_upload_bytes=100,
                upload_chunk_bytes=10,
            )
            service = Mock()
            service.create_job = AsyncMock()
            service.execute_job = Mock()
            with patch(
                "road_damage.api.app.upload_request_body_limit", return_value=5
            ):
                application = create_app(settings, service)
            sent = asyncio.run(
                invoke_asgi(application, [b"x"], content_length=6)
            )
            self.assertEqual(sent[0]["status"], 413)
            service.create_job.assert_not_called()
            service.execute_job.assert_not_called()
            self.assertFalse(output_root.exists())

    def test_chunked_multipart_within_limit_reaches_route(self) -> None:
        with TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as temporary:
            settings = BackendSettings(
                output_root=Path(temporary) / "api",
                max_upload_bytes=100,
                upload_chunk_bytes=10,
            )
            now = datetime.now(timezone.utc)
            service = Mock()
            service.create_job = AsyncMock(
                return_value=AnalysisJobResponse(
                    job_id=uuid4(),
                    status="QUEUED",
                    created_at=now,
                    updated_at=now,
                    progress=AnalysisProgress(frames_processed=0),
                    input_filename="road.mp4",
                    availability=AnalysisArtifactAvailability(),
                )
            )
            service.execute_job = Mock()
            body = (
                b"--phase5b\r\nContent-Disposition: form-data; name=\"file\"; "
                b"filename=\"road.mp4\"\r\nContent-Type: video/mp4\r\n\r\n"
                b"x\r\n--phase5b--\r\n"
            )
            sent = asyncio.run(
                invoke_asgi(
                    create_app(settings, service),
                    [body[:40], body[40:]],
                    content_length=None,
                    content_type=b"multipart/form-data; boundary=phase5b",
                )
            )
            self.assertEqual(sent[0]["status"], 202, sent)
            service.create_job.assert_awaited_once()
            service.execute_job.assert_called_once()

    def test_content_length_over_limit_rejected_before_downstream(self) -> None:
        downstream = RecordingAsgiApplication()
        guard = UploadRequestBodyLimitMiddleware(
            downstream, maximum_body_bytes=5, upload_path="/api/v1/analyses"
        )
        sent = asyncio.run(invoke_asgi(guard, [b"123456"], content_length=6))
        self.assertEqual(sent[0]["status"], 413)
        self.assertEqual(downstream.calls, 0)

    def test_content_length_exactly_at_limit_is_not_rejected(self) -> None:
        downstream = RecordingAsgiApplication()
        guard = UploadRequestBodyLimitMiddleware(
            downstream, maximum_body_bytes=5, upload_path="/api/v1/analyses"
        )
        sent = asyncio.run(invoke_asgi(guard, [b"12345"], content_length=5))
        self.assertEqual(sent[0]["status"], 204)
        self.assertEqual(downstream.body, b"12345")

    def test_chunked_exact_limit_is_accepted(self) -> None:
        downstream = RecordingAsgiApplication()
        guard = UploadRequestBodyLimitMiddleware(
            downstream, maximum_body_bytes=5, upload_path="/api/v1/analyses"
        )
        sent = asyncio.run(invoke_asgi(guard, [b"12", b"345"], content_length=None))
        self.assertEqual(sent[0]["status"], 204)
        self.assertEqual(downstream.body, b"12345")

    def test_chunked_limit_plus_one_is_rejected(self) -> None:
        downstream = RecordingAsgiApplication()
        guard = UploadRequestBodyLimitMiddleware(
            downstream, maximum_body_bytes=5, upload_path="/api/v1/analyses"
        )
        sent = asyncio.run(invoke_asgi(guard, [b"12", b"345", b"6"], content_length=None))
        self.assertEqual(sent[0]["status"], 413)

    def test_overhead_policy_is_bounded_and_separate_from_file_limit(self) -> None:
        self.assertEqual(
            upload_request_body_limit(17),
            17 + MULTIPART_OVERHEAD_ALLOWANCE_BYTES,
        )

    def test_chunked_ingress_rejection_never_calls_service_or_creates_job(self) -> None:
        with TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as temporary:
            output_root = Path(temporary) / "api"
            settings = BackendSettings(
                output_root=output_root,
                max_upload_bytes=100,
                upload_chunk_bytes=10,
            )
            service = Mock()
            service.create_job = AsyncMock()
            service.execute_job = Mock()
            body = (
                b"--phase5b\r\nContent-Disposition: form-data; name=\"file\"; "
                b"filename=\"road.mp4\"\r\nContent-Type: video/mp4\r\n\r\n"
                b"x\r\n--phase5b--\r\n"
            )
            with patch(
                "road_damage.api.app.upload_request_body_limit", return_value=5
            ):
                application = create_app(settings, service)
            sent = asyncio.run(
                invoke_asgi(
                    application,
                    [body],
                    content_length=None,
                    content_type=b"multipart/form-data; boundary=phase5b",
                )
            )
            self.assertEqual(sent[0]["status"], 413, sent)
            service.create_job.assert_not_called()
            service.execute_job.assert_not_called()
            self.assertFalse(output_root.exists())

    def test_service_file_limit_accepts_exact_and_rejects_plus_one(self) -> None:
        with TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as temporary:
            output_root = Path(temporary) / "api"
            settings = BackendSettings(
                output_root=output_root,
                max_upload_bytes=7,
                upload_chunk_bytes=3,
            )
            service = Phase5BAnalysisService(
                settings, video_validator=lambda _path: None
            )
            accepted = asyncio.run(
                service.create_job(MemoryUpload("exact.mp4", b"1234567"))
            )
            self.assertEqual(service.get_job(accepted.job_id).status.value, "QUEUED")
            with self.assertRaises(ApiApplicationError) as caught:
                asyncio.run(
                    service.create_job(MemoryUpload("large.mp4", b"12345678"))
                )
            self.assertEqual(caught.exception.code, "UPLOAD_TOO_LARGE")
            self.assertEqual(len(list(output_root.iterdir())), 1)


class JobDirectoryOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(dir=PROJECT_ROOT / "outputs")
        self.output_root = Path(self.temporary.name) / "api"
        self.settings = BackendSettings(
            output_root=self.output_root,
            max_upload_bytes=64,
            upload_chunk_bytes=4,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_uuid_collision_never_reuses_or_deletes_preexisting_directory(self) -> None:
        collision = uuid4()
        existing = self.output_root / str(collision)
        existing.mkdir(parents=True)
        sentinel = existing / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            job_id_factory=lambda: collision,
        )
        with self.assertRaises(ApiApplicationError) as caught:
            asyncio.run(service.create_job(MemoryUpload("road.mp4", b"video")))
        self.assertEqual(caught.exception.code, "UPLOAD_FAILED")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertFalse((existing / "input").exists())
        with self.assertRaises(UnknownAnalysisJobError):
            service.registry.get(collision)

    def test_child_creation_failure_cleans_only_newly_owned_parent(self) -> None:
        job_id = uuid4()
        self.output_root.mkdir(parents=True)
        original_mkdir = Path.mkdir

        def fail_input(path: Path, *args, **kwargs) -> None:
            if path.name == "input":
                raise OSError("synthetic child failure")
            original_mkdir(path, *args, **kwargs)

        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            job_id_factory=lambda: job_id,
        )
        with patch.object(Path, "mkdir", new=fail_input):
            with self.assertRaises(ApiApplicationError):
                asyncio.run(service.create_job(MemoryUpload("road.mp4", b"video")))
        self.assertFalse((self.output_root / str(job_id)).exists())
        with self.assertRaises(UnknownAnalysisJobError):
            service.registry.get(job_id)

    def test_upload_failure_cleans_newly_owned_parent(self) -> None:
        class FailingUpload(MemoryUpload):
            async def read(self, size: int = -1) -> bytes:
                del size
                raise OSError("synthetic upload failure")

        job_id = uuid4()
        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            job_id_factory=lambda: job_id,
        )
        with self.assertRaises(ApiApplicationError):
            asyncio.run(service.create_job(FailingUpload("road.mp4", b"")))
        self.assertFalse((self.output_root / str(job_id)).exists())
        with self.assertRaises(UnknownAnalysisJobError):
            service.registry.get(job_id)

    def test_validation_failure_cleans_newly_owned_parent(self) -> None:
        job_id = uuid4()
        service = Phase5BAnalysisService(
            self.settings,
            video_validator=Mock(side_effect=StagedVideoValidationError("bad")),
            job_id_factory=lambda: job_id,
        )
        with self.assertRaises(ApiApplicationError):
            asyncio.run(service.create_job(MemoryUpload("road.mp4", b"video")))
        self.assertFalse((self.output_root / str(job_id)).exists())
        with self.assertRaises(UnknownAnalysisJobError):
            service.registry.get(job_id)

    def test_successful_creation_owns_exactly_one_uuid_directory(self) -> None:
        job_id = uuid4()
        service = Phase5BAnalysisService(
            self.settings,
            video_validator=lambda _path: None,
            job_id_factory=lambda: job_id,
        )
        response = asyncio.run(
            service.create_job(MemoryUpload("road.mp4", b"video"))
        )
        self.assertEqual(response.job_id, job_id)
        self.assertEqual([path.name for path in self.output_root.iterdir()], [str(job_id)])
        self.assertTrue((self.output_root / str(job_id) / "input" / "video.mp4").is_file())


if __name__ == "__main__":
    unittest.main()

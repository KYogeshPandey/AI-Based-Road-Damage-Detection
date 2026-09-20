"""Focused Phase 5B backend contracts and safety regression tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import UUID, uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from road_damage.api.config import (  # noqa: E402
    BackendConfigurationError,
    BackendSettings,
)
from road_damage.api.paths import (  # noqa: E402
    UnsafeApiPathError,
    resolve_job_artifact,
    resolve_job_directory,
    validate_public_filename,
)


BACKEND_RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("fastapi", "pydantic", "starlette")
)
BACKEND_SKIP_REASON = (
    "Install requirements-backend.txt to run FastAPI/Pydantic runtime tests."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class ApiPathSafetyTests(unittest.TestCase):
    def test_filename_accepts_only_a_base_name(self) -> None:
        self.assertEqual(validate_public_filename("road-video.mp4"), "road-video.mp4")

    def test_filename_rejects_traversal(self) -> None:
        for value in ("../video.mp4", "..\\video.mp4", "..", "a/../b.mp4"):
            with self.subTest(value=value), self.assertRaises(UnsafeApiPathError):
                validate_public_filename(value)

    def test_filename_rejects_absolute_and_drive_paths(self) -> None:
        values = (
            "/etc/passwd",
            "C:\\Windows\\system.ini",
            "C:relative.txt",
            "\\\\server\\share\\video.mp4",
        )
        for value in values:
            with self.subTest(value=value), self.assertRaises(UnsafeApiPathError):
                validate_public_filename(value)

    def test_filename_rejects_separators_and_alternate_stream_syntax(self) -> None:
        for value in ("folder/video.mp4", "folder\\video.mp4", "video.mp4:stream"):
            with self.subTest(value=value), self.assertRaises(UnsafeApiPathError):
                validate_public_filename(value)

    def test_job_directory_is_uuid_named_and_confined(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = uuid4()
            result = resolve_job_directory(root, job_id)
            self.assertEqual(result.parent, root.resolve())
            self.assertEqual(result.name, str(job_id))

    def test_job_directory_rejects_non_uuid_values(self) -> None:
        with self.assertRaises(UnsafeApiPathError):
            resolve_job_directory(PROJECT_ROOT / "outputs" / "api", "../../src")

    def test_artifact_path_is_confined_to_job_directory(self) -> None:
        job_id = uuid4()
        root = PROJECT_ROOT / "outputs" / "api"
        result = resolve_job_artifact(root, job_id, "summary.json")
        self.assertEqual(result, root.resolve() / str(job_id) / "summary.json")
        with self.assertRaises(UnsafeApiPathError):
            resolve_job_artifact(root, job_id, "../summary.json")


class BackendConfigurationTests(unittest.TestCase):
    def test_defaults_are_local_and_cors_is_disabled(self) -> None:
        settings = BackendSettings()
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 8000)
        self.assertFalse(settings.debug)
        self.assertEqual(settings.cors_origins, ())
        self.assertFalse(settings.cors_allow_credentials)
        self.assertTrue(settings.output_root.is_relative_to(PROJECT_ROOT / "outputs"))

    def test_explicit_backend_environment_values_are_parsed(self) -> None:
        settings = BackendSettings.from_environment(
            {
                "ROAD_DAMAGE_API_HOST": "localhost",
                "ROAD_DAMAGE_API_PORT": "9000",
                "ROAD_DAMAGE_API_DEBUG": "true",
                "ROAD_DAMAGE_API_DOCS_ENABLED": "false",
                "ROAD_DAMAGE_API_MAX_UPLOAD_BYTES": "1024",
                "ROAD_DAMAGE_API_UPLOAD_CHUNK_BYTES": "64",
                "ROAD_DAMAGE_API_OUTPUT_ROOT": "outputs/api-test",
                "ROAD_DAMAGE_API_CORS_ORIGINS": (
                    "http://localhost:3000,https://frontend.example"
                ),
                "ROAD_DAMAGE_API_CORS_ALLOW_CREDENTIALS": "true",
            }
        )
        self.assertEqual(settings.port, 9000)
        self.assertTrue(settings.debug)
        self.assertFalse(settings.docs_enabled)
        self.assertEqual(settings.max_upload_bytes, 1024)
        self.assertEqual(settings.upload_chunk_bytes, 64)
        self.assertEqual(
            settings.cors_origins,
            ("http://localhost:3000", "https://frontend.example"),
        )

    def test_model_and_scientific_overrides_are_rejected(self) -> None:
        forbidden = (
            "ROAD_DAMAGE_API_CONFIDENCE",
            "ROAD_DAMAGE_API_NMS_IOU",
            "ROAD_DAMAGE_API_MODEL_PATH",
            "ROAD_DAMAGE_API_CHECKPOINT_PATH",
            "ROAD_DAMAGE_API_CLASS_NAMES",
            "ROAD_DAMAGE_API_DEVICE",
        )
        for name in forbidden:
            with self.subTest(name=name), self.assertRaises(BackendConfigurationError):
                BackendSettings.from_environment({name: "unsafe"})

    def test_protected_project_areas_cannot_be_output_roots(self) -> None:
        for directory in ("src", "configs", "models", "data", "weights", ".git"):
            with self.subTest(directory=directory), self.assertRaises(
                BackendConfigurationError
            ):
                BackendSettings(output_root=PROJECT_ROOT / directory / "api")

    def test_project_root_and_outputs_root_are_not_valid_output_roots(self) -> None:
        for path in (PROJECT_ROOT, PROJECT_ROOT / "outputs"):
            with self.subTest(path=path), self.assertRaises(BackendConfigurationError):
                BackendSettings(output_root=path)

    def test_wildcard_and_malformed_cors_origins_are_rejected(self) -> None:
        for origins in (("*",), ("localhost:3000",), ("https://example/a",)):
            with self.subTest(origins=origins), self.assertRaises(
                BackendConfigurationError
            ):
                BackendSettings(cors_origins=origins)

    def test_credentialed_cors_requires_an_explicit_allowlist(self) -> None:
        with self.assertRaises(BackendConfigurationError):
            BackendSettings(cors_allow_credentials=True)

    def test_backend_settings_have_no_detector_controls(self) -> None:
        settings = BackendSettings()
        for attribute in ("confidence", "nms_iou", "model_path", "device", "imgsz"):
            self.assertFalse(hasattr(settings, attribute), attribute)


class BackendSourceIsolationTests(unittest.TestCase):
    def test_fresh_application_import_does_not_load_phase4_torch_or_ultralytics(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(SOURCE_ROOT), str(PROJECT_ROOT / ".venv" / "Lib" / "site-packages"))
        )
        probe = (
            "import sys; import road_damage.api.app; "
            "blocked=('road_damage.inference.video_analysis','torch','ultralytics'); "
            "print([name for name in blocked if name in sys.modules])"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.stdout.strip(), "[]")

    def test_api_source_does_not_import_phase4_inference_or_ml_frameworks(self) -> None:
        forbidden = (
            "from road_damage.inference",
            "import road_damage.inference",
            "import ultralytics",
            "from ultralytics",
            "import torch",
            "from torch",
        )
        for path in sorted((SOURCE_ROOT / "road_damage" / "api").rglob("*.py")):
            text = path.read_text(encoding="utf-8").casefold()
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token.casefold(), text)

    def test_phase4_frozen_files_are_byte_unchanged(self) -> None:
        expected = {
            "configs/inference/video_analysis_yolov8s.yaml": (
                "5B3F8C3B0BD4A52DAD12F274F77E641647CD7E263DAAAB34CD42E17942A27228"
            ),
            "src/road_damage/inference/video_analysis.py": (
                "9EA635007486C865EB31716A407F399FDA29005C462DAB66FFD2B31DDA2F175F"
            ),
            "src/road_damage/aggregation/temporal.py": (
                "60EB4A8C12A57D717B850455E5A0286F660ECC7C4E4AA5AC7A8153BB788CD532"
            ),
        }
        for relative, expected_hash in expected.items():
            with self.subTest(relative=relative):
                self.assertEqual(sha256(PROJECT_ROOT / relative), expected_hash)

    def test_backend_dependency_file_is_separate_and_pinned(self) -> None:
        backend = (PROJECT_ROOT / "requirements-backend.txt").read_text(encoding="utf-8")
        root = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        for name in (
            "fastapi==",
            "pydantic==",
            "uvicorn==",
            "httpx==",
            "python-multipart==",
        ):
            self.assertIn(name, backend)
            self.assertNotIn(name, root)


@unittest.skipUnless(BACKEND_RUNTIME_AVAILABLE, BACKEND_SKIP_REASON)
class ApiRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient
        from road_damage.api.app import create_app

        self.app = create_app(BackendSettings())
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        self.client.close()

    def test_health_is_deterministic(self) -> None:
        expected = {
            "status": "ok",
            "service": "AI-Based Road Damage Detection API",
            "api_version": "1.0.0",
        }
        first = self.client.get("/api/v1/health")
        second = self.client.get("/api/v1/health")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), expected)
        self.assertEqual(second.json(), expected)

    def test_health_and_system_do_not_call_inference(self) -> None:
        sentinel = types.ModuleType("road_damage.inference.video_analysis")
        sentinel.analyze_video = Mock(side_effect=AssertionError("inference called"))
        with patch.dict(
            sys.modules, {"road_damage.inference.video_analysis": sentinel}
        ):
            self.assertEqual(self.client.get("/api/v1/health").status_code, 200)
            self.assertEqual(self.client.get("/api/v1/system").status_code, 200)
        sentinel.analyze_video.assert_not_called()

    def test_health_does_not_import_model_or_gpu_frameworks(self) -> None:
        before = set(sys.modules)
        self.client.get("/api/v1/health")
        newly_imported = set(sys.modules).difference(before)
        self.assertFalse(any(name == "ultralytics" or name.startswith("ultralytics.") for name in newly_imported))
        self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in newly_imported))

    def test_system_exposes_only_safe_canonical_capabilities(self) -> None:
        response = self.client.get("/api/v1/system")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_family"], "YOLOv8s")
        self.assertEqual(payload["operating_point_status"], "frozen_validation_selected")
        self.assertTrue(payload["analysis_execution_enabled"])
        self.assertEqual(payload["current_phase"], "Phase 5B")
        self.assertEqual(
            [(item["class_id"], item["class_name"], item["friendly_display_name"]) for item in payload["canonical_classes"]],
            [
                (0, "D00_longitudinal_crack", "Longitudinal Crack"),
                (1, "D10_transverse_crack", "Transverse Crack"),
                (2, "D20_alligator_crack", "Alligator Crack"),
                (3, "D40_pothole", "Pothole"),
            ],
        )
        serialized = json.dumps(payload)
        self.assertNotIn(str(PROJECT_ROOT), serialized)
        self.assertNotIn("best.pt", serialized)

    def test_analysis_execution_capability_is_truthful(self) -> None:
        response = self.client.get("/api/v1/analyses/capabilities")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "submission_enabled": True,
                "execution_enabled": True,
                "current_phase": "Phase 5B",
                "message": "Secure video submission and background analysis execution are enabled.",
                "supported_video_extensions": [".mp4", ".avi", ".mov", ".mkv"],
                "maximum_upload_bytes": 536870912,
            },
        )

    def test_submission_route_requires_the_multipart_file_field(
        self,
    ) -> None:
        self.assertIn("/api/v1/analyses", self.app.openapi()["paths"])
        response = self.client.post(
            "/api/v1/analyses", json={"input_filename": "video.mp4"}
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
        self.assertNotIn("job_id", response.text)
        self.assertNotIn('"status":"COMPLETED"', response.text)

    def test_validation_errors_use_safe_stable_envelope(self) -> None:
        from fastapi import Query

        @self.app.get("/api/v1/test-validation")
        def test_validation(limit: int = Query(ge=0)) -> dict[str, int]:
            return {"limit": limit}

        response = self.client.get("/api/v1/test-validation?limit=-1")
        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], "INVALID_REQUEST")
        self.assertIn("violations", payload["error"]["details"])
        self.assertNotIn("input", json.dumps(payload))

    def test_not_found_uses_safe_stable_envelope(self) -> None:
        response = self.client.get("/api/v1/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "NOT_FOUND",
                    "message": "Resource not found.",
                    "details": None,
                }
            },
        )

    def test_unexpected_error_does_not_leak_exception_or_path(self) -> None:
        @self.app.get("/api/v1/test-error")
        def test_error() -> None:
            raise RuntimeError(f"secret at {PROJECT_ROOT}")

        response = self.client.get("/api/v1/test-error")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "INTERNAL_SERVER_ERROR")
        self.assertNotIn("secret", response.text)
        self.assertNotIn(str(PROJECT_ROOT), response.text)

    def test_docs_and_openapi_are_available_in_development(self) -> None:
        self.assertEqual(self.client.get("/docs").status_code, 200)
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertEqual(schema["info"]["version"], "1.0.0")
        self.assertIn("/api/v1/health", schema["paths"])
        self.assertIn("/api/v1/system", schema["paths"])
        self.assertIn("post", schema["paths"]["/api/v1/analyses"])
        self.assertNotIn("post", schema["paths"]["/api/v1/analyses/capabilities"])

    def test_docs_can_be_disabled(self) -> None:
        from fastapi.testclient import TestClient
        from road_damage.api.app import create_app

        with TestClient(
            create_app(BackendSettings(docs_enabled=False)),
            raise_server_exceptions=False,
        ) as client:
            self.assertEqual(client.get("/docs").status_code, 404)
            self.assertEqual(client.get("/openapi.json").status_code, 404)

    def test_cors_is_absent_by_default_and_allowlisted_when_configured(self) -> None:
        from fastapi.testclient import TestClient
        from road_damage.api.app import create_app

        default_response = self.client.get(
            "/api/v1/health", headers={"Origin": "http://localhost:3000"}
        )
        self.assertNotIn("access-control-allow-origin", default_response.headers)
        settings = BackendSettings(cors_origins=("http://localhost:3000",))
        with TestClient(create_app(settings), raise_server_exceptions=False) as client:
            allowed = client.get(
                "/api/v1/health", headers={"Origin": "http://localhost:3000"}
            )
            denied = client.get(
                "/api/v1/health", headers={"Origin": "https://untrusted.example"}
            )
        self.assertEqual(
            allowed.headers.get("access-control-allow-origin"),
            "http://localhost:3000",
        )
        self.assertNotIn("access-control-allow-origin", denied.headers)


@unittest.skipUnless(BACKEND_RUNTIME_AVAILABLE, BACKEND_SKIP_REASON)
class ApiSchemaAndServiceTests(unittest.TestCase):
    @staticmethod
    def _result_summary_data() -> dict[str, object]:
        return {
            "run_id": "example-run",
            "status": "COMPLETED",
            "frames_inferred": 10,
            "raw_detection_observation_total": 4,
            "temporal_event_total": 4,
            "confirmed_event_total": 2,
            "tentative_event_total": 2,
            "per_class": [
                {
                    "class_id": 0,
                    "class_name": "D00_longitudinal_crack",
                    "raw_detection_observations": 2,
                    "event_total": 2,
                    "confirmed_event_total": 1,
                    "tentative_event_total": 1,
                },
                {
                    "class_id": 1,
                    "class_name": "D10_transverse_crack",
                    "raw_detection_observations": 1,
                    "event_total": 1,
                    "confirmed_event_total": 1,
                    "tentative_event_total": 0,
                },
                {
                    "class_id": 2,
                    "class_name": "D20_alligator_crack",
                    "raw_detection_observations": 0,
                    "event_total": 0,
                    "confirmed_event_total": 0,
                    "tentative_event_total": 0,
                },
                {
                    "class_id": 3,
                    "class_name": "D40_pothole",
                    "raw_detection_observations": 1,
                    "event_total": 1,
                    "confirmed_event_total": 0,
                    "tentative_event_total": 1,
                },
            ],
            "timing": {
                "processing_elapsed_seconds": 1.0,
                "model_inference_seconds": 0.5,
                "effective_inference_fps": 20.0,
                "overall_processing_fps": 10.0,
            },
        }

    @staticmethod
    def _job_data(
        status: str,
        *,
        created_at: datetime,
        updated_at: datetime,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return {
            "job_id": uuid4(),
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "progress": {"frames_processed": 0},
            "input_filename": "road.mp4",
            "availability": {},
            "error": error,
        }

    def test_job_lifecycle_enum_is_stable(self) -> None:
        from road_damage.api.schemas.analysis import AnalysisJobStatus

        self.assertEqual(
            [item.value for item in AnalysisJobStatus],
            ["QUEUED", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED"],
        )

    def test_progress_constraints_are_enforced(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import AnalysisProgress

        with self.assertRaises(ValidationError):
            AnalysisProgress(frames_processed=-1)
        with self.assertRaises(ValidationError):
            AnalysisProgress(frames_processed=2, total_frames=1)
        with self.assertRaises(ValidationError):
            AnalysisProgress(frames_processed=0, percent_complete=101)

    def test_submission_schema_rejects_server_paths(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import AnalysisSubmissionRequest

        for value in ("../video.mp4", "C:\\private\\video.mp4", "/tmp/video.mp4"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AnalysisSubmissionRequest(input_filename=value)

    def test_event_contract_preserves_canonical_machine_values(self) -> None:
        from road_damage.api.schemas.analysis import DamageEventResponse

        event = DamageEventResponse(
            event_id="RD0012",
            class_id=2,
            class_name="D20_alligator_crack",
            friendly_display_name="Alligator Crack",
            status="confirmed",
            first_frame=10,
            last_frame=15,
            first_timestamp_ms=1000,
            last_timestamp_ms=1500,
            duration_seconds=0.5,
            observation_count=4,
            mean_confidence=0.37,
            max_confidence=0.50,
        )
        payload = event.model_dump(mode="json")
        self.assertEqual(payload["event_id"], "RD0012")
        self.assertEqual(payload["class_id"], 2)
        self.assertEqual(payload["class_name"], "D20_alligator_crack")
        self.assertEqual(payload["friendly_display_name"], "Alligator Crack")

    def test_event_contract_rejects_mismatched_friendly_or_canonical_name(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import DamageEventResponse

        common = {
            "event_id": "RD0001",
            "class_id": 3,
            "status": "tentative",
            "first_frame": 0,
            "last_frame": 0,
            "first_timestamp_ms": 0,
            "last_timestamp_ms": 0,
            "duration_seconds": 0.0,
            "observation_count": 1,
            "mean_confidence": 0.2,
            "max_confidence": 0.2,
        }
        with self.assertRaises(ValidationError):
            DamageEventResponse(
                **common,
                class_name="D20_alligator_crack",
                friendly_display_name="Pothole",
            )
        with self.assertRaises(ValidationError):
            DamageEventResponse(
                **common,
                class_name="D40_pothole",
                friendly_display_name="Alligator Crack",
            )

    def test_event_contract_rejects_rd0000(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import DamageEventResponse

        with self.assertRaises(ValidationError):
            DamageEventResponse(
                event_id="RD0000",
                class_id=0,
                class_name="D00_longitudinal_crack",
                friendly_display_name="Longitudinal Crack",
                status="tentative",
                first_frame=0,
                last_frame=0,
                first_timestamp_ms=0,
                last_timestamp_ms=0,
                duration_seconds=0.0,
                observation_count=1,
                mean_confidence=0.2,
                max_confidence=0.2,
            )

    def test_result_summary_uses_phase4_status_and_heuristic_terminology(self) -> None:
        from road_damage.api.schemas.analysis import AnalysisResultSummary

        payload = AnalysisResultSummary(
            **self._result_summary_data()
        ).model_dump(mode="json")
        self.assertEqual(payload["status"], "COMPLETED")
        self.assertEqual(payload["confirmed_event_total"], 2)
        self.assertEqual(payload["tentative_event_total"], 2)
        self.assertIn("Heuristic temporal damage events", payload["event_interpretation"])
        self.assertIn("not ground-truth", payload["event_interpretation"])

    def test_result_summary_rejects_per_class_confirmed_sum_mismatch(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import AnalysisResultSummary

        data = self._result_summary_data()
        first_class = data["per_class"][0]
        first_class["confirmed_event_total"] = 2
        first_class["tentative_event_total"] = 0
        with self.assertRaisesRegex(ValidationError, "Per-class confirmed"):
            AnalysisResultSummary(**data)

    def test_result_summary_rejects_per_class_tentative_sum_mismatch(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import AnalysisResultSummary

        data = self._result_summary_data()
        first_class = data["per_class"][0]
        first_class["confirmed_event_total"] = 0
        first_class["tentative_event_total"] = 2
        with self.assertRaisesRegex(ValidationError, "Per-class tentative"):
            AnalysisResultSummary(**data)

    def test_all_job_lifecycle_states_accept_consistent_records(self) -> None:
        from road_damage.api.schemas.analysis import AnalysisJobResponse

        created = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
        started = created + timedelta(seconds=1)
        completed = started + timedelta(seconds=1)
        failure = {"code": "PROCESSING_FAILED", "message": "Processing failed."}
        valid_records = (
            self._job_data("QUEUED", created_at=created, updated_at=created),
            self._job_data(
                "RUNNING",
                created_at=created,
                updated_at=started,
                started_at=started,
            ),
            self._job_data(
                "COMPLETED",
                created_at=created,
                updated_at=completed,
                started_at=started,
                completed_at=completed,
            ),
            self._job_data(
                "FAILED",
                created_at=created,
                updated_at=completed,
                started_at=started,
                completed_at=completed,
                error=failure,
            ),
            self._job_data(
                "INTERRUPTED",
                created_at=created,
                updated_at=completed,
                started_at=started,
                completed_at=completed,
            ),
            self._job_data(
                "INTERRUPTED",
                created_at=created,
                updated_at=completed,
                started_at=started,
                completed_at=completed,
                error={"code": "USER_STOPPED", "message": "Stopped by user."},
            ),
        )
        for data in valid_records:
            with self.subTest(status=data["status"], error=data["error"]):
                job = AnalysisJobResponse(**data)
                self.assertIsInstance(job.job_id, UUID)

    def test_queued_job_rejects_started_completed_and_error_fields(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import AnalysisJobResponse

        created = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
        later = created + timedelta(seconds=1)
        contradictions = (
            {"started_at": later},
            {"completed_at": later},
            {"error": {"code": "UNEXPECTED", "message": "Unexpected."}},
        )
        for contradiction in contradictions:
            data = self._job_data(
                "QUEUED", created_at=created, updated_at=later
            )
            data.update(contradiction)
            with self.subTest(contradiction=contradiction), self.assertRaises(
                ValidationError
            ):
                AnalysisJobResponse(**data)

    def test_running_job_requires_start_and_rejects_completion_and_error(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import AnalysisJobResponse

        created = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
        started = created + timedelta(seconds=1)
        completed = started + timedelta(seconds=1)
        contradictions = (
            {},
            {"started_at": started, "completed_at": completed},
            {
                "started_at": started,
                "error": {"code": "UNEXPECTED", "message": "Unexpected."},
            },
        )
        for contradiction in contradictions:
            data = self._job_data(
                "RUNNING", created_at=created, updated_at=completed
            )
            data.update(contradiction)
            with self.subTest(contradiction=contradiction), self.assertRaises(
                ValidationError
            ):
                AnalysisJobResponse(**data)

    def test_completed_job_requires_both_timestamps_and_rejects_error(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import AnalysisJobResponse

        created = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
        started = created + timedelta(seconds=1)
        completed = started + timedelta(seconds=1)
        contradictions = (
            {"completed_at": completed},
            {"started_at": started},
            {
                "started_at": started,
                "completed_at": completed,
                "error": {"code": "UNEXPECTED", "message": "Unexpected."},
            },
        )
        for contradiction in contradictions:
            data = self._job_data(
                "COMPLETED", created_at=created, updated_at=completed
            )
            data.update(contradiction)
            with self.subTest(contradiction=contradiction), self.assertRaises(
                ValidationError
            ):
                AnalysisJobResponse(**data)

    def test_failed_job_requires_error(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import AnalysisJobResponse

        created = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
        started = created + timedelta(seconds=1)
        completed = started + timedelta(seconds=1)
        with self.assertRaises(ValidationError):
            AnalysisJobResponse(
                **self._job_data(
                    "FAILED",
                    created_at=created,
                    updated_at=completed,
                    started_at=started,
                    completed_at=completed,
                )
            )

    def test_job_timestamps_must_follow_lifecycle_order(self) -> None:
        from pydantic import ValidationError
        from road_damage.api.schemas.analysis import AnalysisJobResponse

        created = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
        first = created + timedelta(seconds=1)
        second = created + timedelta(seconds=2)
        third = created + timedelta(seconds=3)
        contradictions = (
            self._job_data(
                "RUNNING",
                created_at=created,
                updated_at=first,
                started_at=second,
            ),
            self._job_data(
                "COMPLETED",
                created_at=created,
                updated_at=third,
                started_at=second,
                completed_at=first,
            ),
            self._job_data(
                "COMPLETED",
                created_at=created,
                updated_at=second,
                started_at=first,
                completed_at=third,
            ),
        )
        for data in contradictions:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                AnalysisJobResponse(**data)

    def test_phase5b_service_implements_interface_without_loading_inference(self) -> None:
        from road_damage.api.services.analysis_service import (
            AnalysisService,
            Phase5BAnalysisService,
        )

        loader = Mock(side_effect=AssertionError("inference imported"))
        service = Phase5BAnalysisService(BackendSettings(), analysis_loader=loader)
        self.assertIsInstance(service, AnalysisService)
        capability = service.capability()
        self.assertTrue(capability.execution_enabled)
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

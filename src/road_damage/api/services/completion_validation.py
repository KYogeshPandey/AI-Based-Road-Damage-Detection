"""Independent reconciliation of frozen Phase 4 terminal artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[4]
FROZEN_PHASE4_CONFIG = (
    PROJECT_ROOT / "configs" / "inference" / "video_analysis_yolov8s.yaml"
)
RUN_LOCK_NAME = ".run.lock"
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class Phase4ArtifactValidationError(ValueError):
    """Raised when persisted Phase 4 evidence cannot prove a terminal state."""


@dataclass(frozen=True, slots=True)
class VerifiedPhase4TerminalState:
    status: str
    run_id: str
    frames_inferred: int


@dataclass(frozen=True, slots=True)
class FrozenPhase4ArtifactContract:
    output_schema_version: str
    artifacts: Mapping[str, str]


def _reject_constant(value: str) -> None:
    raise Phase4ArtifactValidationError(
        f"Non-finite JSON constant {value!r} is not permitted."
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase4ArtifactValidationError("Duplicate JSON object key is not permitted.")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except Phase4ArtifactValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise Phase4ArtifactValidationError(f"{label} is not valid JSON.") from exc
    if not isinstance(value, Mapping):
        raise Phase4ArtifactValidationError(f"{label} must contain one JSON object.")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase4ArtifactValidationError(f"{label} must be an object.")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Phase4ArtifactValidationError(f"{label} must be a non-empty string.")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Phase4ArtifactValidationError(f"{label} must be a non-negative integer.")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Phase4ArtifactValidationError(f"{label} must be a SHA-256 value.")
    return value.upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Phase4ArtifactValidationError("Required Phase 4 artifact is unreadable.") from exc
    return digest.hexdigest().upper()


def _require_schema(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("schema_version") != expected:
        raise Phase4ArtifactValidationError(f"{label} schema is incompatible.")


def _load_frozen_contract() -> FrozenPhase4ArtifactContract:
    config = _load_json_object(FROZEN_PHASE4_CONFIG, "frozen Phase 4 configuration")
    output_schema = _nonempty_string(
        config.get("output_schema_version"),
        "frozen Phase 4 configuration.output_schema_version",
    )
    artifact_values = _mapping(
        config.get("artifacts"), "frozen Phase 4 configuration.artifacts"
    )
    required_keys = {
        "run_manifest",
        "summary",
        "events",
        "frame_detections",
        "completion",
        "annotated_video",
    }
    if not required_keys.issubset(artifact_values):
        raise Phase4ArtifactValidationError(
            "Frozen Phase 4 artifact configuration is incomplete."
        )
    artifacts: dict[str, str] = {}
    for key in required_keys:
        name = _nonempty_string(
            artifact_values.get(key), f"frozen Phase 4 configuration.artifacts.{key}"
        )
        if Path(name).name != name:
            raise Phase4ArtifactValidationError(
                "Frozen Phase 4 artifact configuration contains an unsafe name."
            )
        artifacts[key] = name
    if len(set(artifacts.values())) != len(artifacts):
        raise Phase4ArtifactValidationError(
            "Frozen Phase 4 artifact configuration contains duplicate names."
        )
    return FrozenPhase4ArtifactContract(output_schema, artifacts)


def _hash_mapping(value: Any, label: str) -> dict[str, str]:
    mapping = _mapping(value, label)
    result: dict[str, str] = {}
    for name, digest in mapping.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise Phase4ArtifactValidationError(f"{label} contains an unsafe artifact name.")
        result[name] = _sha256(digest, f"{label}.{name}")
    return result


def _identity_sha(payload: Mapping[str, Any], section: str, label: str) -> str:
    return _sha256(
        _mapping(payload.get(section), f"{label}.{section}").get("sha256"),
        f"{label}.{section}.sha256",
    )


def validate_completed_phase4_run(
    output_directory: Path,
    staged_input: Path,
    returned_result: Mapping[str, Any],
) -> VerifiedPhase4TerminalState:
    """Prove a genuine frozen Phase 4 success from persisted bytes."""
    contract = _load_frozen_contract()
    if (output_directory / RUN_LOCK_NAME).exists():
        raise Phase4ArtifactValidationError("Phase 4 run lock is still active.")

    completion_path = output_directory / contract.artifacts["completion"]
    manifest_path = output_directory / contract.artifacts["run_manifest"]
    summary_path = output_directory / contract.artifacts["summary"]
    events_path = output_directory / contract.artifacts["events"]
    completion = _load_json_object(completion_path, "completion.json")
    manifest = _load_json_object(manifest_path, "run_manifest.json")
    summary = _load_json_object(summary_path, "summary.json")
    events = _load_json_object(events_path, "events.json")

    _require_schema(
        completion,
        f"{contract.output_schema_version}.completion",
        "completion.json",
    )
    _require_schema(
        manifest,
        f"{contract.output_schema_version}.run_manifest",
        "run_manifest.json",
    )
    _require_schema(
        summary, f"{contract.output_schema_version}.summary", "summary.json"
    )
    _require_schema(
        events, f"{contract.output_schema_version}.events", "events.json"
    )
    if completion.get("status") != "COMPLETED":
        raise Phase4ArtifactValidationError("Completion receipt does not record success.")
    _nonempty_string(completion.get("completed_utc"), "completion.json.completed_utc")
    if manifest.get("status") != "COMPLETED" or summary.get("status") != "COMPLETED":
        raise Phase4ArtifactValidationError("Phase 4 result artifacts do not record success.")
    if returned_result.get("status") != "COMPLETED":
        raise Phase4ArtifactValidationError("Returned Phase 4 result does not record success.")

    run_id = _nonempty_string(completion.get("run_id"), "completion.json.run_id")
    for payload, label in (
        (manifest, "run_manifest.json"),
        (summary, "summary.json"),
        (events, "events.json"),
        (returned_result, "returned result"),
    ):
        if payload.get("run_id") != run_id:
            raise Phase4ArtifactValidationError(f"{label} run identity is inconsistent.")

    recorded_manifest_hash = _sha256(
        completion.get("run_manifest_sha256"),
        "completion.json.run_manifest_sha256",
    )
    if _sha256_file(manifest_path) != recorded_manifest_hash:
        raise Phase4ArtifactValidationError("Run-manifest SHA-256 does not match receipt.")

    completion_hashes = _hash_mapping(
        completion.get("artifacts_sha256"), "completion.json.artifacts_sha256"
    )
    manifest_hashes = _hash_mapping(
        manifest.get("artifacts_sha256"), "run_manifest.json.artifacts_sha256"
    )
    if completion_hashes != manifest_hashes:
        raise Phase4ArtifactValidationError("Artifact hash mappings are inconsistent.")

    processing_request = _mapping(
        manifest.get("processing_request"), "run_manifest.json.processing_request"
    )
    if processing_request.get("annotated_video_enabled") is not True:
        raise Phase4ArtifactValidationError(
            "Backend Phase 4 execution must include the annotated video."
        )
    required_artifacts = {
        contract.artifacts["summary"],
        contract.artifacts["events"],
        contract.artifacts["frame_detections"],
        contract.artifacts["annotated_video"],
    }
    if set(completion_hashes) != required_artifacts:
        raise Phase4ArtifactValidationError("Completion artifact set is incompatible.")
    for name, expected_hash in completion_hashes.items():
        artifact = output_directory / name
        if not artifact.is_file() or _sha256_file(artifact) != expected_hash:
            raise Phase4ArtifactValidationError("Completion artifact SHA-256 mismatch.")

    frames_inferred = _nonnegative_int(
        completion.get("frames_inferred"), "completion.json.frames_inferred"
    )
    event_count = _nonnegative_int(
        completion.get("event_count"), "completion.json.event_count"
    )
    summary_frames = _mapping(summary.get("frames"), "summary.json.frames")
    summary_events = _mapping(summary.get("events"), "summary.json.events")
    manifest_progress = _mapping(
        manifest.get("progress"), "run_manifest.json.progress"
    )
    if (
        summary_frames.get("inferred") != frames_inferred
        or manifest_progress.get("frames_inferred") != frames_inferred
        or returned_result.get("frames") != summary.get("frames")
    ):
        raise Phase4ArtifactValidationError("Frame totals are inconsistent.")
    if (
        summary_events.get("total") != event_count
        or manifest_progress.get("finalized_events") != event_count
        or events.get("event_count") != event_count
    ):
        raise Phase4ArtifactValidationError("Event totals are inconsistent.")

    model_sha = _sha256(completion.get("model_sha256"), "completion.json.model_sha256")
    input_sha = _sha256(completion.get("input_sha256"), "completion.json.input_sha256")
    validation = _mapping(manifest.get("validation"), "run_manifest.json.validation")
    if (
        _identity_sha(validation, "model", "run_manifest.json.validation") != model_sha
        or _identity_sha(summary, "model", "summary.json") != model_sha
    ):
        raise Phase4ArtifactValidationError("Model identity is inconsistent.")
    if (
        _identity_sha(validation, "input_video", "run_manifest.json.validation")
        != input_sha
        or _identity_sha(summary, "input_video", "summary.json") != input_sha
        or _sha256_file(staged_input) != input_sha
    ):
        raise Phase4ArtifactValidationError("Input-video identity is inconsistent.")

    for payload, label in (
        (completion, "completion.json"),
        (manifest, "run_manifest.json"),
        (summary, "summary.json"),
    ):
        for field in (
            "internal_test_accessed",
            "training_executed",
            "threshold_tuning_performed",
        ):
            if payload.get(field) is not False:
                raise Phase4ArtifactValidationError(f"{label}.{field} must be false.")

    return VerifiedPhase4TerminalState("COMPLETED", run_id, frames_inferred)


def validate_interrupted_phase4_run(
    output_directory: Path,
    returned_result: Mapping[str, Any],
) -> VerifiedPhase4TerminalState:
    """Verify the frozen Phase 4 interruption representation before mapping it."""
    contract = _load_frozen_contract()
    if (output_directory / RUN_LOCK_NAME).exists():
        raise Phase4ArtifactValidationError("Phase 4 run lock is still active.")
    if (output_directory / contract.artifacts["completion"]).exists():
        raise Phase4ArtifactValidationError("Interrupted run retained a completion receipt.")
    manifest = _load_json_object(
        output_directory / contract.artifacts["run_manifest"], "run_manifest.json"
    )
    summary = _load_json_object(
        output_directory / contract.artifacts["summary"], "summary.json"
    )
    _require_schema(
        manifest,
        f"{contract.output_schema_version}.run_manifest",
        "run_manifest.json",
    )
    _require_schema(
        summary, f"{contract.output_schema_version}.summary", "summary.json"
    )
    if (
        returned_result.get("status") != "INTERRUPTED"
        or manifest.get("status") != "INTERRUPTED"
        or summary.get("status") != "INTERRUPTED"
    ):
        raise Phase4ArtifactValidationError("Interrupted state is inconsistent.")
    run_id = _nonempty_string(summary.get("run_id"), "summary.json.run_id")
    if manifest.get("run_id") != run_id or returned_result.get("run_id") != run_id:
        raise Phase4ArtifactValidationError("Interrupted run identity is inconsistent.")
    frames = _nonnegative_int(
        _mapping(summary.get("frames"), "summary.json.frames").get("inferred"),
        "summary.json.frames.inferred",
    )
    if returned_result.get("frames") != summary.get("frames"):
        raise Phase4ArtifactValidationError("Interrupted frame totals are inconsistent.")
    return VerifiedPhase4TerminalState("INTERRUPTED", run_id, frames)

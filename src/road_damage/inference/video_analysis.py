"""Frozen YOLOv8s video inference with heuristic temporal damage events."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.aggregation.temporal import (  # noqa: E402
    AggregationConfig,
    AssociationDecision,
    DetectionObservation,
    TemporalDamageAggregator,
)
from road_damage.dataset.rdd2022_common import (  # noqa: E402
    RDD2022Error,
    sha256_file,
    write_json_atomic,
)
from road_damage.demo.mentor_demo import (  # noqa: E402
    CLASS_NAMES,
    FROZEN_CONFIDENCE,
    FROZEN_MODEL_RELATIVE_PATH,
    FROZEN_MODEL_SHA256,
    FROZEN_MODEL_SIZE_BYTES,
    FROZEN_NMS_IOU,
    Detection,
    MentorDemoError,
    _detections_from_result,
    _validate_model_class_names,
    verify_model_identity,
)
from road_damage.demo.video_demo import (  # noqa: E402
    VideoDemoError,
    VideoInfo,
    read_video_info,
    reject_redirects,
    timestamp_ms,
    validate_video_source,
    verify_encoded_video,
)
from road_damage.video.inspect_video import _codec_description, _fourcc_text  # noqa: E402


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("configs/inference/video_analysis_yolov8s.yaml")
SCHEMA_VERSION = "road_damage.video_analysis.v1"
EVENT_INTERPRETATION = (
    "Temporally aggregated damage events are deterministic operational heuristics, "
    "not ground-truth physical-world unique damages."
)
VIDEO_LIMITATION = (
    "The input video has no road-damage ground truth. Application outputs must not be "
    "reported as detector accuracy, recall, F1, mAP, or target-domain performance."
)
ANNOTATION_DISPLAY_NAMES: Mapping[int, str] = MappingProxyType(
    {
        0: "Longitudinal Crack",
        1: "Transverse Crack",
        2: "Alligator Crack",
        3: "Pothole",
    }
)


class VideoAnalysisError(RuntimeError):
    """Actionable configuration, input, inference, or output failure."""


EXPECTED_CONFIG: dict[str, Any] = {
    "schema_version": "road_damage.video_analysis_config.v1",
    "output_schema_version": SCHEMA_VERSION,
    "model": {
        "family": "YOLOv8s",
        "task": "detect",
        "path": FROZEN_MODEL_RELATIVE_PATH.as_posix(),
        "sha256": FROZEN_MODEL_SHA256,
        "size_bytes": FROZEN_MODEL_SIZE_BYTES,
    },
    "operating_point": {
        "status": "frozen_validation_selected",
        "confidence": FROZEN_CONFIDENCE,
        "nms_iou": FROZEN_NMS_IOU,
    },
    "inference": {
        "imgsz": 640,
        "max_detections": 300,
        "augment": False,
        "frame_stride": 1,
        "device": "auto",
    },
    "classes": {str(key): value for key, value in CLASS_NAMES.items()},
    "annotation": {
        "enabled_by_default": True,
        "show_event_id": True,
        "codec_fourcc": "mp4v",
        "container_suffix": ".mp4",
        "minimum_line_thickness": 2,
    },
    "temporal_aggregation": {
        "same_class_only": True,
        "minimum_iou": 0.20,
        "maximum_center_distance_normalized": 0.12,
        "minimum_area_ratio": 0.25,
        "maximum_gap_seconds": 0.20,
        "minimum_confirmed_observations": 3,
    },
    "progress_every_frames": 100,
    "artifacts": {
        "run_manifest": "run_manifest.json",
        "summary": "summary.json",
        "events": "events.json",
        "frame_detections": "frame_detections.jsonl",
        "completion": "completion.json",
        "annotated_video": "annotated_video.mp4",
    },
}


@dataclass(frozen=True)
class VideoAnalysisConfig:
    model_path: Path
    model_sha256: str
    model_size_bytes: int
    confidence: float
    nms_iou: float
    imgsz: int
    max_detections: int
    frame_stride: int
    device_policy: str
    annotation: Mapping[str, Any]
    aggregation: AggregationConfig
    progress_every_frames: int
    artifacts: Mapping[str, str]


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    manifest: Path
    summary: Path
    events: Path
    frame_detections: Path
    completion: Path
    annotated_video: Path
    lock: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Translate the shared atomic writer's dataset-specific boundary error."""
    try:
        write_json_atomic(path, value)
    except RDD2022Error as exc:
        raise VideoAnalysisError(f"Could not write video-analysis artifact: {path}") from exc


def _absolute(path: Path, root: Path = ROOT) -> Path:
    return Path(os.path.abspath(path if path.is_absolute() else root / path))


def load_config(root: Path = ROOT) -> VideoAnalysisConfig:
    """Load the JSON-compatible YAML and reject every frozen-value override."""
    path = _absolute(CONFIG_PATH, root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VideoAnalysisError(f"Video-analysis config does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoAnalysisError(f"Cannot read video-analysis config: {path}") from exc
    if raw != EXPECTED_CONFIG:
        raise VideoAnalysisError(
            "Video-analysis config changed a frozen model, operating point, class, "
            "inference, aggregation, annotation, or artifact contract value."
        )
    aggregation = raw["temporal_aggregation"]
    return VideoAnalysisConfig(
        Path(raw["model"]["path"]),
        raw["model"]["sha256"],
        raw["model"]["size_bytes"],
        raw["operating_point"]["confidence"],
        raw["operating_point"]["nms_iou"],
        raw["inference"]["imgsz"],
        raw["inference"]["max_detections"],
        raw["inference"]["frame_stride"],
        raw["inference"]["device"],
        dict(raw["annotation"]),
        AggregationConfig(
            aggregation["minimum_iou"],
            aggregation["maximum_center_distance_normalized"],
            aggregation["minimum_area_ratio"],
            aggregation["maximum_gap_seconds"],
            aggregation["minimum_confirmed_observations"],
        ),
        raw["progress_every_frames"],
        dict(raw["artifacts"]),
    )


def verify_frozen_checkpoint(
    config: VideoAnalysisConfig, root: Path = ROOT
) -> tuple[Path, str]:
    """Allow only the exact frozen YOLOv8s best.pt and fail closed on drift."""
    required = _absolute(FROZEN_MODEL_RELATIVE_PATH, root)
    candidate = _absolute(config.model_path, root)
    if candidate != required:
        raise VideoAnalysisError(f"Only the frozen YOLOv8s best.pt is allowed: {required}")
    try:
        reject_redirects(candidate)
        actual = verify_model_identity(
            candidate,
            expected_sha256=config.model_sha256,
            expected_size_bytes=config.model_size_bytes,
        )
    except (VideoDemoError, MentorDemoError, OSError) as exc:
        raise VideoAnalysisError(str(exc)) from exc
    return candidate, actual


def resolve_device() -> int | str:
    """Resolve runtime compute without changing any scientific inference setting."""
    try:
        import torch
    except ImportError as exc:
        raise VideoAnalysisError("PyTorch is unavailable; cannot resolve inference device.") from exc
    return 0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"


def validate_loaded_model(model: Any) -> dict[str, Any]:
    """Validate task/classes and inspect architecture metadata where exposed safely."""
    try:
        _validate_model_class_names(model.names)
    except (AttributeError, MentorDemoError) as exc:
        raise VideoAnalysisError(str(exc)) from exc
    try:
        task = model.task
    except AttributeError as exc:
        raise VideoAnalysisError(
            "Frozen checkpoint must expose an explicit task equal to 'detect'."
        ) from exc
    if task != "detect":
        raise VideoAnalysisError(f"Frozen checkpoint task must be detect, got {task!r}.")
    yaml = getattr(getattr(model, "model", None), "yaml", None)
    inspected: dict[str, Any] = {"task": task, "class_count": 4}
    if isinstance(yaml, Mapping):
        if "nc" in yaml and int(yaml["nc"]) != 4:
            raise VideoAnalysisError("Frozen checkpoint architecture class count is not four.")
        if "scale" in yaml and str(yaml["scale"]) != "s":
            raise VideoAnalysisError("Frozen checkpoint architecture scale is not YOLOv8s/small.")
        inspected.update(
            {key: yaml[key] for key in ("nc", "scale", "yaml_file") if key in yaml}
        )
    return inspected


def inspect_video_input(
    source: Path, *, root: Path = ROOT, cv2_module: Any = None
) -> dict[str, Any]:
    """Validate and fingerprint a readable video without modifying it."""
    try:
        resolved = validate_video_source(source, root)
    except (VideoDemoError, OSError) as exc:
        raise VideoAnalysisError(str(exc)) from exc
    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise VideoAnalysisError("OpenCV is unavailable for video validation.") from exc
    cv2 = cv2_module
    before = resolved.stat()
    source_sha256 = sha256_file(resolved)
    capture = cv2.VideoCapture(str(resolved))
    try:
        if not capture.isOpened():
            raise VideoAnalysisError(f"OpenCV cannot open the input video: {resolved}")
        try:
            info = read_video_info(capture, cv2)
        except VideoDemoError as exc:
            raise VideoAnalysisError(str(exc)) from exc
        fourcc = _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))
        ok, frame = capture.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            raise VideoAnalysisError("Input video has no decodable frames.")
        if frame.shape[:2] != (info.height, info.width):
            raise VideoAnalysisError("Decoded dimensions differ from reported video metadata.")
        try:
            backend = capture.getBackendName()
        except (AttributeError, cv2.error):
            backend = "unknown"
    finally:
        capture.release()
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise VideoAnalysisError("Input video changed while it was being validated.")
    return {
        "absolute_path": str(resolved),
        "filename": resolved.name,
        "size_bytes": before.st_size,
        "sha256": source_sha256,
        "container": resolved.suffix.lstrip(".").lower() or "unknown",
        "codec_fourcc": fourcc or "unknown",
        "codec_description": _codec_description(fourcc),
        "decoder_backend": backend,
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frame_count": info.frame_count,
        "duration_seconds": (
            info.frame_count / info.fps if info.frame_count is not None else None
        ),
        "timestamp_policy": (
            "zero-based original source frame index / nominal source FPS, rounded to integer milliseconds"
        ),
        "size_and_mtime_unchanged_during_validation": True,
    }


def validate_output_path(output_dir: Path, source: Path, root: Path = ROOT) -> Path:
    """Check an empty/nonexistent dedicated run directory without retaining a probe."""
    output = _absolute(output_dir, root)
    source = _absolute(source, root)
    lowered = {part.casefold() for part in output.parts}
    if lowered.intersection(
        {"test", "internal_test", "internal-test", "test_images", "testimages"}
    ):
        raise VideoAnalysisError("Internal/official-test paths are forbidden as outputs.")
    for subtree in ("data/raw", "data/processed", "data/exports"):
        if output.is_relative_to(_absolute(Path(subtree), root)):
            raise VideoAnalysisError("Dataset directories are forbidden as analysis outputs.")
    if output in (source, source.parent, _absolute(Path("."), root)):
        raise VideoAnalysisError("Output must be a dedicated directory separate from the source.")
    try:
        reject_redirects(output)
    except VideoDemoError as exc:
        raise VideoAnalysisError(str(exc)) from exc
    if output.exists():
        if not output.is_dir():
            raise VideoAnalysisError(f"Output path exists and is not a directory: {output}")
        try:
            next(output.iterdir())
        except StopIteration:
            pass
        else:
            raise VideoAnalysisError(f"Refusing to overwrite non-empty output: {output}")
    ancestor = output if output.exists() else output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise VideoAnalysisError(f"Output ancestor is not a directory: {ancestor}")
    try:
        with tempfile.TemporaryFile(dir=ancestor) as probe:
            probe.write(b"road-damage-video-analysis-output-probe")
    except OSError as exc:
        raise VideoAnalysisError(f"Output location is not writable: {ancestor}") from exc
    return output


def preflight(
    source: Path,
    output_dir: Path,
    *,
    root: Path = ROOT,
    cv2_module: Any = None,
) -> dict[str, Any]:
    """Validate config, checkpoint, video, device, and output; never load YOLO."""
    config = load_config(root)
    model_path, model_sha256 = verify_frozen_checkpoint(config, root)
    input_metadata = inspect_video_input(source, root=root, cv2_module=cv2_module)
    output = validate_output_path(output_dir, Path(input_metadata["absolute_path"]), root)
    device = resolve_device()
    return {
        "schema_version": f"{SCHEMA_VERSION}.preflight",
        "status": "PASSED",
        "model_inference_executed": False,
        "internal_test_accessed": False,
        "config_path": str(_absolute(CONFIG_PATH, root)),
        "config_sha256": sha256_file(_absolute(CONFIG_PATH, root)),
        "model": {
            "family": "YOLOv8s",
            "path": str(model_path),
            "sha256": model_sha256,
            "size_bytes": config.model_size_bytes,
        },
        "class_mapping": CLASS_NAMES,
        "operating_point": {
            "confidence": config.confidence,
            "nms_iou": config.nms_iou,
            "status": "frozen_validation_selected",
        },
        "input_video": input_metadata,
        "output_directory": str(output),
        "output_ready": True,
        "resolved_device": device,
    }


def reserve_run_directory(output: Path, config: VideoAnalysisConfig) -> RunPaths:
    """Reserve the exact dedicated directory and refuse any pre-existing content."""
    created = False
    try:
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
            created = True
        elif any(output.iterdir()):
            raise VideoAnalysisError(f"Refusing to use non-empty output: {output}")
        lock = output / ".run.lock"
        with lock.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("reserved\n")
        artifacts = config.artifacts
        return RunPaths(
            output,
            output / artifacts["run_manifest"],
            output / artifacts["summary"],
            output / artifacts["events"],
            output / artifacts["frame_detections"],
            output / artifacts["completion"],
            output / artifacts["annotated_video"],
            lock,
        )
    except OSError as exc:
        if created:
            try:
                output.rmdir()
            except OSError:
                pass
        raise VideoAnalysisError(f"Could not reserve output directory: {output}") from exc


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def collect_provenance(root: Path, config: VideoAnalysisConfig) -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for package in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python-headless"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    source_paths = (
        _absolute(CONFIG_PATH, root),
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "aggregation" / "temporal.py",
    )
    return {
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_status_short": _git(root, "status", "--short"),
        "environment": versions,
        "source_sha256": {
            str(path.relative_to(root) if path.is_relative_to(root) else path): sha256_file(path)
            for path in source_paths
        },
        "resolved_config": {
            **asdict(config),
            "model_path": config.model_path.as_posix(),
        },
    }


def observation_from_detection(
    detection: Detection,
    detection_index: int,
    frame_index: int,
    fps: float,
    width: int,
    height: int,
) -> DetectionObservation:
    """Build the explicit boundary schema from one decoded model detection."""
    x1, y1, x2, y2 = (float(value) for value in detection.bbox_xyxy_pixel)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return DetectionObservation(
        detection_index,
        frame_index,
        timestamp_ms(frame_index, fps),
        detection.class_id,
        detection.class_name,
        detection.confidence,
        (x1, y1, x2, y2),
        (x1 / width, y1 / height, x2 / width, y2 / height),
        area,
        area / (width * height),
    )


def frame_record(
    frame_index: int,
    fps: float,
    width: int,
    height: int,
    observations: Sequence[DetectionObservation],
    decisions: Sequence[AssociationDecision],
) -> dict[str, Any]:
    """Serialize one inferred source frame in stable detection-index order."""
    by_index = {decision.detection_index: decision for decision in decisions}
    if set(by_index) != {item.detection_index for item in observations}:
        raise VideoAnalysisError("Every direct detection must receive one event association.")
    timestamp = timestamp_ms(frame_index, fps)
    return {
        "schema_version": f"{SCHEMA_VERSION}.frame",
        "frame_index": frame_index,
        "timestamp_ms": timestamp,
        "timestamp_seconds": timestamp / 1000,
        "source_width": width,
        "source_height": height,
        "detections": [
            item.to_record(
                by_index[item.detection_index].event_id,
                by_index[item.detection_index].to_record(),
            )
            for item in sorted(observations, key=lambda value: value.detection_index)
        ],
    }


def jsonl_bytes(value: Mapping[str, Any]) -> bytes:
    """Deterministic compact JSONL serialization for long frame streams."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def render_frame(
    frame: Any,
    observations: Sequence[DetectionObservation],
    decisions: Sequence[AssociationDecision],
    config: VideoAnalysisConfig,
    cv2: Any,
) -> Any:
    """Render the exact detections used by aggregation; inference is never repeated."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    thickness = max(
        int(config.annotation["minimum_line_thickness"]),
        int(round(min(width, height) / 320)),
    )
    font_scale = max(0.45, min(0.85, width / 1100))
    colors = {0: (0, 220, 255), 1: (255, 180, 0), 2: (0, 210, 0), 3: (30, 30, 255)}
    by_index = {item.detection_index: item for item in decisions}
    for observation in observations:
        decision = by_index[observation.detection_index]
        x1, y1, x2, y2 = (round(value) for value in observation.bbox_xyxy_pixel)
        color = colors[observation.class_id]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = (
            f"{ANNOTATION_DISPLAY_NAMES[observation.class_id]}"
            f" | {observation.confidence:.2f}"
        )
        if config.annotation["show_event_id"]:
            label += f" | {decision.event_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        top = max(0, y1 - text_height - baseline - 8)
        right = min(width - 1, x1 + text_width + 8)
        cv2.rectangle(annotated, (x1, top), (right, y1), color, cv2.FILLED)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(text_height + 2, y1 - baseline - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def _event_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    categories = {
        key: {name: 0 for name in CLASS_NAMES.values()}
        for key in ("all", "confirmed", "tentative")
    }
    for event in events:
        name = str(event["class_name"])
        categories["all"][name] += 1
        categories[str(event["status"])][name] += 1
    return categories


def _open_writer(
    path: Path, info: VideoInfo, config: VideoAnalysisConfig, cv2: Any
) -> Any:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*str(config.annotation["codec_fourcc"])),
        info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        writer.release()
        raise VideoAnalysisError(
            f"VideoWriter could not open {path} with {config.annotation['codec_fourcc']}."
        )
    return writer


def _verify_persisted_outputs(
    paths: RunPaths,
    expected_summary: Mapping[str, Any],
    annotated_video_enabled: bool,
) -> dict[str, str]:
    """Stream, reconcile, and hash every closed completion dependency."""

    def fail(message: str) -> None:
        raise VideoAnalysisError(f"Persisted output verification failed: {message}")

    def reject_constant(value: str) -> None:
        fail(f"non-finite JSON numeric constant {value!r} is not allowed.")

    def load_json_object(path: Path, label: str) -> Mapping[str, Any]:
        try:
            value = json.loads(
                path.read_text(encoding="utf-8"), parse_constant=reject_constant
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise VideoAnalysisError(
                f"Persisted output verification failed: {label} could not be read."
            ) from exc
        if not isinstance(value, Mapping):
            fail(f"{label} must contain one JSON object.")
        return value

    def mapping_at(value: Any, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            fail(f"{label} must be an object.")
        return value

    def list_at(value: Any, label: str) -> list[Any]:
        if not isinstance(value, list):
            fail(f"{label} must be an array.")
        return value

    def nonnegative_int(value: Any, label: str, *, positive: bool = False) -> int:
        minimum = 1 if positive else 0
        if type(value) is not int or value < minimum:
            qualifier = "positive" if positive else "non-negative"
            fail(f"{label} must be a {qualifier} integer.")
        return value

    def finite_number(
        value: Any,
        label: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(f"{label} must be numeric.")
        number = float(value)
        if not math.isfinite(number):
            fail(f"{label} must be finite.")
        if minimum is not None and number < minimum:
            fail(f"{label} must be at least {minimum}.")
        if maximum is not None and number > maximum:
            fail(f"{label} must be at most {maximum}.")
        return number

    def valid_bbox(value: Any, label: str) -> None:
        coordinates = list_at(value, label)
        if len(coordinates) != 4:
            fail(f"{label} must contain four coordinates.")
        for index, coordinate in enumerate(coordinates):
            finite_number(coordinate, f"{label}[{index}]")

    def valid_event_id(value: Any, label: str) -> str:
        if not isinstance(value, str) or re.fullmatch(r"RD[0-9]{4,}", value) is None:
            fail(f"{label} is not a valid RD-prefixed event ID.")
        if int(value[2:]) < 1:
            fail(f"{label} must have a positive numeric sequence.")
        return value

    def valid_class(record: Mapping[str, Any], label: str) -> tuple[int, str]:
        class_id = record.get("class_id")
        if type(class_id) is not int or class_id not in CLASS_NAMES:
            fail(f"{label}.class_id must be one of 0,1,2,3.")
        class_name = record.get("class_name")
        if class_name != CLASS_NAMES[class_id]:
            fail(
                f"{label}.class_name does not match class ID {class_id}: "
                f"expected {CLASS_NAMES[class_id]!r}."
            )
        return class_id, class_name

    expected_class_names = set(CLASS_NAMES.values())

    def class_counts(value: Any, label: str) -> dict[str, int]:
        counts = mapping_at(value, label)
        if set(counts) != expected_class_names:
            fail(f"{label} must contain exactly the four frozen class names.")
        return {
            name: nonnegative_int(counts[name], f"{label}.{name}")
            for name in CLASS_NAMES.values()
        }

    def summary_snapshot(value: Mapping[str, Any], label: str) -> dict[str, Any]:
        frames = mapping_at(value.get("frames"), f"{label}.frames")
        raw = mapping_at(
            value.get("raw_detection_observations"),
            f"{label}.raw_detection_observations",
        )
        events_summary = mapping_at(value.get("events"), f"{label}.events")
        snapshot = {
            "frames_inferred": nonnegative_int(
                frames.get("inferred"), f"{label}.frames.inferred"
            ),
            "raw_total": nonnegative_int(
                raw.get("total"), f"{label}.raw_detection_observations.total"
            ),
            "raw_by_class": class_counts(
                raw.get("by_class"),
                f"{label}.raw_detection_observations.by_class",
            ),
            "event_total": nonnegative_int(
                events_summary.get("total"), f"{label}.events.total"
            ),
            "event_by_class": class_counts(
                events_summary.get("by_class"), f"{label}.events.by_class"
            ),
            "confirmed_total": nonnegative_int(
                events_summary.get("confirmed_total"),
                f"{label}.events.confirmed_total",
            ),
            "confirmed_by_class": class_counts(
                events_summary.get("confirmed_by_class"),
                f"{label}.events.confirmed_by_class",
            ),
            "tentative_total": nonnegative_int(
                events_summary.get("tentative_total"),
                f"{label}.events.tentative_total",
            ),
            "tentative_by_class": class_counts(
                events_summary.get("tentative_by_class"),
                f"{label}.events.tentative_by_class",
            ),
        }
        if sum(snapshot["raw_by_class"].values()) != snapshot["raw_total"]:
            fail(f"{label} raw total does not equal its per-class counts.")
        if sum(snapshot["event_by_class"].values()) != snapshot["event_total"]:
            fail(f"{label} event total does not equal its per-class counts.")
        if snapshot["confirmed_total"] + snapshot["tentative_total"] != snapshot[
            "event_total"
        ]:
            fail(f"{label} confirmed and tentative totals do not equal all events.")
        if sum(snapshot["confirmed_by_class"].values()) != snapshot[
            "confirmed_total"
        ]:
            fail(f"{label} confirmed total does not equal its per-class counts.")
        if sum(snapshot["tentative_by_class"].values()) != snapshot[
            "tentative_total"
        ]:
            fail(f"{label} tentative total does not equal its per-class counts.")
        for name in CLASS_NAMES.values():
            if (
                snapshot["confirmed_by_class"][name]
                + snapshot["tentative_by_class"][name]
                != snapshot["event_by_class"][name]
            ):
                fail(f"{label} status counts do not reconcile for {name}.")
        return snapshot

    summary = load_json_object(paths.summary, "summary.json")
    events_payload = load_json_object(paths.events, "events.json")
    persisted_summary = summary_snapshot(summary, "summary.json")
    expected_snapshot = summary_snapshot(expected_summary, "in-memory run state")
    if persisted_summary != expected_snapshot:
        fail("summary.json contradicts the in-memory run state.")

    events = list_at(events_payload.get("events"), "events.json.events")
    declared_event_count = nonnegative_int(
        events_payload.get("event_count"), "events.json.event_count"
    )
    if declared_event_count != len(events):
        fail("events.json event_count does not equal its event-array length.")
    persisted_event_ids: set[str] = set()
    for event_offset, event_value in enumerate(events):
        event_label = f"events.json.events[{event_offset}]"
        event = mapping_at(event_value, event_label)
        event_id = valid_event_id(event.get("event_id"), f"{event_label}.event_id")
        if event_id in persisted_event_ids:
            fail(f"duplicate event ID {event_id!r} exists in events.json.")
        persisted_event_ids.add(event_id)

    frame_row_count = 0
    raw_detection_total = 0
    raw_detection_count_by_class = {
        name: 0 for name in CLASS_NAMES.values()
    }
    referenced_event_ids: set[str] = set()
    try:
        with paths.frame_detections.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    fail(f"frame_detections.jsonl line {line_number} is blank.")
                try:
                    row = json.loads(line, parse_constant=reject_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise VideoAnalysisError(
                        "Persisted output verification failed: "
                        f"frame_detections.jsonl line {line_number} is invalid JSON."
                    ) from exc
                row = mapping_at(row, f"frame_detections.jsonl line {line_number}")
                nonnegative_int(row.get("frame_index"), f"frame row {line_number}.frame_index")
                nonnegative_int(row.get("timestamp_ms"), f"frame row {line_number}.timestamp_ms")
                finite_number(
                    row.get("timestamp_seconds"),
                    f"frame row {line_number}.timestamp_seconds",
                    minimum=0,
                )
                nonnegative_int(
                    row.get("source_width"),
                    f"frame row {line_number}.source_width",
                    positive=True,
                )
                nonnegative_int(
                    row.get("source_height"),
                    f"frame row {line_number}.source_height",
                    positive=True,
                )
                detections = list_at(
                    row.get("detections"), f"frame row {line_number}.detections"
                )
                frame_row_count += 1
                for detection_offset, detection_value in enumerate(detections):
                    detection_label = (
                        f"frame row {line_number}.detections[{detection_offset}]"
                    )
                    detection = mapping_at(detection_value, detection_label)
                    _, class_name = valid_class(detection, detection_label)
                    raw_detection_total += 1
                    raw_detection_count_by_class[class_name] += 1
                    referenced_event_id = valid_event_id(
                        detection.get("event_id"), f"{detection_label}.event_id"
                    )
                    if referenced_event_id not in persisted_event_ids:
                        fail(
                            "frame_detections.jsonl references unknown event ID "
                            f"{referenced_event_id!r}."
                        )
                    referenced_event_ids.add(referenced_event_id)
                    nonnegative_int(
                        detection.get("detection_index"),
                        f"{detection_label}.detection_index",
                    )
                    finite_number(
                        detection.get("confidence"),
                        f"{detection_label}.confidence",
                        minimum=0,
                        maximum=1,
                    )
                    valid_bbox(
                        detection.get("bbox_xyxy_pixel"),
                        f"{detection_label}.bbox_xyxy_pixel",
                    )
                    valid_bbox(
                        detection.get("bbox_xyxy_normalized"),
                        f"{detection_label}.bbox_xyxy_normalized",
                    )
                    finite_number(
                        detection.get("bbox_area_pixels"),
                        f"{detection_label}.bbox_area_pixels",
                        minimum=0,
                    )
                    finite_number(
                        detection.get("bbox_relative_area"),
                        f"{detection_label}.bbox_relative_area",
                        minimum=0,
                    )
                    association = mapping_at(
                        detection.get("association"), f"{detection_label}.association"
                    )
                    for numeric_name in (
                        "iou_with_previous",
                        "center_distance_normalized",
                        "area_ratio",
                    ):
                        numeric_value = association.get(numeric_name)
                        if numeric_value is not None:
                            finite_number(
                                numeric_value,
                                f"{detection_label}.association.{numeric_name}",
                                minimum=0,
                            )
    except (OSError, UnicodeError) as exc:
        raise VideoAnalysisError(
            "Persisted output verification failed: frame_detections.jsonl could not be streamed."
        ) from exc

    event_by_class = {name: 0 for name in CLASS_NAMES.values()}
    confirmed_by_class = {name: 0 for name in CLASS_NAMES.values()}
    tentative_by_class = {name: 0 for name in CLASS_NAMES.values()}
    for event_offset, event_value in enumerate(events):
        event_label = f"events.json.events[{event_offset}]"
        event = mapping_at(event_value, event_label)
        event_id = valid_event_id(event.get("event_id"), f"{event_label}.event_id")
        _, class_name = valid_class(event, event_label)
        status = event.get("status")
        if status not in ("confirmed", "tentative"):
            fail(f"{event_label}.status must be confirmed or tentative.")
        event_by_class[class_name] += 1
        (confirmed_by_class if status == "confirmed" else tentative_by_class)[
            class_name
        ] += 1
        for field in (
            "first_frame",
            "last_frame",
            "first_timestamp_ms",
            "last_timestamp_ms",
            "representative_frame",
            "representative_timestamp_ms",
        ):
            nonnegative_int(event.get(field), f"{event_label}.{field}")
        nonnegative_int(
            event.get("observation_count"),
            f"{event_label}.observation_count",
            positive=True,
        )
        for field in (
            "first_timestamp_seconds",
            "last_timestamp_seconds",
            "representative_timestamp_seconds",
            "duration_seconds",
            "maximum_bbox_area_pixels",
            "maximum_bbox_relative_area",
            "mean_bbox_relative_area",
        ):
            finite_number(event.get(field), f"{event_label}.{field}", minimum=0)
        for field in ("mean_confidence", "max_confidence"):
            finite_number(
                event.get(field), f"{event_label}.{field}", minimum=0, maximum=1
            )
        for field in (
            "representative_bbox_xyxy_pixel",
            "representative_bbox_xyxy_normalized",
            "last_bbox_xyxy_pixel",
        ):
            valid_bbox(event.get(field), f"{event_label}.{field}")
        statistics = mapping_at(
            event.get("association_statistics"),
            f"{event_label}.association_statistics",
        )
        matched = nonnegative_int(
            statistics.get("matched_observations"),
            f"{event_label}.association_statistics.matched_observations",
        )
        iou_matches = nonnegative_int(
            statistics.get("iou_rule_matches"),
            f"{event_label}.association_statistics.iou_rule_matches",
        )
        center_matches = nonnegative_int(
            statistics.get("center_scale_rule_matches"),
            f"{event_label}.association_statistics.center_scale_rule_matches",
        )
        if iou_matches + center_matches != matched:
            fail(f"{event_label} association statistics do not reconcile.")
        if matched != event["observation_count"] - 1:
            fail(f"{event_label} matched observations contradict observation_count.")

    confirmed_total = sum(confirmed_by_class.values())
    tentative_total = sum(tentative_by_class.values())
    recomputed = {
        "frames_inferred": frame_row_count,
        "raw_total": raw_detection_total,
        "raw_by_class": raw_detection_count_by_class,
        "event_total": len(events),
        "event_by_class": event_by_class,
        "confirmed_total": confirmed_total,
        "confirmed_by_class": confirmed_by_class,
        "tentative_total": tentative_total,
        "tentative_by_class": tentative_by_class,
    }
    if recomputed != persisted_summary:
        fail("summary totals contradict recomputed frame/event artifacts.")

    artifacts = {
        paths.summary.name: sha256_file(paths.summary),
        paths.events.name: sha256_file(paths.events),
        paths.frame_detections.name: sha256_file(paths.frame_detections),
    }
    if annotated_video_enabled:
        artifacts[paths.annotated_video.name] = sha256_file(paths.annotated_video)
    return artifacts


def _invalidate_completion_receipt(paths: RunPaths) -> None:
    """Ensure a published success receipt cannot survive a late terminal error."""
    try:
        paths.completion.unlink(missing_ok=True)
        return
    except Exception as unlink_error:
        if not paths.completion.exists():
            return
        invalidated = paths.completion.with_name("completion.invalidated.json")
        try:
            os.replace(paths.completion, invalidated)
            try:
                invalidated.unlink(missing_ok=True)
            except Exception:
                LOGGER.error(
                    "Invalidated completion receipt remains quarantined at %s.",
                    invalidated,
                )
            return
        except Exception as replace_error:
            raise VideoAnalysisError(
                "CRITICAL: completion.json was published but could not be invalidated "
                f"after a late terminal error (unlink: {unlink_error}; "
                f"quarantine rename: {replace_error})."
            ) from replace_error


def _persist_late_terminal_state(
    paths: RunPaths,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    *,
    status: str,
    stop_reason: str,
    error: BaseException,
) -> None:
    """Invalidate success first, then publish one non-success terminal state."""
    if status not in ("INTERRUPTED", "FAILED_TECHNICAL"):
        raise VideoAnalysisError(f"Unsupported late terminal status: {status!r}.")
    error_text = f"{type(error).__name__}: {error}"
    try:
        _invalidate_completion_receipt(paths)
    except Exception as invalidation_error:
        status = "FAILED_TECHNICAL"
        stop_reason = "completion_invalidation_failure"
        error_text = f"{error_text}; {invalidation_error}"
        summary["status"] = status
        summary["error"] = error_text
        summary["frames"]["stop_reason"] = stop_reason
        manifest["status"] = status
        manifest["stop_reason"] = stop_reason
        manifest["error"] = error_text
        manifest.pop("artifacts_sha256", None)
        _write_json(paths.summary, summary)
        _write_json(paths.manifest, manifest)
        raise

    summary["status"] = status
    summary["error"] = error_text
    summary["frames"]["stop_reason"] = stop_reason
    manifest["status"] = status
    manifest["stop_reason"] = stop_reason
    manifest["error"] = error_text
    manifest.pop("artifacts_sha256", None)
    _write_json(paths.summary, summary)
    _write_json(paths.manifest, manifest)


def analyze_video(
    source: Path,
    output_dir: Path,
    *,
    max_frames: int | None = None,
    start_frame: int = 0,
    no_annotated_video: bool = False,
    root: Path = ROOT,
    stop_event: threading.Event | None = None,
    yolo_factory: Callable[..., Any] | None = None,
    cv2_module: Any = None,
) -> dict[str, Any]:
    """Application service used by the CLI and future backend orchestration."""
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise VideoAnalysisError("--max-frames must be a positive integer.")
    if type(start_frame) is not int or start_frame < 0:
        raise VideoAnalysisError("--start-frame must be a non-negative integer.")
    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise VideoAnalysisError("OpenCV is unavailable for video analysis.") from exc
    cv2 = cv2_module
    checked = preflight(source, output_dir, root=root, cv2_module=cv2)
    config = load_config(root)
    metadata = checked["input_video"]
    if metadata["frame_count"] is not None and start_frame >= metadata["frame_count"]:
        raise VideoAnalysisError(
            f"--start-frame {start_frame} is outside {metadata['frame_count']} source frames."
        )
    output = Path(checked["output_directory"])
    paths = reserve_run_directory(output, config)
    stop = stop_event if stop_event is not None else threading.Event()
    run_id = output.name
    started_utc = utc_now()
    started_perf = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}.run_manifest",
        "run_id": run_id,
        "status": "STARTED",
        "started_utc": started_utc,
        "validation": checked,
        "processing_request": {
            "start_frame": start_frame,
            "max_frames": max_frames,
            "frame_stride": config.frame_stride,
            "annotated_video_enabled": not no_annotated_video,
        },
        "temporal_aggregation": {
            **asdict(config.aggregation),
            "same_class_only": True,
            "assignment": "global greedy one-to-one; IoU descending, center distance ascending, area ratio descending, event then detection ID",
        },
        "provenance": collect_provenance(root, config),
        "progress": {
            "frames_decoded": 0,
            "frames_inferred": 0,
            "raw_detection_observations": 0,
            "last_source_frame": None,
        },
        "internal_test_accessed": False,
        "threshold_tuning_performed": False,
        "training_executed": False,
    }
    _write_json(paths.manifest, manifest)

    capture = writer = frame_handle = None
    aggregator: TemporalDamageAggregator | None = None
    frames_decoded = frames_inferred = raw_total = frames_with_detections = 0
    raw_by_class = {name: 0 for name in CLASS_NAMES.values()}
    inference_seconds = 0.0
    last_frame_index: int | None = None
    final_reason = "initialization_failed"
    failure: BaseException | None = None
    interrupted = False
    loaded_model_identity: dict[str, Any] | None = None
    encoding_verification: dict[str, Any] | None = None
    source_before = Path(metadata["absolute_path"]).stat()
    info: VideoInfo | None = None
    model_path = Path(checked["model"]["path"])
    try:
        capture = cv2.VideoCapture(metadata["absolute_path"])
        if not capture.isOpened():
            raise VideoAnalysisError("Input video became unreadable after preflight.")
        info = read_video_info(capture, cv2)
        current_identity = (info.width, info.height, info.fps, info.frame_count)
        expected_identity = (
            metadata["width"], metadata["height"], metadata["fps"], metadata["frame_count"]
        )
        if current_identity != expected_identity:
            raise VideoAnalysisError("Input video metadata changed after preflight.")
        if start_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise VideoAnalysisError(f"Decoder could not seek to --start-frame {start_frame}.")
        if start_frame:
            reported_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if (
                math.isfinite(reported_position)
                and abs(reported_position - start_frame) > 1
            ):
                raise VideoAnalysisError(
                    "Decoder did not seek to the requested original source frame: "
                    f"requested {start_frame}, reported {reported_position}."
                )
        if not no_annotated_video:
            writer = _open_writer(paths.annotated_video, info, config, cv2)
        frame_handle = paths.frame_detections.open("xb")
        aggregator = TemporalDamageAggregator(config.aggregation, info.fps)
        if stop.is_set():
            interrupted = True
            final_reason = "stop_requested_before_inference"
        else:
            if yolo_factory is None:
                from ultralytics import YOLO

                yolo_factory = YOLO
            model = yolo_factory(str(model_path), task="detect")
            loaded_model_identity = validate_loaded_model(model)
            frame_index = start_frame
            while not interrupted:
                if stop.is_set():
                    interrupted = True
                    final_reason = "stop_requested"
                    break
                if max_frames is not None and frames_inferred >= max_frames:
                    final_reason = "max_frames_reached"
                    break
                ok, frame = capture.read()
                if not ok or frame is None:
                    expected_end = info.frame_count is None or frame_index >= info.frame_count
                    if not expected_end:
                        raise VideoAnalysisError(
                            f"Decode ended at source frame {frame_index} before reported frame count {info.frame_count}."
                        )
                    final_reason = "end_of_video"
                    break
                frames_decoded += 1
                if frame.shape[:2] != (info.height, info.width):
                    raise VideoAnalysisError("Source resolution changed during processing.")
                inference_started = time.perf_counter()
                prediction = model.predict(
                    source=frame,
                    imgsz=config.imgsz,
                    device=checked["resolved_device"],
                    conf=config.confidence,
                    iou=config.nms_iou,
                    augment=False,
                    max_det=config.max_detections,
                    save=False,
                    save_txt=False,
                    save_crop=False,
                    show=False,
                    verbose=False,
                    stream=False,
                )
                inference_seconds += time.perf_counter() - inference_started
                if len(prediction) != 1:
                    raise VideoAnalysisError("Expected one model result for one input frame.")
                detections = _detections_from_result(
                    prediction[0], info.width, info.height
                )
                observations = tuple(
                    observation_from_detection(
                        detection,
                        index,
                        frame_index,
                        info.fps,
                        info.width,
                        info.height,
                    )
                    for index, detection in enumerate(detections)
                )
                decisions = aggregator.process_frame(frame_index, observations)
                record = frame_record(
                    frame_index,
                    info.fps,
                    info.width,
                    info.height,
                    observations,
                    decisions,
                )
                frame_handle.write(jsonl_bytes(record))
                if writer is not None:
                    writer.write(render_frame(frame, observations, decisions, config, cv2))
                frames_inferred += 1
                last_frame_index = frame_index
                raw_total += len(observations)
                frames_with_detections += bool(observations)
                for observation in observations:
                    raw_by_class[observation.class_name] += 1
                if (
                    frames_inferred == 1
                    or frames_inferred % config.progress_every_frames == 0
                ):
                    frame_handle.flush()
                    manifest["progress"] = {
                        "frames_decoded": frames_decoded,
                        "frames_inferred": frames_inferred,
                        "raw_detection_observations": raw_total,
                        "last_source_frame": last_frame_index,
                        "active_event_ids": list(aggregator.active_event_ids()),
                        "updated_utc": utc_now(),
                    }
                    _write_json(paths.manifest, manifest)
                    elapsed = time.perf_counter() - started_perf
                    LOGGER.info(
                        "Inferred %d frame(s); source frame %d; elapsed %.1fs; effective inference FPS %.2f",
                        frames_inferred,
                        frame_index,
                        elapsed,
                        frames_inferred / inference_seconds if inference_seconds else 0.0,
                    )
                frame_index += config.frame_stride
            if stop.is_set() and final_reason not in ("max_frames_reached", "end_of_video"):
                interrupted = True
                final_reason = "stop_requested"
    except KeyboardInterrupt:
        interrupted = True
        final_reason = "keyboard_interrupt"
    except Exception as exc:
        failure = exc
        final_reason = "technical_failure"
    finally:
        cleanup_errors: list[str] = []
        if aggregator is not None:
            try:
                aggregator.finalize_all(
                    "interrupted_or_failed" if failure or interrupted else final_reason
                )
            except Exception as exc:
                cleanup_errors.append(f"temporal event finalization: {exc}")
        for label, resource, method in (
            ("VideoCapture", capture, "release"),
            ("VideoWriter", writer, "release"),
            ("frame detections", frame_handle, "close"),
        ):
            if resource is None:
                continue
            try:
                getattr(resource, method)()
            except Exception as exc:
                cleanup_errors.append(f"{label}: {exc}")
        if cleanup_errors:
            failure = failure or VideoAnalysisError(
                "Resource finalization failed: " + "; ".join(cleanup_errors)
            )
            final_reason = "resource_finalization_failure"

    try:
        event_records = aggregator.event_records() if aggregator is not None else []
    except Exception as exc:
        failure = failure or exc
        final_reason = "temporal_event_serialization_failure"
        event_records = []
    event_counts = _event_counts(event_records)
    elapsed_seconds = time.perf_counter() - started_perf
    full_source_processed = (
        final_reason == "end_of_video" and start_frame == 0 and max_frames is None
    )
    warnings = [VIDEO_LIMITATION, EVENT_INTERPRETATION]
    if not full_source_processed:
        warnings.append(
            "Only the requested/available processing scope was analyzed; this is not a full-video result."
        )
    if not no_annotated_video:
        warnings.append("Annotated output contains video only; source audio is not preserved.")
    status = (
        "FAILED_TECHNICAL"
        if failure
        else ("INTERRUPTED" if interrupted else "FINALIZING")
    )
    summary = {
        "schema_version": f"{SCHEMA_VERSION}.summary",
        "run_id": run_id,
        "status": status,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "input_video": metadata,
        "model": {
            **checked["model"],
            "loaded_model_identity": loaded_model_identity,
        },
        "operating_point": checked["operating_point"],
        "processing_configuration": {
            "imgsz": config.imgsz,
            "max_detections": config.max_detections,
            "device": checked["resolved_device"],
            "frame_stride": config.frame_stride,
            "start_frame": start_frame,
            "max_frames": max_frames,
            "annotated_video_enabled": not no_annotated_video,
            "aggregation": {
                **asdict(config.aggregation),
                "maximum_gap_frames_at_source_fps": (
                    aggregator.max_gap_frames if aggregator is not None else None
                ),
            },
        },
        "frames": {
            "decoded": frames_decoded,
            "inferred": frames_inferred,
            "skipped_before_start": start_frame,
            "skipped_by_stride": 0,
            "with_detections": frames_with_detections,
            "last_source_frame": last_frame_index,
            "full_source_processed": full_source_processed,
            "stop_reason": final_reason,
        },
        "raw_detection_observations": {
            "total": raw_total,
            "by_class": raw_by_class,
        },
        "events": {
            "total": len(event_records),
            "by_class": event_counts["all"],
            "confirmed_total": sum(event["status"] == "confirmed" for event in event_records),
            "confirmed_by_class": event_counts["confirmed"],
            "tentative_total": sum(event["status"] == "tentative" for event in event_records),
            "tentative_by_class": event_counts["tentative"],
            "interpretation": EVENT_INTERPRETATION,
        },
        "timing": {
            "processing_elapsed_seconds": elapsed_seconds,
            "model_inference_seconds": inference_seconds,
            "effective_inference_fps": (
                frames_inferred / inference_seconds if inference_seconds else 0.0
            ),
            "overall_processing_fps": (
                frames_inferred / elapsed_seconds if elapsed_seconds else 0.0
            ),
            "inference_fps_definition": "inferred frames divided by measured model.predict wall time",
        },
        "output_paths": {
            "run_directory": str(paths.run_dir),
            "run_manifest": str(paths.manifest),
            "summary": str(paths.summary),
            "events": str(paths.events),
            "frame_detections": str(paths.frame_detections),
            "annotated_video": (
                str(paths.annotated_video) if not no_annotated_video else None
            ),
            "completion": str(paths.completion),
        },
        "encoding_verification": encoding_verification,
        "warnings": warnings,
        "error": str(failure) if failure else None,
        "internal_test_accessed": False,
        "training_executed": False,
        "threshold_tuning_performed": False,
    }
    events_payload = {
        "schema_version": f"{SCHEMA_VERSION}.events",
        "run_id": run_id,
        "interpretation": EVENT_INTERPRETATION,
        "event_count": len(event_records),
        "events": event_records,
    }

    if not no_annotated_video and writer is not None and frames_inferred and failure is None:
        try:
            if info is None:
                raise VideoAnalysisError("Missing video metadata for output verification.")
            encoding_verification = verify_encoded_video(
                paths.annotated_video, info, frames_inferred, cv2
            )
            summary["encoding_verification"] = encoding_verification
        except Exception as exc:
            failure = exc
            status = "FAILED_TECHNICAL"
            summary["status"] = status
            summary["error"] = str(exc)
            final_reason = "annotated_video_verification_failure"
            summary["frames"]["stop_reason"] = final_reason

    _write_json(paths.events, events_payload)
    _write_json(paths.summary, summary)
    manifest.update(
        {
            "status": status,
            "finished_utc": summary["finished_utc"],
            "progress": {
                "frames_decoded": frames_decoded,
                "frames_inferred": frames_inferred,
                "raw_detection_observations": raw_total,
                "last_source_frame": last_frame_index,
                "finalized_events": len(event_records),
            },
            "stop_reason": final_reason,
            "error": str(failure) if failure else None,
        }
    )
    _write_json(paths.manifest, manifest)

    try:
        if status == "FINALIZING":
            # First reconcile closed artifacts while no persisted status claims success.
            artifact_hashes = _verify_persisted_outputs(
                paths, summary, not no_annotated_video
            )
            source_after = Path(metadata["absolute_path"]).stat()
            if (source_before.st_size, source_before.st_mtime_ns) != (
                source_after.st_size,
                source_after.st_mtime_ns,
            ):
                raise VideoAnalysisError("Source video changed during analysis.")
            _, final_model_sha = verify_frozen_checkpoint(config, root)
            if final_model_sha != checked["model"]["sha256"]:
                raise VideoAnalysisError("Frozen checkpoint identity changed during analysis.")

            # Publish COMPLETED into the summary only after the first reconciliation,
            # then re-read/reconcile/hash the final bytes and reverify the checkpoint.
            status = "COMPLETED"
            summary["status"] = status
            _write_json(paths.summary, summary)
            artifact_hashes = _verify_persisted_outputs(
                paths, summary, not no_annotated_video
            )
            _, final_model_sha = verify_frozen_checkpoint(config, root)
            if final_model_sha != checked["model"]["sha256"]:
                raise VideoAnalysisError("Frozen checkpoint identity changed during analysis.")

            manifest["status"] = status
            manifest["artifacts_sha256"] = artifact_hashes
            _write_json(paths.manifest, manifest)
            completion = {
                "schema_version": f"{SCHEMA_VERSION}.completion",
                "run_id": run_id,
                "status": "COMPLETED",
                "completed_utc": utc_now(),
                "run_manifest_sha256": sha256_file(paths.manifest),
                "artifacts_sha256": artifact_hashes,
                "frames_inferred": frames_inferred,
                "event_count": len(event_records),
                "model_sha256": checked["model"]["sha256"],
                "input_sha256": metadata["sha256"],
                "internal_test_accessed": False,
                "training_executed": False,
                "threshold_tuning_performed": False,
            }
            _write_json(paths.completion, completion)
            persisted_completion = json.loads(paths.completion.read_text(encoding="utf-8"))
            if (
                persisted_completion.get("status") != "COMPLETED"
                or persisted_completion.get("run_manifest_sha256")
                != sha256_file(paths.manifest)
            ):
                raise VideoAnalysisError("Completion receipt failed post-write verification.")
        else:
            _invalidate_completion_receipt(paths)

        paths.lock.unlink(missing_ok=True)
        if failure is not None:
            raise VideoAnalysisError(
                f"Video analysis failed: {failure}; partial diagnostics: {paths.run_dir}"
            ) from failure
        # Keep the successful return inside this protected region: a late
        # KeyboardInterrupt must invalidate an already-published receipt.
        return summary
    except KeyboardInterrupt as exc:
        try:
            _persist_late_terminal_state(
                paths,
                summary,
                manifest,
                status="INTERRUPTED",
                stop_reason="late_keyboard_interrupt",
                error=exc,
            )
        except Exception as state_error:
            try:
                paths.lock.unlink(missing_ok=True)
            except Exception as cleanup_error:
                raise VideoAnalysisError(
                    "Late KeyboardInterrupt receipt invalidation/state publication "
                    f"failed, and run-lock cleanup also failed: {cleanup_error}"
                ) from state_error
            raise VideoAnalysisError(
                "Late KeyboardInterrupt could not be recorded safely."
            ) from state_error
        try:
            paths.lock.unlink(missing_ok=True)
        except Exception as cleanup_error:
            _persist_late_terminal_state(
                paths,
                summary,
                manifest,
                status="FAILED_TECHNICAL",
                stop_reason="late_interrupt_cleanup_failure",
                error=cleanup_error,
            )
            raise VideoAnalysisError(
                "Late KeyboardInterrupt was recorded, but run-lock cleanup failed."
            ) from cleanup_error
        return summary
    except Exception as exc:
        try:
            _persist_late_terminal_state(
                paths,
                summary,
                manifest,
                status="FAILED_TECHNICAL",
                stop_reason="completion_verification_failure",
                error=exc,
            )
        finally:
            try:
                paths.lock.unlink(missing_ok=True)
            except Exception as cleanup_error:
                raise VideoAnalysisError(
                    "Video analysis failed and run-lock cleanup also failed: "
                    f"{cleanup_error}"
                ) from cleanup_error
        raise VideoAnalysisError(
            f"Video analysis failed: {exc}; partial diagnostics: {paths.run_dir}"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source road-video file")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Empty/nonexistent dedicated run directory"
    )
    parser.add_argument(
        "--max-frames", type=int, help="Safely limit inferred frames for smoke testing"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="Zero-based original source frame at which to start"
    )
    parser.add_argument(
        "--no-annotated-video", action="store_true", help="Write JSON artifacts without encoded video"
    )
    parser.add_argument(
        "--preflight", action="store_true", help="Validate only; do not load the model or run inference"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.preflight:
        try:
            report = preflight(args.input, args.output_dir)
            sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            return 0
        except (VideoAnalysisError, OSError, ValueError, ImportError) as exc:
            LOGGER.error("%s", exc)
            return 2

    stop = threading.Event()
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_stop(signum: int, frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    try:
        result = analyze_video(
            args.input,
            args.output_dir,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            no_annotated_video=args.no_annotated_video,
            stop_event=stop,
        )
        return 130 if result["status"] == "INTERRUPTED" else 0
    except (VideoAnalysisError, OSError, ValueError, ImportError) as exc:
        LOGGER.error("%s", exc)
        return 2
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
